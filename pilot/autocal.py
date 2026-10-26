"""Fully automatic camera -> satellite calibration from road structure.

The idea: a traffic camera and a satellite tile share one thing that is easy to
find in both -- the road.  So instead of matching appearance (hopeless across a
90-degree viewpoint change) we match *road structure*:

  1. observe()      watch the camera for a while.  Vehicle tracks give the
                    vanishing point of the road and a "where vehicles drive"
                    mask; a median background gives a vehicle-free lane-paint
                    mask.
  2. fetch_osm()    OpenStreetMap road centrelines around the camera, rasterised
                    into the satellite image as a "where vehicles can drive"
                    mask.
  3. search()       the camera pose is parameterised physically
                    (bearing, tilt, roll, focal, height, position).  The
                    vanishing point + the road's OSM bearing determine bearing
                    and tilt exactly, so only focal / height / position are
                    unknown -- and moving the camera on the ground plane is a
                    pure translation of the projection, so position is solved
                    for *all* offsets at once by normalised cross-correlation.
  4. polish()       Nelder-Mead on the full parameter set, then ECC on the
                    lane-paint masks for the last pixel.

Everything is plain functions; nothing here keeps state.
"""
import math
import os
import time

import cv2
import numpy as np
import requests

import pipeline

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]
UA = {"User-Agent": "FLTwin-pilot/0.1 (traffic digital twin research)"}

# nominal lane width (m) and default lane count per OSM highway class
ROAD_W = {"motorway": (3.7, 3), "trunk": (3.6, 2), "primary": (3.5, 2),
          "secondary": (3.4, 2), "tertiary": (3.2, 2), "residential": (3.0, 1),
          "unclassified": (3.0, 1), "service": (2.5, 1),
          "motorway_link": (3.7, 1), "trunk_link": (3.6, 1),
          "primary_link": (3.5, 1), "secondary_link": (3.4, 1),
          "tertiary_link": (3.2, 1)}
MAJOR = ("motorway", "trunk", "primary", "secondary", "tertiary")


# ---------------------------------------------------------------- camera side

def observe(clip_path, out_dir, stride=2, max_frames=250, log=print,
            progress=None):
    """Watch a recorded clip and extract the camera's road structure."""
    cap = cv2.VideoCapture(clip_path)
    frames = []
    fi = 0
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if fi % stride == 0:
            frames.append(fr)
        fi += 1
    cap.release()
    if len(frames) < 5:
        raise RuntimeError(f"clip has too few frames ({len(frames)})")
    h, w = frames[0].shape[:2]
    log(f"[obs] {len(frames)} frames of {w}x{h} to analyse")

    # --- vehicle-free background: per-pixel median of evenly spaced frames ---
    idx = np.linspace(0, len(frames) - 1, min(41, len(frames))).astype(int)
    bg = np.median(np.stack([frames[i] for i in idx]), axis=0).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "obs_bg.jpg"), bg)
    log(f"[obs] median background from {len(idx)} frames (vehicles removed)")

    # --- vehicle tracks ---
    model = pipeline.get_model(log)
    hist = {}
    cars = []
    drive = np.zeros((h, w), np.float32)
    for i, fr in enumerate(frames):
        res = model.track(fr, persist=True, verbose=False, conf=0.2, iou=0.5,
                          imgsz=pipeline.IMGSZ, classes=list(pipeline.VEHICLES),
                          tracker="bytetrack.yaml", device=pipeline.device())[0]
        b = res.boxes
        if b is not None and b.id is not None:
            xyxy = b.xyxy.cpu().numpy()
            ids = b.id.cpu().numpy().astype(int)
            cls = b.cls.cpu().numpy().astype(int)
            for box, tid, cl in zip(xyxy, ids, cls):
                fx, fy = (box[0] + box[2]) / 2, box[3]
                hist.setdefault(int(tid), []).append((fx, fy))
                r = max(3, int(0.30 * (box[2] - box[0])))
                cv2.circle(drive, (int(fx), int(fy)), r, 1.0, -1)
                # a car's rear track width is a metric ruler lying on the road
                if int(cl) == 2 and (box[2] - box[0]) > 18:
                    cars.append([float(box[0]), float(box[3]),
                                 float(box[2]), float(box[3])])
        if progress and i % 10 == 0:
            progress(i, len(frames))
    log(f"[obs] {len(hist)} vehicle tracks")

    # --- vanishing point from the motion lines of long tracks ---
    lines, mids = [], []
    for pts in hist.values():
        if len(pts) < 4:
            continue
        p = np.float32(pts)
        if np.linalg.norm(p[-1] - p[0]) < 25:      # parked / jittery
            continue
        # total least squares line through the trajectory
        c = p.mean(axis=0)
        u, s, vt = np.linalg.svd(p - c)
        d = vt[0]
        n = np.array([-d[1], d[0]])
        lines.append(np.array([n[0], n[1], -n @ c]))   # a*x + b*y + c = 0
        mids.append((c, p[-1] - p[0]))
    vp, n_in = _ransac_vp(lines)
    if vp is None:
        log("[obs] not enough vehicle motion for a vanishing point; "
            "falling back to lane-marking lines")
        vp, n_in = _vp_from_paint(bg, log)
    if vp is None:
        raise RuntimeError("could not find the road vanishing point "
                           "(no vehicle motion and no clear lane markings)")

    # do vehicles recede from the camera (move toward the vanishing point)?
    votes = 0
    for c, dvec in mids:
        toward = np.array([vp[0], vp[1]]) - c
        votes += 1 if float(toward @ dvec) > 0 else -1
    recede = votes >= 0
    log(f"[obs] vanishing point ({vp[0]:.0f}, {vp[1]:.0f}) from {n_in} motion "
        f"lines; traffic {'recedes from' if recede else 'approaches'} the camera")

    # --- drivable-area and lane-paint masks ---
    if drive.max() > 0:
        drive = np.clip(drive / max(1.0, np.percentile(drive[drive > 0], 60)),
                        0, 1)
    drive = cv2.GaussianBlur(drive, (0, 0), 5)
    horizon = float(vp[1])
    road = road_surface(bg, drive, horizon, log)
    road_near = (cv2.dilate((road > 0.5).astype(np.uint8),
                            np.ones((9, 9), np.uint8)) * 255)
    paint = pipeline._paint_mask(bg, road_near)
    cv2.imwrite(os.path.join(out_dir, "obs_drive.png"),
                (drive * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, "obs_road.png"),
                (road * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, "obs_paint.png"),
                (paint * 255).astype(np.uint8))
    out = {"bg": bg, "drive": drive, "road": road, "paint": paint,
           "vp": [float(vp[0]), float(vp[1])], "horizon": horizon,
           "recede": bool(recede), "n_tracks": len(hist),
           "cars": np.float32(cars) if cars else np.zeros((0, 4), np.float32),
           "vp_lines": int(n_in), "w": w, "h": h}
    out["vp"], _ = refine_vp(paint, out["vp"], horizon, log)
    out["horizon"] = float(out["vp"][1])
    out["rays"] = lane_rays(paint, out["vp"], out["horizon"], log)
    log(f"[obs] {len(cars)} car footprints kept as a metric scale reference")
    sols = solve_intrinsics(out, log)
    out["height_car"] = sols[0].get("height_cars") if sols else None
    return out


def road_surface(bg, drive, horizon_y, log=print):
    """Segment the paved surface, using the vehicle tracks as free labels.

    Wherever a vehicle has driven is definitely road, so the colour statistics
    of those pixels define 'pavement' for this camera -- no training, no
    hand-tuned thresholds, and it adapts to asphalt, concrete, wet or dry.
    """
    h, w = bg.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(bg, (0, 0), 2), cv2.COLOR_BGR2Lab)
    lab = lab.astype(np.float32)
    below = np.zeros((h, w), np.uint8)
    below[max(0, int(horizon_y) + 8):, :] = 1
    seed = ((drive > 0.25) & (below > 0))
    if seed.sum() < 400:                      # quiet road: seed near the bottom
        seed = np.zeros((h, w), bool)
        seed[int(h * 0.88):, int(w * 0.25):int(w * 0.75)] = True
        log("[obs] few vehicle tracks; seeding pavement colour from the "
            "bottom of the frame")
    s = lab[seed]
    mu_ab, L = s[:, 1:].mean(axis=0), s[:, 0]
    cov = np.cov(s[:, 1:].T) + np.eye(2) * 4.0
    inv = np.linalg.inv(cov)
    d = lab[:, :, 1:] - mu_ab
    maha = np.einsum("ijk,kl,ijl->ij", d, inv, d)
    lo, hi = np.percentile(L, 2) - 35, np.percentile(L, 98) + 55
    cand = ((maha < 12.0) & (lab[:, :, 0] > lo) & (lab[:, :, 0] < hi)
            & (below > 0)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    keep = np.zeros((h, w), np.float32)
    seed_lbl = set(np.unique(lbl[seed])) - {0}
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.004 * h * w:
            continue
        touches_bottom = lbl[h - 1].__contains__(i) if h else False
        if i in seed_lbl or touches_bottom:
            keep[lbl == i] = 1.0
    if keep.sum() < 0.02 * h * w:             # segmentation failed -> use tracks
        log("[obs] pavement segmentation weak; falling back to the track mask")
        keep = (drive > 0.15).astype(np.float32)
    log(f"[obs] pavement mask covers {100 * keep.mean():.0f}% of the frame")
    return cv2.GaussianBlur(keep, (0, 0), 2)


def _ransac_vp(lines, iters=600, tol=0.02):
    """Vanishing point = the point closest to the most motion lines."""
    if len(lines) < 3:
        return None, 0
    L = np.float32(lines)
    L = L / (np.linalg.norm(L[:, :2], axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(0)
    best, best_in = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(L), 2, replace=False)
        v = np.cross(L[i], L[j])
        if abs(v[2]) < 1e-6:
            continue
        v = v / v[2]
        d = np.abs(L @ v) / (np.linalg.norm(v[:2]) + 1e-9)
        n = int((d < tol * np.linalg.norm(v[:2])).sum())
        if n > best_in:
            best, best_in = v, n
    if best is None or best_in < 3:
        return None, 0
    d = np.abs(L @ best) / (np.linalg.norm(best[:2]) + 1e-9)
    inl = L[d < tol * np.linalg.norm(best[:2])]
    _, _, vt = np.linalg.svd(inl)          # least-squares intersection
    v = vt[-1]
    return (v / v[2])[:2], best_in


def _vp_from_paint(bg, log=print):
    """Fallback: vanishing point from long lane-marking segments."""
    m = (pipeline._paint_mask(bg) * 255).astype(np.uint8)
    m = cv2.threshold(m, 40, 255, cv2.THRESH_BINARY)[1]
    seg = cv2.HoughLinesP(m, 1, np.pi / 360, 40, minLineLength=40,
                          maxLineGap=12)
    if seg is None or len(seg) == 0:
        return None, 0
    lines = []
    for x1, y1, x2, y2 in np.asarray(seg).reshape(-1, 4):
        if abs(y2 - y1) < 4:               # near-horizontal: not a lane line
            continue
        n = np.array([-(y2 - y1), x2 - x1], np.float64)
        c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        lines.append(np.array([n[0], n[1], -n @ c]))
    log(f"[obs] {len(lines)} lane-marking segments for the vanishing point")
    return _ransac_vp(lines, tol=0.03)


# ---------------------------------------------------------- OpenStreetMap side

def fetch_osm(lat, lon, radius_m=900, log=print):
    q = ("[out:json][timeout:60];"
         f"way(around:{int(radius_m)},{lat:.6f},{lon:.6f})[highway];"
         "out geom;")
    last = None
    for attempt in range(2):
        for url in OVERPASS_URLS:
            try:
                r = requests.post(url, data={"data": q}, headers=UA, timeout=120)
                if r.status_code == 200 and r.json().get("elements"):
                    els = r.json()["elements"]
                    log(f"[osm] {len(els)} ways within {radius_m} m "
                        f"({url.split('/')[2]})")
                    return els
                last = f"HTTP {r.status_code}"
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
        time.sleep(3)
    raise RuntimeError(f"OpenStreetMap query failed ({last})")


def rebase_H(H, georef_from, georef_to):
    """Re-express a camera->satellite homography in another tile mosaic."""
    s = 2.0 ** (georef_to["zoom"] - georef_from["zoom"])
    T = np.array([[s, 0, georef_from["gx0"] * s - georef_to["gx0"]],
                  [0, s, georef_from["gy0"] * s - georef_to["gy0"]],
                  [0, 0, 1.0]])
    Hn = T @ np.float64(H)
    return Hn / Hn[2, 2]


def latlon_to_satpx(georef, lat, lon):
    n = 2 ** georef["zoom"] * 256
    gx = (lon + 180.0) / 360.0 * n
    gy = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return gx - georef["gx0"], gy - georef["gy0"]


def roads_to_pixels(els, georef):
    """OSM ways -> polylines in satellite pixels, with width and travel sense."""
    out = []
    for e in els:
        if e.get("type") != "way" or "geometry" not in e:
            continue
        t = e.get("tags", {})
        hw = t.get("highway")
        if hw not in ROAD_W:
            continue
        pts = np.float32([latlon_to_satpx(georef, g["lat"], g["lon"])
                          for g in e["geometry"]])
        if len(pts) < 2:
            continue
        lw, dl = ROAD_W[hw]
        try:
            lanes = int(str(t.get("lanes", dl)).split(";")[0])
        except ValueError:
            lanes = dl
        width = lanes * lw + (4.0 if hw.startswith("motorway") else 1.5)
        out.append({"pts": pts, "highway": hw, "width_m": width,
                    "ref": (t.get("ref") or "").strip(),
                    "name": (t.get("name") or "").strip(),
                    "oneway": t.get("oneway") in ("yes", "1", "true"),
                    "major": hw in MAJOR})
    return out


def road_raster(roads, shape, mpp):
    """Binary 'vehicles can be here' mask from the OSM road network."""
    m = np.zeros(shape, np.float32)
    for r in roads:
        th = max(2, int(round(r["width_m"] / mpp)))
        cv2.polylines(m, [np.int32(r["pts"])], False, 1.0, th, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), 1.5)


def sat_pavement(sat_img, osm_r, log=print):
    """Segment paved surface in the satellite tile, seeded by the OSM roads.

    The OSM raster alone is a constant-width ribbon, which cannot tell a
    5-lane carriageway from a 3-lane one -- so it cannot pin down the camera's
    scale.  Real pavement can: it carries the true widths, the shoulders and
    the shape of every gore and ramp.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(sat_img, (0, 0), 1.5),
                       cv2.COLOR_BGR2Lab).astype(np.float32)
    seed = cv2.erode((osm_r > 0.6).astype(np.uint8),
                     np.ones((5, 5), np.uint8)) > 0
    if seed.sum() < 500:
        seed = osm_r > 0.5
    if seed.sum() < 200:
        return (osm_r > 0.3).astype(np.float32)
    s = lab[seed]
    mu = s[:, 1:].mean(axis=0)
    cov = np.cov(s[:, 1:].T) + np.eye(2) * 3.0
    inv = np.linalg.inv(cov)
    d = lab[:, :, 1:] - mu
    maha = np.einsum("ijk,kl,ijl->ij", d, inv, d)
    lo, hi = np.percentile(s[:, 0], 2) - 30, np.percentile(s[:, 0], 98) + 45
    m = ((maha < 9.0) & (lab[:, :, 0] > lo) & (lab[:, :, 0] < hi))
    # keep only the road corridor: rooftops and tree canopy are grey enough to
    # pass a colour test, but they are never next to the carriageway we want
    near = cv2.dilate((osm_r > 0.2).astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=1)
    near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (61, 61)))
    m = (m & (near > 0))
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    log(f"[cal] satellite pavement mask covers {100 * m.mean():.0f}% of the tile")
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)


def _norm_ref(s):
    return "".join(ch for ch in (s or "").upper() if ch.isalnum())


def plan_area(els, lat, lon, roadway, direction, log=print):
    """Pick where to centre the satellite search.

    FL511's listed coordinates are often far from the real mount, so instead of
    trusting them we snap to the nearest point of the road the camera is
    supposed to be watching.  Returns (lat, lon) to centre on.
    """
    want = _norm_ref(roadway)
    dir_bearing = {"NORTHBOUND": 0, "EASTBOUND": 90, "SOUTHBOUND": 180,
                   "WESTBOUND": 270}.get((direction or "").upper())
    best, best_s = None, -1e9
    for e in els:
        if e.get("type") != "way" or "geometry" not in e:
            continue
        t = e.get("tags", {})
        if t.get("highway") not in MAJOR:
            continue
        ref = _norm_ref(t.get("ref")) or _norm_ref(t.get("name"))
        g = np.float64([[p["lat"], p["lon"]] for p in e["geometry"]])
        d = np.hypot((g[:, 0] - lat) * 111320,
                     (g[:, 1] - lon) * 111320 * math.cos(math.radians(lat)))
        k = int(np.argmin(d))
        s = -float(d[k]) / 300.0
        if want and ref:
            s += 6.0 if ref == want else (3.0 if want in ref or ref in want
                                          else 0.0)
        if t.get("highway") == "motorway":
            s += 1.0
        if dir_bearing is not None and len(g) > 2:
            k0, k1 = max(0, k - 2), min(len(g) - 1, k + 2)
            v = g[k1] - g[k0]
            b = math.degrees(math.atan2(
                v[1] * math.cos(math.radians(lat)), v[0])) % 360
            dd = min(abs((b - dir_bearing + 180) % 360 - 180),
                     abs((b + 180 - dir_bearing + 180) % 360 - 180))
            s += 1.5 * max(0.0, 1.0 - dd / 60.0)
        if s > best_s:
            best_s, best = s, (float(g[k, 0]), float(g[k, 1]),
                               t.get("ref") or t.get("name") or "", float(d[k]))
    if best is None:
        log("[cal] no matching road in OSM; centring on the listed coordinates")
        return lat, lon
    log(f"[cal] centring the search on {best[2] or 'the nearest major road'}, "
        f"{best[3]:.0f} m from the coordinates FL511 lists")
    return best[0], best[1]


def candidate_roads(roads, cam_px, roadway, direction, max_dist_px):
    """Roads the camera plausibly watches, best guess first."""
    want = _norm_ref(roadway)
    dir_bearing = {"NORTHBOUND": 0, "EASTBOUND": 90, "SOUTHBOUND": 180,
                   "WESTBOUND": 270}.get((direction or "").upper())
    cands = []
    for r in roads:
        if not r["major"]:
            continue
        d = np.linalg.norm(r["pts"] - np.float32(cam_px), axis=1)
        k = int(np.argmin(d))
        if d[k] > max_dist_px:
            continue
        k0, k1 = max(0, k - 2), min(len(r["pts"]) - 1, k + 2)
        if k1 == k0:
            continue
        v = r["pts"][k1] - r["pts"][k0]
        # satellite pixels: +x east, +y south -> compass bearing
        bearing = math.degrees(math.atan2(v[0], -v[1])) % 360
        ref = _norm_ref(r["ref"]) or _norm_ref(r["name"])
        score = 0.0
        if want and ref:
            score += 3.0 if ref == want else (1.5 if want in ref or ref in want
                                              else 0.0)
        if dir_bearing is not None:
            dd = abs((bearing - dir_bearing + 180) % 360 - 180)
            score += 2.0 * max(0.0, 1.0 - dd / 60.0)
        score += 1.0 if r["highway"] == "motorway" else 0.0
        score -= float(d[k]) / max_dist_px
        cands.append({"bearing": bearing, "score": score, "dist_px": float(d[k]),
                      "label": f"{r['ref'] or r['name'] or r['highway']}",
                      "highway": r["highway"], "oneway": r["oneway"]})
    cands.sort(key=lambda c: -c["score"])
    # merge near-duplicate bearings, keeping the best-scoring representative
    keep = []
    for c in cands:
        if all(abs((c["bearing"] - k["bearing"] + 180) % 360 - 180) > 12
               for k in keep):
            keep.append(c)
    return keep[:4]


# ------------------------------------------------------------------- geometry

def rot(bearing_deg, tilt_deg, roll_deg=0.0):
    """World (X=east, Y=south, Z=up) -> camera (x right, y down, z forward)."""
    b, t = math.radians(bearing_deg), math.radians(tilt_deg)
    cb, sb, ct, st = math.cos(b), math.sin(b), math.cos(t), math.sin(t)
    R = np.array([[cb, sb, 0.0],
                  [-sb * st, cb * st, -ct],
                  [sb * ct, -cb * ct, -st]])
    if roll_deg:
        r = math.radians(roll_deg)
        cr, sr = math.cos(r), math.sin(r)
        R = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.0]]) @ R
    return R


def pan_tilt_from_vp(vp, f, cx, cy, road_bearing, roll_deg=0.0):
    """The road's vanishing point + its compass bearing fix pan and tilt."""
    q = np.array([(vp[0] - cx) / f, (vp[1] - cy) / f, 1.0])
    if roll_deg:
        r = math.radians(-roll_deg)
        cr, sr = math.cos(r), math.sin(r)
        q = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.0]]) @ q
    q = q / np.linalg.norm(q)
    a, b, c = q
    delta = math.degrees(math.asin(float(np.clip(a, -1, 1))))
    tilt = math.degrees(math.atan2(-b, c))
    return road_bearing - delta, tilt


def H_img_to_world(f, cx, cy, bearing, tilt, roll, C):
    """Homography from image pixels to ground-plane metres (Z=0)."""
    R = rot(bearing, tilt, roll)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.column_stack([R[:, 0], R[:, 1], -R @ np.asarray(C, float)])
    return np.linalg.inv(K @ M)


def _fov_to_f(fov_deg, w):
    return (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


# ------------------------------------------- self-calibration from lane paint
LANE_M = 3.6576          # US standard lane width, 12 ft


def _angle_hist(paint_pts, wts, vp, nbins=900):
    """Weighted histogram of the angle of each paint pixel around `vp`.

    A lane line is radial from the vanishing point, so it collapses into one
    sharp spike here -- dashes included, which is why this beats running a line
    detector over broken markings.

    The bin range follows the data rather than being fixed, so that a candidate
    vanishing point far off in the distance (where every angle collapses toward
    one value) does not look artificially peaked.
    """
    d = paint_pts - np.float64(vp)
    r = np.hypot(d[:, 0], d[:, 1])
    ok = r > 45
    if ok.sum() < 200:
        return None, None
    th = np.arctan2(d[ok, 1], d[ok, 0])
    lo, hi = float(th.min()), float(th.max())
    if hi - lo < 1e-4:
        return None, None
    hist, edges = np.histogram(th, bins=nbins, range=(lo, hi),
                               weights=wts[ok] / r[ok])
    hist = cv2.GaussianBlur(hist.reshape(-1, 1).astype(np.float32),
                            (1, 7), 0).ravel()
    return hist, (edges[:-1] + edges[1:]) / 2


def refine_vp(paint, vp0, horizon, log=print):
    """Sharpen the vanishing point using the lane paint.

    The right vanishing point is the one that makes every lane line radial,
    which is exactly the one that makes the angle histogram most peaked.
    """
    ys, xs = np.nonzero(paint > 0.12)
    keep = ys > horizon + 12
    if keep.sum() < 300:
        return vp0, 0.0
    pts = np.float64(np.column_stack([xs[keep], ys[keep]]))
    wts = paint[ys[keep], xs[keep]].astype(np.float64)

    def sharp(v):
        h, _ = _angle_hist(pts, wts, v)
        if h is None:
            return -1.0
        s = h.sum()
        return float((h ** 2).sum() / (s * s + 1e-12))

    # bounded search around the motion-derived point: it is already close, and
    # an unbounded hill-climb would happily wander off to infinity
    v0 = np.float64(vp0)
    best, bs = v0, sharp(v0)
    for span, step in ((80, 16), (20, 4), (5, 1)):
        cand = [(dx, dy) for dx in range(-span, span + 1, step)
                for dy in range(-span, span + 1, step)]
        for dx, dy in cand:
            v = best + np.float64([dx, dy])
            if np.linalg.norm(v - v0) > 110:
                continue
            s = sharp(v)
            if s > bs:
                best, bs = v, s
    log(f"[cal] vanishing point sharpened with lane paint: "
        f"({vp0[0]:.0f}, {vp0[1]:.0f}) -> ({best[0]:.0f}, {best[1]:.0f})")
    return [float(best[0]), float(best[1])], bs


def lane_rays(paint, vp, horizon, log=print, max_lines=16):
    """Lane boundary lines, as angles leaving the road's vanishing point."""
    from scipy.signal import find_peaks
    ys, xs = np.nonzero(paint > 0.12)
    keep = ys > horizon + 12
    if keep.sum() < 300:
        return []
    pts = np.float64(np.column_stack([xs[keep], ys[keep]]))
    wts = paint[ys[keep], xs[keep]].astype(np.float64)
    hist, centres = _angle_hist(pts, wts, vp)
    if hist is None:
        return []
    pk, props = find_peaks(hist, height=0.06 * hist.max(),
                           distance=6, prominence=0.04 * hist.max())
    if len(pk) == 0:
        return []
    order = np.argsort(-props["prominences"])[:max_lines]
    rays = sorted((float(centres[pk[i]]), float(props["prominences"][i]))
                  for i in order)
    log(f"[cal] {len(rays)} lane lines recovered from the paint mask")
    return rays


def _grid_fit(offsets, weights):
    """Fit lateral offsets to a regular lane grid o0 + k*width.

    Robust by design: some detected lines are not lane edges at all (barriers,
    a light pole, the far carriageway across the median) and a wide stripe can
    be found twice, so this votes for the width with the most inliers instead
    of least-squares over everything.

    Returns (width, relative residual, n_inliers).
    """
    o = np.asarray(offsets, float)
    wt = np.asarray(weights, float)
    if len(o) < 4:
        return None, 9.0, 0
    idx = np.argsort(o)
    o, wt = o[idx], wt[idx]
    diffs = np.diff(o)
    diffs = diffs[diffs > 1e-9]
    if len(diffs) == 0:
        return None, 9.0, 0
    # merge lines that are really the same painted stripe
    base = float(np.median(diffs))
    merged_o, merged_w = [o[0]], [wt[0]]
    for v, ww in zip(o[1:], wt[1:]):
        if v - merged_o[-1] < 0.30 * base:
            merged_w[-1] += ww
        else:
            merged_o.append(v)
            merged_w.append(ww)
    o, wt = np.array(merged_o), np.array(merged_w)
    if len(o) < 4:
        return None, 9.0, 0
    diffs = np.diff(o)
    base = float(np.median(diffs[diffs > 1e-9]))
    best = (None, 9.0, 0)
    for w in np.linspace(base * 0.45, base * 2.2, 140):
        for anchor in o:
            k = np.round((o - anchor) / w)
            resid = o - (anchor + k * w)
            inl = np.abs(resid) < 0.16 * w
            n = int(inl.sum())
            if n < 4:
                continue
            A = np.column_stack([np.ones(n), k[inl]])
            sol, *_ = np.linalg.lstsq(A * np.sqrt(wt[inl])[:, None],
                                      o[inl] * np.sqrt(wt[inl]), rcond=None)
            if abs(sol[1]) < 1e-9:
                continue
            r = float(np.sqrt(np.average((A @ sol - o[inl]) ** 2,
                                         weights=wt[inl])) / abs(sol[1]))
            # prefer more inliers, then a tighter fit
            if (n, -r) > (best[2], -best[1]):
                best = (abs(float(sol[1])), r, n)
    return best


CAR_W = 1.85          # mean passenger-car track width (m)


def height_from_cars(obs, Hiw, log=print):
    """Camera height from the apparent width of tracked cars.

    Independent of the lane markings, which matters because a lane grid can
    lock onto half-lane spacing and be wrong by a clean factor of two.
    """
    cars = obs.get("cars")
    if cars is None or len(cars) < 15:
        return None, 0
    a = cv2.perspectiveTransform(cars[:, :2].reshape(-1, 1, 2),
                                 np.float64(Hiw)).reshape(-1, 2)
    b = cv2.perspectiveTransform(cars[:, 2:].reshape(-1, 1, 2),
                                 np.float64(Hiw)).reshape(-1, 2)
    wid = np.linalg.norm(a - b, axis=1)          # in "height = 1 m" units
    wid = wid[np.isfinite(wid) & (wid > 0)]
    if len(wid) < 15:
        return None, 0
    lo, hi = np.percentile(wid, [20, 80])        # drop trucks and bad boxes
    core = wid[(wid >= lo) & (wid <= hi)]
    if len(core) < 8:
        return None, 0
    return float(CAR_W / np.median(core)), len(core)


def solve_intrinsics(obs, log=print, fov_range=(12.0, 70.0), n_fov=120):
    """Focal length, roll and camera height from the lane markings alone.

    Equally spaced parallel lines plus their vanishing point pin down the
    focal length: only the true focal makes the projected lane lines come out
    evenly spaced.  The known 12 ft lane width then fixes the camera height.
    """
    w_img, h_img = obs["w"], obs["h"]
    cx, cy = w_img / 2.0, h_img / 2.0
    rays = obs.get("rays") or []
    if len(rays) < 3:
        log("[cal] too few lane lines to self-calibrate; "
            "falling back to searching focal length and height")
        return []
    v = np.float64(obs["vp"])
    y_ref = obs["horizon"] + 0.72 * (h_img - obs["horizon"])
    out = []
    for fov in np.linspace(fov_range[0], fov_range[1], n_fov):
        f = _fov_to_f(float(fov), w_img)
        for roll in (-6.0, -3.0, 0.0, 3.0, 6.0):
            pan, tilt = pan_tilt_from_vp(obs["vp"], f, cx, cy, 0.0, roll)
            if not (0.5 < tilt < 60.0):
                continue
            Hiw = H_img_to_world(f, cx, cy, pan, tilt, roll, (0, 0, 1.0))
            pts, wts = [], []
            for th, wgt in rays:
                dirv = np.array([math.cos(th), math.sin(th)])
                if abs(dirv[1]) < 1e-6:
                    continue
                t = (y_ref - v[1]) / dirv[1]
                if t <= 0:
                    continue
                p = v + t * dirv
                if not (-w_img < p[0] < 2 * w_img):
                    continue
                pts.append(p)
                wts.append(wgt)
            if len(pts) < 3:
                continue
            g = cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2),
                                         np.float64(Hiw)).reshape(-1, 2)
            # lateral offset across the road (road runs along bearing 0 here)
            width, res, n_in = _grid_fit(g[:, 0], np.float64(wts))
            h_car, n_car = height_from_cars(obs, Hiw)
            height, src = None, ""
            if width and width > 0:
                h_lane = LANE_M / width
                if h_car:
                    # the lane grid can be out by a whole-number factor; pick
                    # the multiple that agrees with the cars
                    k = max(1, min(3, int(round(h_lane / h_car))))
                    height, src = h_lane / k, ("lane paint" if k == 1
                                               else f"lane paint /{k}")
                else:
                    height, src = h_lane, "lane paint"
            elif h_car:
                height, src, res, n_in = h_car, "car widths", 0.5, 0
            if height is None or not (7.0 <= height <= 26.0):
                continue
            out.append({"fov": float(fov), "roll": roll, "height": height,
                        "tilt": tilt, "lane_residual": float(res),
                        "lane_inliers": n_in, "scale_src": src,
                        "height_cars": h_car, "n_cars": n_car})
    out.sort(key=lambda r: (-r["lane_inliers"], r["lane_residual"]))
    if out:
        b = out[0]
        log(f"[cal] self-calibrated: fov {b['fov']:.1f}°, roll {b['roll']:.0f}°,"
            f" tilt {b['tilt']:.1f}°, mast height {b['height']:.1f} m "
            f"(scale from {b['scale_src']}"
            + (f", cars say {b['height_cars']:.1f} m from {b['n_cars']} boxes"
               if b.get("height_cars") else "") + ")")
    else:
        log("[cal] self-calibration found no physically plausible camera")
    return out


def ground_project(mask, Hiw, res, half_m, max_range_m):
    """Warp a camera-space mask onto a metric ground raster centred on the
    camera.  Returns (raster, valid-footprint raster)."""
    n = int(2 * half_m / res)
    S = np.array([[1 / res, 0, n / 2.0], [0, 1 / res, n / 2.0], [0, 0, 1.0]])
    Hs = S @ Hiw
    h, w = mask.shape[:2]
    # drop everything at or above the horizon (denominator changes sign there)
    yy, xx = np.mgrid[0:h, 0:w]
    den = Hiw[2, 0] * xx + Hiw[2, 1] * yy + Hiw[2, 2]
    ref = Hiw[2, 0] * (w / 2) + Hiw[2, 1] * (h - 1) + Hiw[2, 2]
    below = ((den * ref) > 0).astype(np.float32)
    out = cv2.warpPerspective(mask * below, Hs, (n, n), flags=cv2.INTER_LINEAR)
    fp = cv2.warpPerspective(below, Hs, (n, n), flags=cv2.INTER_LINEAR)
    yy2, xx2 = np.mgrid[0:n, 0:n]
    far = ((xx2 - n / 2) ** 2 + (yy2 - n / 2) ** 2) > (max_range_m / res) ** 2
    out[far] = 0
    fp[far] = 0
    return out, (fp > 0.5).astype(np.float32)


def masked_ncc(tmpl, tmask, image):
    """Normalised cross-correlation of a masked template over an image.

    Returns a map whose (r, c) entry scores placing the template's top-left
    corner at (r, c) in `image` (image must be at least as large as tmpl).
    """
    from scipy.signal import fftconvolve
    n = float(tmask.sum())
    if n < 50:
        return None
    tm = tmpl * tmask
    t0 = tm - tmask * (tm.sum() / n)          # zero-mean inside the mask
    den_t = math.sqrt(float((t0 ** 2).sum())) + 1e-9
    flip = lambda a: a[::-1, ::-1]
    num = fftconvolve(image, flip(t0), mode="valid")
    s1 = fftconvolve(image, flip(tmask), mode="valid")
    s2 = fftconvolve(image ** 2, flip(tmask), mode="valid")
    var = np.maximum(s2 - s1 ** 2 / n, 1e-6)
    return num / (den_t * np.sqrt(var))


# --------------------------------------------------------------------- search

def search(obs, sat_img, georef, roads, cam_px, roadway, direction,
           res=2.0, max_range_m=420.0, road_dist_m=800.0, log=print,
           progress=None):
    """Coarse pose search.  Returns a list of candidate solutions, best first.

    `road_dist_m` is deliberately generous: FL511's listed camera coordinates
    are often hundreds of metres from the real mount, and the candidate road is
    only used for its *bearing* -- the position comes out of the correlation.
    """
    h_img, w_img = obs["h"], obs["w"]
    cx, cy = w_img / 2.0, h_img / 2.0
    mpp = georef["mpp"]
    sat_h, sat_w = sat_img.shape[:2]

    cands = candidate_roads(roads, cam_px, roadway, direction,
                            max_dist_px=road_dist_m / mpp)
    if not cands:
        raise RuntimeError("no major road found near the camera in OSM")
    log("[cal] candidate roads: " + ", ".join(
        f"{c['label']} ({c['bearing']:.0f}°)" for c in cands))

    # target rasters at `res` metres per pixel.  The correlation target mixes
    # the OSM ribbon (right topology, crude width) with segmented pavement
    # (true widths, shoulders, gores) -- the width is what fixes the scale.
    scale = mpp / res
    tw, th = int(sat_w * scale), int(sat_h * scale)
    osm_full = road_raster(roads, (sat_h, sat_w), mpp)
    pav_full = sat_pavement(sat_img, osm_full, log)
    osm_r = cv2.resize(0.45 * osm_full + 0.55 * pav_full, (tw, th))
    paint_r = cv2.resize(pipeline._paint_mask(sat_img) *
                         (osm_full > 0.2).astype(np.float32), (tw, th))
    half_m = max_range_m + 40
    results = []
    h_car = obs.get("height_car")
    if h_car:
        heights = [round(h_car * f, 1) for f in (0.8, 1.0, 1.25)]
        log(f"[cal] tracked cars put the camera {h_car:.1f} m above the road "
            "surface — that sets the scale, and therefore the speeds")
    else:
        heights = [6, 9, 12, 16, 20]
    combos = [(fov, ht) for fov in (18, 24, 31, 40, 52) for ht in heights]
    total = len(cands) * len(combos) * 3
    step = 0
    for ci, c in enumerate(cands):
        # bearing of travel *away* from the camera along this road
        base = c["bearing"] if obs["recede"] else c["bearing"] + 180.0
        for db in (-8.0, 0.0, 8.0):
            road_b = base + db
            for fov, ht in combos:
                step += 1
                if progress and step % 5 == 0:
                    progress(step, total)
                f = _fov_to_f(fov, w_img)
                pan, tilt = pan_tilt_from_vp(obs["vp"], f, cx, cy, road_b)
                if not (1.0 < tilt < 60.0):
                    continue
                Hiw = H_img_to_world(f, cx, cy, pan, tilt, 0.0, (0, 0, ht))
                drv, fp = ground_project(obs["road"], Hiw, res, half_m,
                                         max_range_m)
                if fp.sum() < 200:
                    continue
                pnt, _ = ground_project(obs["paint"], Hiw, res, half_m,
                                        max_range_m)
                n = drv.shape[0]
                if n >= th or n >= tw:
                    continue
                m1 = masked_ncc(drv, fp, osm_r)
                if m1 is None:
                    continue
                m2 = masked_ncc(pnt, fp, paint_r)
                sc = m1 + 0.4 * (m2 if m2 is not None else 0)
                k = int(np.argmax(sc))
                r0, c0 = divmod(k, sc.shape[1])
                # camera sits at the centre of the projected raster
                cam_x = (c0 + n / 2.0) / scale
                cam_y = (r0 + n / 2.0) / scale
                results.append({"score": float(sc[r0, c0]), "fov": fov,
                                "height": ht, "road_bearing": road_b,
                                "pan": pan, "tilt": tilt, "roll": 0.0,
                                "cam_px": [cam_x, cam_y],
                                "road": c["label"], "road_i": ci})
    if not results:
        raise RuntimeError("pose search found no valid camera geometry")
    results.sort(key=lambda r: -r["score"])
    log(f"[cal] coarse search: {len(results)} poses, best score "
        f"{results[0]['score']:.3f} on {results[0]['road']} "
        f"(fov {results[0]['fov']}°, height {results[0]['height']} m)")
    return results


def _score_pose(p, obs, osm_r, paint_r, mpp, res, max_range_m):
    """Symmetric point-sampled score of one pose (fast; used by the optimiser).

    Both directions are measured, because a one-way score is trivially gamed:
    shrinking the projection onto a wide patch of tarmac scores a perfect
    'all my road pixels are on a road' while explaining almost nothing.
      forward  - camera pavement that lands on an OSM road
      reverse  - OSM road inside the view that the camera calls pavement
    Their harmonic mean punishes both kinds of cheating.
    """
    h_img, w_img = obs["h"], obs["w"]
    cx, cy = w_img / 2.0, h_img / 2.0
    f = _fov_to_f(p["fov"], w_img)
    pan, tilt = pan_tilt_from_vp(obs["vp"], f, cx, cy, p["road_bearing"],
                                 p["roll"])
    if not (1.0 < tilt < 60.0):
        return -9.0
    C = (p["cam_px"][0] * mpp, p["cam_px"][1] * mpp, p["height"])
    Hiw = H_img_to_world(f, cx, cy, pan, tilt, p["roll"], C)
    cam_o = np.float32([C[0] / res, C[1] / res])

    def fwd(pts, img):
        if pts is None or len(pts) == 0:
            return None
        wp = cv2.perspectiveTransform(pts.reshape(-1, 1, 2),
                                      np.float64(Hiw)).reshape(-1, 2) / res
        d = np.linalg.norm(wp - cam_o, axis=1)
        ok = ((wp[:, 0] >= 0) & (wp[:, 1] >= 0) & (wp[:, 0] < img.shape[1] - 1)
              & (wp[:, 1] < img.shape[0] - 1) & (d < max_range_m / res))
        if ok.sum() < 60:
            return None
        v = img[wp[ok, 1].astype(int), wp[ok, 0].astype(int)]
        return float(v.mean()) * (ok.sum() / len(pts))

    a = fwd(obs["_road_pts"], osm_r)
    if a is None:
        return -9.0
    # reverse: OSM road points near the camera, seen from the camera
    op = obs["_osm_pts"]
    sel = np.linalg.norm(op - cam_o, axis=1) < max_range_m / res
    if sel.sum() < 60:
        return -9.0
    q = cv2.perspectiveTransform((op[sel] * res).reshape(-1, 1, 2),
                                 np.linalg.inv(np.float64(Hiw))).reshape(-1, 2)
    inim = ((q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < w_img - 1)
            & (q[:, 1] < h_img - 1) & (q[:, 1] > obs["horizon"] + 5))
    if inim.sum() < 40:
        return -9.0
    b = float(obs["road"][q[inim, 1].astype(int),
                          q[inim, 0].astype(int)].mean())
    sc = 2 * a * b / max(a + b, 1e-6)
    pt = fwd(obs["_paint_pts"], paint_r)
    return sc + 0.3 * (pt or 0.0)


def polish(pose, obs, sat_img, georef, roads, res=1.0, max_range_m=420.0,
           log=print):
    """Nelder-Mead refinement of a coarse pose, then an ECC paint alignment."""
    from scipy.optimize import minimize
    mpp = georef["mpp"]
    sat_h, sat_w = sat_img.shape[:2]
    scale = mpp / res
    osm_full = road_raster(roads, (sat_h, sat_w), mpp)
    pav = sat_pavement(sat_img, osm_full, log=lambda *a: None)
    osm_r = cv2.resize(0.45 * osm_full + 0.55 * pav,
                       (int(sat_w * scale), int(sat_h * scale)))
    paint_r = cv2.resize(pipeline._paint_mask(sat_img) *
                         (osm_full > 0.2).astype(np.float32),
                         (int(sat_w * scale), int(sat_h * scale)))
    obs = dict(obs)
    obs["_road_pts"] = _sample_mask(obs["road"], 3000, thresh=0.5)
    obs["_paint_pts"] = _sample_mask(obs["paint"], 2000)
    obs["_osm_pts"] = _sample_mask(osm_r, 4000, thresh=0.35)

    p0 = dict(pose)
    x0 = np.array([p0["cam_px"][0] * mpp, p0["cam_px"][1] * mpp, p0["height"],
                   p0["fov"], p0["road_bearing"], p0.get("roll", 0.0)])

    def unpack(x):
        return {"cam_px": [x[0] / mpp, x[1] / mpp], "height": x[2],
                "fov": x[3], "road_bearing": x[4], "roll": x[5]}

    # Where the scale comes from.  Overlap scores are almost blind to it -- a
    # slightly smaller footprint always sits a little more neatly inside the
    # road -- so left free the optimiser just walks to whatever bound it is
    # given.  Scale has to come from something measured in metres, and the
    # tracked cars are exactly that: a 1.85 m ruler lying on the road surface,
    # averaged over hundreds of samples.
    h_car = obs.get("height_car")
    h_lo, h_hi = (5.0, 25.0) if not h_car else (0.75 * h_car, 1.3 * h_car)

    def cost(x):
        if not (h_lo <= x[2] <= h_hi and 10 <= x[3] <= 75 and abs(x[5]) <= 12):
            return 9.0
        return -_score_pose(unpack(x), obs, osm_r, paint_r, mpp, res,
                            max_range_m)

    r = minimize(cost, x0, method="Nelder-Mead",
                 options={"maxiter": 900, "xatol": 0.05, "fatol": 1e-5,
                          "initial_simplex": _simplex(x0)})
    p = unpack(r.x)
    p["score"] = float(-r.fun)
    p["road"] = pose.get("road", "")
    log(f"[cal] refined: score {pose['score']:.3f} -> {p['score']:.3f}, "
        f"fov {p['fov']:.1f}°, height {p['height']:.1f} m, "
        f"roll {p['roll']:.1f}°")
    return p


def _similarity(cx, cy, deg, scale):
    t = math.radians(deg)
    c, s = math.cos(t) * scale, math.sin(t) * scale
    return np.array([[c, -s, cx - c * cx + s * cy],
                     [s, c, cy - s * cx - c * cy],
                     [0, 0, 1.0]])


def paint_lock(cam_paint, sat_img, H, road_mask, mpp, res=0.25,
               rot=(-6, 6, 1.0), scales=(0.90, 1.10, 0.025), max_shift_m=45,
               log=print, progress=None):
    """Snap a nearly-right homography onto the satellite's lane markings.

    ECC is the obvious tool here and it fails: lane lines are near-identical
    parallel stripes, so a gradient solver walks into the wrong lane or slides
    along the road.  Instead this searches rotation and scale exhaustively and
    solves translation exactly (by correlation) for each -- no local minima to
    fall into.
    """
    k = mpp / res
    h, w = sat_img.shape[:2]
    tw, th = int(w * k), int(h * k)
    tgt = pipeline._paint_mask(sat_img)
    if road_mask is not None:
        rm = cv2.dilate((road_mask > 0.25).astype(np.uint8),
                        np.ones((7, 7), np.uint8)).astype(np.float32)
        tgt = tgt * rm
    tgt = cv2.resize(tgt, (tw, th))
    tgt = cv2.GaussianBlur(tgt, (0, 0), 1.6)
    pad = int(max_shift_m / res)
    tgt_p = cv2.copyMakeBorder(tgt, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    S = np.array([[k, 0, 0], [0, k, 0], [0, 0, 1.0]])
    ch, cw = cam_paint.shape[:2]
    corners = cv2.perspectiveTransform(
        np.float32([[[0, ch - 1], [cw - 1, ch - 1], [cw / 2, ch * 0.55]]]),
        np.float64(S @ np.float64(H))).reshape(-1, 2)
    cx, cy = float(corners[:, 0].mean()), float(corners[:, 1].mean())

    rots = np.arange(rot[0], rot[1] + 1e-9, rot[2])
    scs = np.arange(scales[0], scales[1] + 1e-9, scales[2])
    best = (None, -9.0, None)
    n = 0
    for deg in rots:
        for sc in scs:
            n += 1
            if progress and n % 10 == 0:
                progress(n, len(rots) * len(scs))
            A = _similarity(cx, cy, float(deg), float(sc))
            Hs = A @ S @ np.float64(H)
            tmpl = cv2.warpPerspective(cam_paint, np.float32(Hs), (tw, th))
            vmask = cv2.warpPerspective(
                np.ones((ch, cw), np.float32), np.float32(Hs), (tw, th))
            vmask = (vmask > 0.5).astype(np.float32)
            if vmask.sum() < 2000:
                continue
            ys, xs = np.nonzero(vmask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            t_c = np.ascontiguousarray(tmpl[y0:y1, x0:x1])
            m_c = np.ascontiguousarray(vmask[y0:y1, x0:x1])
            t_c = cv2.GaussianBlur(t_c, (0, 0), 1.6) * m_c
            img = tgt_p[y0:y1 + 2 * pad, x0:x1 + 2 * pad]
            if img.shape[0] <= t_c.shape[0] or img.shape[1] <= t_c.shape[1]:
                continue
            sc_map = masked_ncc(t_c, m_c, img)
            if sc_map is None:
                continue
            idx = int(np.argmax(sc_map))
            r, c = divmod(idx, sc_map.shape[1])
            val = float(sc_map[r, c])
            if val > best[1]:
                dx, dy = (c - pad) / k, (r - pad) / k    # back to satellite px
                T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
                Hn = T @ np.linalg.inv(S) @ A @ S @ np.float64(H)
                best = (Hn / Hn[2, 2], val, (float(deg), float(sc), dx, dy))
    if best[0] is None:
        log("[cal] lane-marking lock-on found nothing; keeping the search pose")
        return np.float64(H), 0.0
    d, s, dx, dy = best[2]
    log(f"[cal] locked onto lane markings: rotate {d:+.1f}°, scale {s:.3f}, "
        f"shift {math.hypot(dx, dy) * mpp:.1f} m, match {best[1]:.3f}")
    return best[0], best[1]


def ecc_polish(cam_paint, sat_img, H, road_mask=None, log=print,
               sigmas=(8.0, 4.0, 2.0), ds=2):
    """Final lock-on: align the camera's lane paint to the satellite's.

    The pose search gets within a lane or two; this closes the gap by matching
    the markings directly -- coarse blur first so it cannot settle into the
    wrong lane, then sharpening.

    `road_mask` matters: a plain top-hat of a satellite tile lights up every
    tree and rooftop, which swamps the lane markings and leaves ECC with two
    uncorrelated images.  Restricting the target to the road corridor fixes it.
    """
    h, w = sat_img.shape[:2]
    S = np.array([[1.0 / ds, 0, 0], [0, 1.0 / ds, 0], [0, 0, 1.0]])
    small = cv2.resize(sat_img, (w // ds, h // ds))
    hs, ws = small.shape[:2]
    tgt0 = pipeline._paint_mask(small)
    if road_mask is not None:
        rm = cv2.resize(road_mask, (ws, hs))
        rm = cv2.dilate((rm > 0.25).astype(np.uint8), np.ones((9, 9), np.uint8))
        tgt0 = tgt0 * rm.astype(np.float32)
    Hc = np.float64(H)
    best_cc = 0.0
    for sg in sigmas:
        Hs = S @ Hc
        warped = cv2.warpPerspective(cam_paint, np.float32(Hs), (ws, hs))
        valid = cv2.warpPerspective(np.ones(cam_paint.shape[:2], np.float32),
                                    np.float32(Hs), (ws, hs))
        region = ((valid > 0.5) & (tgt0 > 0)).astype(np.uint8)
        if region.sum() < 3000:
            log("[cal] projected view hardly overlaps the road in this tile; "
                "skipping paint alignment")
            break
        mov = cv2.GaussianBlur(warped, (0, 0), sg / ds)
        tgt = cv2.GaussianBlur(tgt0, (0, 0), sg / ds)
        mov = cv2.normalize(mov, None, 0, 1, cv2.NORM_MINMAX)
        tgt = cv2.normalize(tgt, None, 0, 1, cv2.NORM_MINMAX)
        warp = np.eye(3, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        try:
            cc, warp = cv2.findTransformECC(tgt, mov, warp,
                                            cv2.MOTION_HOMOGRAPHY, crit,
                                            region, 5)
        except cv2.error:
            log(f"[cal] paint alignment did not converge at blur {sg:.0f}px")
            continue
        step = np.linalg.inv(np.float64(warp)) @ Hs
        Hc = np.linalg.inv(S) @ step
        Hc = Hc / Hc[2, 2]
        best_cc = float(cc)
        log(f"[cal] paint alignment at blur {sg:.0f}px: correlation {cc:.3f}")
    return Hc, best_cc


def _simplex(x0):
    steps = np.array([12.0, 12.0, 2.0, 5.0, 3.0, 2.0])
    s = [x0]
    for i in range(len(x0)):
        v = x0.copy()
        v[i] += steps[i]
        s.append(v)
    return np.array(s)


def _sample_mask(mask, n, thresh=0.12):
    ys, xs = np.nonzero(mask > thresh)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    if len(xs) > n:
        k = np.random.default_rng(0).choice(len(xs), n, replace=False)
        xs, ys = xs[k], ys[k]
    return np.float32(np.column_stack([xs, ys]))


def pose_to_H(pose, obs, mpp):
    """Camera-image -> satellite-pixel homography for a solved pose."""
    w_img, h_img = obs["w"], obs["h"]
    cx, cy = w_img / 2.0, h_img / 2.0
    f = _fov_to_f(pose["fov"], w_img)
    pan, tilt = pan_tilt_from_vp(obs["vp"], f, cx, cy, pose["road_bearing"],
                                 pose["roll"])
    C = (pose["cam_px"][0] * mpp, pose["cam_px"][1] * mpp, pose["height"])
    Hiw = H_img_to_world(f, cx, cy, pan, tilt, pose["roll"], C)
    S = np.array([[1 / mpp, 0, 0], [0, 1 / mpp, 0], [0, 0, 1.0]])
    H = S @ Hiw
    return H / H[2, 2], {"pan": pan, "tilt": tilt, "f": f}


def auto_calibrate(clip_path, cam_info, work_dir, zoom=19, grid=7,
                   max_range_m=260.0, log=print, progress=None):
    """One call: watch the camera, read the map, and solve the homography.

    Returns everything the dashboard needs, including a picture of how it got
    there.  Nothing here is interactive -- the operator only steps in if the
    reported confidence is poor.
    """
    st = {"stage": "watching the camera"}

    def stage(s):
        st["stage"] = s
        log(f"[cal] {s}")
        if progress:
            progress(s)

    stage("watching the camera (background, tracks, lane lines)")
    obs = observe(clip_path, work_dir, log=log)

    stage("reading the road network from OpenStreetMap")
    lat0, lon0 = cam_info["lat"], cam_info["lon"]
    els = fetch_osm(lat0, lon0, 1400, log)
    clat, clon = plan_area(els, lat0, lon0, cam_info.get("roadway"),
                           cam_info.get("direction"), log)

    stage("fetching a wide satellite view to search in")
    wide_png = os.path.join(work_dir, "autocal_wide.png")
    gw = pipeline.fetch_satellite(clat, clon, 18, 13, wide_png, log)
    wide = cv2.imread(wide_png)
    roads = roads_to_pixels(els, gw)
    cam_px = latlon_to_satpx(gw, lat0, lon0)

    stage("searching for the camera pose")
    cands = search(obs, wide, gw, roads, cam_px, cam_info.get("roadway"),
                   cam_info.get("direction"), log=log)
    # A road looks nearly the same from either end, so the fit score alone can
    # tie a camera looking one way against one looking the other.  FL511 states
    # which direction each camera watches, and the tracks tell us whether that
    # traffic is coming or going, which settles it.
    want_b = {"NORTHBOUND": 0, "EASTBOUND": 90, "SOUTHBOUND": 180,
              "WESTBOUND": 270}.get((cam_info.get("direction") or "").upper())
    best, best_rank = None, -9e9
    for c in cands[:5]:
        p = polish(c, obs, wide, gw, roads, log=log)
        p["road"] = c.get("road", "")
        rank = p["score"]
        if want_b is not None:
            travel = p["road_bearing"] if obs["recede"] \
                else p["road_bearing"] + 180.0
            dd = abs((travel - want_b + 180) % 360 - 180)
            p["direction_err_deg"] = dd
            rank += 0.25 * max(0.0, 1.0 - dd / 75.0)
        if rank > best_rank:
            best, best_rank = p, rank
    if want_b is not None and best.get("direction_err_deg") is not None:
        log(f"[cal] traffic direction agrees with FL511's "
            f"'{cam_info.get('direction')}' to within "
            f"{best['direction_err_deg']:.0f}°")
    H_wide, info = pose_to_H(best, obs, gw["mpp"])

    stage("centring the satellite image on what the camera sees")
    mid = cv2.perspectiveTransform(
        np.float32([[[obs["w"] / 2, obs["h"] * 0.80]]]),
        np.float64(H_wide)).reshape(2)
    mlat, mlon = pipeline.pixel_to_latlon(gw, float(mid[0]), float(mid[1]))
    sat_png = os.path.join(work_dir, "sat.png")
    georef = pipeline.fetch_satellite(mlat, mlon, zoom, grid, sat_png, log)
    sat = cv2.imread(sat_png)
    H = rebase_H(H_wide, gw, georef)
    roads2 = roads_to_pixels(els, georef)
    osm2 = road_raster(roads2, sat.shape[:2], georef["mpp"])

    stage("locking onto the lane markings")
    H2, cc = paint_lock(obs["paint"], sat, H, osm2, georef["mpp"], res=0.4,
                        rot=(-4, 4, 1.0), scales=(0.88, 1.12, 0.03),
                        max_shift_m=22, log=log)
    # only accept a *small* correction: the lane-paint objective is periodic
    # along the road, so a large "improvement" usually means it slid a block
    shift = np.linalg.norm(
        cv2.perspectiveTransform(np.float32([[[obs["w"] / 2, obs["h"] * .8]]]),
                                 np.float64(H2)).reshape(2) - mid * 0
        - cv2.perspectiveTransform(np.float32([[[obs["w"] / 2, obs["h"] * .8]]]),
                                   np.float64(H)).reshape(2)) * georef["mpp"]
    if cc > 0.25 and shift < 25:
        log(f"[cal] accepted the lane lock-on ({shift:.1f} m adjustment)")
        H = H2
    else:
        log(f"[cal] rejected the lane lock-on (match {cc:.2f}, "
            f"{shift:.0f} m) — keeping the geometric solution")
        cc = 0.0

    cam_px2 = latlon_to_satpx(georef, *pipeline.pixel_to_latlon(
        gw, best["cam_px"][0], best["cam_px"][1]))
    zone = view_zone(H, obs["w"], obs["h"], cam_px2, georef["mpp"],
                     max_range_m=max_range_m)
    clat2, clon2 = pipeline.pixel_to_latlon(gw, *best["cam_px"])
    fig = process_figure(obs, sat, wide, gw, georef, roads, roads2, best,
                         H, cam_px2, zone)
    cv2.imwrite(os.path.join(work_dir, "autocal_process.png"), fig)
    cv2.imwrite(os.path.join(work_dir, "calib.jpg"), obs["bg"])
    out = {"H": np.asarray(H).tolist(), "range_m": max_range_m,
           "zone": None if zone is None else np.asarray(zone).tolist(),
           "georef": georef, "center": [mlat, mlon], "zoom": zoom, "grid": grid,
           "pose": {k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in best.items() if k != "cam_px"},
           "camera_latlon": [clat2, clon2], "pan": info["pan"],
           "tilt": info["tilt"], "score": best["score"], "lane_match": cc,
           "n_tracks": obs["n_tracks"], "n_lane_lines": len(obs["rays"] or []),
           "vp": obs["vp"]}
    log(f"[cal] done — camera at {clat2:.6f}, {clon2:.6f}, looking "
        f"{_compass(info['pan'])} ({info['pan']:.0f}°), {info['tilt']:.1f}° "
        f"below horizontal, mast {best['height']:.1f} m, fit score "
        f"{best['score']:.2f}")
    return out


def _compass(b):
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW",
             "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((b % 360) / 22.5 + 0.5) % 16]


def process_figure(obs, sat, wide, gw, georef, roads_w, roads_s, pose, H,
                   cam_px, zone):
    """A picture of how the calibration was reached, for the operator to judge."""
    def label(img, text):
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    S = 620
    # 1. what the camera showed us
    a = obs["bg"].copy()
    a[:, :, 2] = np.maximum(a[:, :, 2], (obs["road"] * 150).astype(np.uint8))
    v = np.float64(obs["vp"])
    for th, _w in (obs["rays"] or []):
        d = np.array([math.cos(th), math.sin(th)])
        p2 = v + d * 2400
        cv2.line(a, (int(v[0]), int(v[1])), (int(p2[0]), int(p2[1])),
                 (0, 255, 255), 1)
    cv2.circle(a, (int(v[0]), int(v[1])), 9, (0, 0, 255), 2)
    a = label(cv2.resize(a, (S, int(S * a.shape[0] / a.shape[1]))),
              f"1. camera: {obs['n_tracks']} tracks, "
              f"{len(obs['rays'] or [])} lane lines, vanishing point")

    # 2. the map we matched against
    b = wide.copy()
    rr = road_raster(roads_w, wide.shape[:2], gw["mpp"])
    b[:, :, 1] = np.maximum(b[:, :, 1], (rr * 150).astype(np.uint8))
    cp = np.int32(pose["cam_px"])
    cv2.circle(b, tuple(cp), 16, (0, 255, 255), 3)
    r = 900
    x0, y0 = max(0, cp[0] - r), max(0, cp[1] - r)
    b = b[y0:y0 + 2 * r, x0:x0 + 2 * r]
    b = label(cv2.resize(b, (S, S)), "2. OpenStreetMap roads + camera found")

    # 3. the camera projected onto the satellite
    c = sat.copy()
    warp = cv2.warpPerspective(obs["bg"], np.float64(H),
                               (sat.shape[1], sat.shape[0]))
    m = warp.sum(2) > 0
    c[m] = (0.5 * warp[m] + 0.5 * c[m]).astype(np.uint8)
    if zone is not None:
        cv2.polylines(c, [np.int32(zone)], True, (0, 255, 0), 3)
    cv2.circle(c, (int(cam_px[0]), int(cam_px[1])), 10, (0, 255, 255), -1)
    c = label(cv2.resize(c, (S, S)), "3. camera view projected + trusted zone")

    # 4. lane markings, camera (red) vs satellite (green)
    osm_s = road_raster(roads_s, sat.shape[:2], georef["mpp"])
    cpaint = cv2.warpPerspective(obs["paint"], np.float64(H),
                                 (sat.shape[1], sat.shape[0]))
    spaint = pipeline._paint_mask(sat) * (osm_s > 0.2)
    d = (cv2.cvtColor(cv2.cvtColor(sat, cv2.COLOR_BGR2GRAY),
                      cv2.COLOR_GRAY2BGR) * 0.5).astype(np.uint8)
    d[:, :, 2] = np.maximum(d[:, :, 2],
                            (np.clip(cpaint * 3.5, 0, 1) * 255).astype(np.uint8))
    d[:, :, 1] = np.maximum(d[:, :, 1],
                            (np.clip(spaint * 2.5, 0, 1) * 255).astype(np.uint8))
    d = label(cv2.resize(d, (S, S)),
              "4. lane paint: red = camera, green = satellite")

    hgt = max(a.shape[0], b.shape[0])
    row1 = np.hstack([_pad(a, hgt), _pad(b, hgt)])
    row2 = np.hstack([_pad(c, hgt), _pad(d, hgt)])
    return np.vstack([row1, row2])


def _pad(img, h):
    if img.shape[0] >= h:
        return img[:h]
    return np.vstack([img, np.zeros((h - img.shape[0], img.shape[1], 3),
                                    np.uint8)])


def view_zone(H, w_img, h_img, cam_px, mpp, max_range_m=250.0, near_m=12.0):
    """Trusted zone: the camera's ground footprint clipped to a range band.

    This replaces the old 'convex hull of clicked points' zone, which was tiny
    and rejected nearly every vehicle.
    """
    Hf = np.float64(H)
    pts = []
    for t in np.linspace(0, 1, 60):
        pts.append([t * (w_img - 1), h_img - 1])
    for t in np.linspace(0, 1, 40):          # up the sides
        pts.append([0, h_img - 1 - t * (h_img - 1)])
        pts.append([w_img - 1, h_img - 1 - t * (h_img - 1)])
    p = np.float32(pts).reshape(-1, 1, 2)
    den = (Hf[2, 0] * p[:, 0, 0] + Hf[2, 1] * p[:, 0, 1] + Hf[2, 2])
    ref = Hf[2, 0] * (w_img / 2) + Hf[2, 1] * (h_img - 1) + Hf[2, 2]
    p = p[(den * ref) > 0]
    if len(p) < 3:
        return None
    g = cv2.perspectiveTransform(p, Hf).reshape(-1, 2)
    c = np.float32(cam_px)
    d = np.linalg.norm(g - c, axis=1) * mpp
    g = g[d < max_range_m]
    if len(g) < 3:
        return None
    ring = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    band = np.float32([[c[0] + math.cos(a) * max_range_m / mpp,
                        c[1] + math.sin(a) * max_range_m / mpp] for a in ring])
    hull = cv2.convexHull(np.vstack([g, np.float32([c])]))
    band_poly = cv2.convexHull(band)
    inter = cv2.intersectConvexConvex(hull, band_poly)[1]
    if inter is None or len(inter) < 3:
        inter = hull
    zone = inter.reshape(-1, 2)
    # trim the sliver right under the camera (bad geometry, extreme foreshorten)
    keep = np.linalg.norm(zone - c, axis=1) * mpp > near_m
    return (zone if keep.sum() < 3 else zone).astype(float)
