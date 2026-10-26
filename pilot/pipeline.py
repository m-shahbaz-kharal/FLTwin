"""Homography, satellite tiles, YOLO detection/tracking, top-down projection."""
import json
import math
import os

import cv2
import numpy as np
import requests
import imageio_ffmpeg

BASE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(BASE, "work")
MODELS = os.path.join(BASE, "models")

# detector settings (changeable from the dashboard)
MODEL_NAME = "yolo11x.pt"
IMGSZ = 1280
CONF = 0.20
DEVICE = "auto"          # "auto" | "cpu" | "cuda" | "cuda:0"
SPEED_WINDOW_S = 0.8     # time baseline for speed/heading estimates

_model = None
_model_key = None


def device():
    """'cuda:0' when a GPU is really usable, else 'cpu'."""
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def set_model(name=None, imgsz=None, conf=None):
    global MODEL_NAME, IMGSZ, CONF
    if name:
        MODEL_NAME = name
    if imgsz:
        IMGSZ = int(imgsz)
    if conf:
        CONF = float(conf)
    return {"model": MODEL_NAME, "imgsz": IMGSZ, "conf": CONF,
            "device": device()}

# COCO vehicle classes: id -> (name, (length_m, width_m), BGR color)
VEHICLES = {
    2: ("car", (4.7, 1.9), (0, 200, 255)),
    3: ("motorcycle", (2.2, 0.9), (0, 255, 255)),
    5: ("bus", (12.0, 2.6), (255, 0, 200)),
    7: ("truck", (8.5, 2.6), (0, 140, 255)),
}
TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")

def get_model(log=print):
    global _model, _model_key
    dev = device()
    if _model is None or _model_key != (MODEL_NAME, dev):
        from ultralytics import YOLO
        path = os.path.join(MODELS, MODEL_NAME)
        log(f"[yolo] loading {MODEL_NAME} on {dev}")
        _model = YOLO(path)
        if dev.startswith("cuda"):
            _model.to(dev)
        _model_key = (MODEL_NAME, dev)
    return _model


# ---------------- satellite tiles (Esri World Imagery) ----------------

def fetch_satellite(lat, lon, zoom, grid, out_png, log=print):
    """Stitch a grid x grid tile mosaic centred on (lat, lon). Returns georef."""
    n = 2 ** zoom
    xf = (lon + 180.0) / 360.0 * n
    yf = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    cx, cy = int(xf), int(yf)
    half = grid // 2
    x0, y0 = cx - half, cy - half
    mosaic = np.zeros((grid * 256, grid * 256, 3), np.uint8)
    for j in range(grid):
        for i in range(grid):
            url = TILE_URL.format(z=zoom, y=y0 + j, x=x0 + i)
            r = requests.get(url, headers={"User-Agent": "FLTwin-pilot"},
                             timeout=20)
            if r.status_code != 200:
                log(f"[sat] tile {zoom}/{y0 + j}/{x0 + i} -> HTTP {r.status_code}")
                continue
            tile = cv2.imdecode(np.frombuffer(r.content, np.uint8),
                                cv2.IMREAD_COLOR)
            if tile is not None:
                mosaic[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = tile
    cv2.imwrite(out_png, mosaic)
    # metres per pixel of a 256px web-mercator tile at this latitude
    mpp = 40075016.686 * math.cos(math.radians(lat)) / (n * 256)
    georef = {"type": "esri", "zoom": zoom,
              "gx0": x0 * 256, "gy0": y0 * 256,   # global pixel of top-left
              "mpp": mpp, "size": grid * 256}
    log(f"[sat] mosaic {grid}x{grid} tiles @ z{zoom}, "
        f"{mpp:.3f} m/px, {mpp * grid * 256:.0f} m across")
    return georef


def pixel_to_latlon(georef, px, py):
    z = georef["zoom"]
    n = 2 ** z * 256
    gx, gy = float(georef["gx0"] + px), float(georef["gy0"] + py)
    lon = gx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * gy / n))))
    return float(lat), float(lon)


# ---------------- homography ----------------

def compute_homography(pairs):
    """pairs: [[cx, cy, sx, sy], ...] -> (H, rms_px, per_point_px)."""
    src = np.float32([[p[0], p[1]] for p in pairs])
    dst = np.float32([[p[2], p[3]] for p in pairs])
    method = cv2.RANSAC if len(pairs) >= 8 else 0
    H, _ = cv2.findHomography(src, dst, method, 5.0)
    if H is None:
        raise RuntimeError("homography could not be computed")
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1).astype(float)
    return H, float(np.sqrt((err ** 2).mean())), np.round(err, 1).tolist()


def project_points(H, pts):
    a = np.float32(pts).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(a, np.float32(H)).reshape(-1, 2)


def make_overlay(cam_img, sat_img, H, pairs, hull):
    """Semi-transparent warp of the camera frame onto the satellite image."""
    h, w = sat_img.shape[:2]
    warped = cv2.warpPerspective(cam_img, np.float32(H), (w, h))
    mask = (warped.sum(axis=2) > 0)
    out = sat_img.copy()
    out[mask] = (0.55 * warped[mask] + 0.45 * out[mask]).astype(np.uint8)
    for i, p in enumerate(pairs):
        cv2.circle(out, (int(p[2]), int(p[3])), 6, (0, 0, 255), 2)
        cv2.putText(out, str(i + 1), (int(p[2]) + 8, int(p[3]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if hull is not None:
        cv2.polylines(out, [np.int32(hull)], True, (0, 255, 0), 2)
    return out


def zone_hull(pairs, margin_px=25):
    """Trusted zone = convex hull of the satellite-side reference points."""
    pts = np.float32([[p[2], p[3]] for p in pairs])
    hull = cv2.convexHull(pts).reshape(-1, 2)
    c = hull.mean(axis=0)
    return ((hull - c) * (1 + margin_px / (np.linalg.norm(hull - c, axis=1)
                                           .mean() + 1e-6)) + c)


def _paint_mask(bgr, region=None):
    """Soft mask of bright road paint (lane lines) via morphological top-hat."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    th = cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX)
    th = cv2.GaussianBlur(th, (0, 0), 3)
    if region is not None:
        th = cv2.bitwise_and(th, th, mask=region)
    return th.astype(np.float32) / 255.0


def refine_homography(cam_img, sat_img, H, hull, log=print):
    """Polish a rough homography by ECC-aligning lane-paint masks of the
    warped camera frame and the satellite image inside the trusted zone.
    Returns (H_refined, ecc_score)."""
    h, w = sat_img.shape[:2]
    Hf = np.float32(H)
    warped = cv2.warpPerspective(cam_img, Hf, (w, h))
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [np.int32(hull)], 255)
    region = cv2.dilate(region, np.ones((81, 81), np.uint8))
    valid = cv2.warpPerspective(
        np.full(cam_img.shape[:2], 255, np.uint8), Hf, (w, h))
    region = cv2.bitwise_and(region, valid)
    template = _paint_mask(sat_img, region)
    moving = _paint_mask(warped, region)
    warp = np.eye(3, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 400, 1e-7)
    cc, warp = cv2.findTransformECC(template, moving, warp,
                                    cv2.MOTION_HOMOGRAPHY, crit, None, 5)
    Hn = np.linalg.inv(warp) @ Hf
    Hn = Hn / Hn[2, 2]
    log(f"[refine] ECC converged, correlation {cc:.3f}")
    return Hn, float(cc)


def view_change_score(ref_img, cur_img):
    """Median keypoint displacement (px) between two frames of the same camera.
    Large value (> ~10) suggests the PTZ camera moved and calibration is stale."""
    orb = cv2.ORB_create(800)
    g1 = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return -1.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 15:
        return -1.0
    matches = sorted(matches, key=lambda m: m.distance)[:150]
    disp = [np.linalg.norm(np.float32(k1[m.queryIdx].pt) -
                           np.float32(k2[m.trainIdx].pt)) for m in matches]
    return float(np.median(disp))


# ---------------- detection + rendering ----------------

def run_detection(clip_path, H, hull, sat_img_path, georef, out_dir,
                  stride=3, imgsz=None, conf=None, log=print, progress=None):
    """Run YOLO+ByteTrack on the clip, project to satellite, render side-by-side
    mp4 and a detections JSON. Returns summary stats."""
    model = get_model(log)
    imgsz = int(imgsz or IMGSZ)
    conf = float(conf or CONF)
    dev = device()
    log(f"[detect] {MODEL_NAME} @ imgsz {imgsz}, conf {conf}, device {dev}")
    Hm = np.float32(H)
    sat_base = cv2.imread(sat_img_path)
    hull_i = np.int32(hull).reshape(-1, 1, 2) if hull is not None else None
    mpp = georef.get("mpp") if georef else None

    cap = cv2.VideoCapture(clip_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 15.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = max(1.0, fps_in / stride)
    log(f"[detect] clip: {n_frames} frames @ {fps_in:.1f} fps, "
        f"stride {stride} -> processing {n_frames // stride} frames")

    tracks = {}   # id -> {"hist": [(t, sx, sy)], "cls": int, "speed": mph}
    frames_json = []
    writer = None
    out_mp4 = os.path.join(out_dir, "result.mp4")
    fi = 0
    done = 0
    n_untracked = 0
    n_det = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % stride:
            fi += 1
            continue
        t = fi / fps_in
        fi += 1
        done += 1
        res = model.track(frame, persist=True, verbose=False, conf=conf,
                          iou=0.5, imgsz=imgsz, classes=list(VEHICLES),
                          tracker="bytetrack.yaml", device=dev)[0]

        left = frame.copy()
        right = sat_base.copy()
        if hull_i is not None:
            cv2.polylines(right, [hull_i], True, (0, 255, 0), 2)
        fj = {"t": round(t, 2), "vehicles": []}

        boxes = res.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            # Untracked detections still count. ByteTrack needs a few frames to
            # confirm a track, and anything it has not confirmed comes back
            # with id None -- dropping those threw away most of the traffic.
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                ids = np.array([-(i + 1) for i in range(len(xyxy))])
                n_untracked += len(xyxy)
            clss = boxes.cls.cpu().numpy().astype(int)
            n_det += len(xyxy)
            feet = np.float32([[(b[0] + b[2]) / 2, b[3]] for b in xyxy])
            sat_pts = project_points(Hm, feet)
            for b, tid, cls, sp in zip(xyxy, ids, clss, sat_pts):
                name, dims, color = VEHICLES[int(cls)]
                inside = (hull_i is None or
                          cv2.pointPolygonTest(hull_i, (float(sp[0]),
                                                        float(sp[1])), True) > -20)
                # left panel: camera bbox
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(left, (x1, y1), (x2, y2), color,
                              2 if inside else 1)
                label = f"{tid}"
                cv2.putText(left, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if not inside:
                    continue
                tr = tracks.setdefault(int(tid), {"hist": [], "cls": int(cls),
                                                  "speed": None})
                tr["hist"].append((t, float(sp[0]), float(sp[1])))
                # heading + speed from recent motion
                heading, speed_mph = None, None
                h_ = tr["hist"]
                if len(h_) >= 2:
                    # Measure over a fixed ~0.8 s of time, not a fixed number of
                    # samples: a sample-count window shrinks as the frame stride
                    # drops, and short baselines turn position jitter into
                    # nonsense speeds.
                    t1, x1s, y1s = h_[-1]
                    t0, x0, y0 = h_[0]
                    for (ta, xa, ya) in reversed(h_):
                        if t1 - ta >= SPEED_WINDOW_S:
                            t0, x0, y0 = ta, xa, ya
                            break
                    dx, dy, dt = x1s - x0, y1s - y0, t1 - t0
                    min_px = (1.5 / mpp) if mpp else 3.0
                    if dt >= 0.25 and math.hypot(dx, dy) > min_px:
                        heading = math.atan2(dy, dx)
                        if mpp:
                            speed_mph = (math.hypot(dx, dy) * mpp / dt) * 2.23694
                            prev = tr["speed"]
                            speed_mph = (0.7 * prev + 0.3 * speed_mph
                                         if prev else speed_mph)
                            tr["speed"] = speed_mph
                    tr["heading"] = heading if heading is not None \
                        else tr.get("heading")
                heading = tr.get("heading")
                # right panel: oriented footprint rectangle
                if mpp:
                    L, W = dims[0] / mpp, dims[1] / mpp
                else:
                    L, W = 18, 8
                ang = heading if heading is not None else 0.0
                ca, sa = math.cos(ang), math.sin(ang)
                corners = np.float32(
                    [[dx_ * ca - dy_ * sa + sp[0], dx_ * sa + dy_ * ca + sp[1]]
                     for dx_, dy_ in [(-L / 2, -W / 2), (L / 2, -W / 2),
                                      (L / 2, W / 2), (-L / 2, W / 2)]])
                cv2.fillPoly(right, [np.int32(corners)], color)
                cv2.polylines(right, [np.int32(corners)], True, (255, 255, 255), 1)
                if tr.get("speed"):
                    cv2.putText(right, f"{tr['speed']:.0f}",
                                (int(sp[0]) + 8, int(sp[1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                v = {"id": int(tid), "cls": name,
                     "bbox_cam": [round(float(x), 1) for x in b],
                     "sat_px": [round(float(sp[0]), 1), round(float(sp[1]), 1)]}
                if tr.get("speed"):
                    v["speed_mph"] = round(tr["speed"], 1)
                if georef and georef.get("type") == "esri":
                    lat, lon = pixel_to_latlon(georef, sp[0], sp[1])
                    v["lat"], v["lon"] = round(lat, 6), round(lon, 6)
                fj["vehicles"].append(v)
        frames_json.append(fj)

        # compose side-by-side (right resized to left height)
        hL = left.shape[0]
        right_r = cv2.resize(right, (int(right.shape[1] * hL / right.shape[0]),
                                     hL))
        comp = np.hstack([left, right_r])
        if comp.shape[1] > 1920:
            s = 1920 / comp.shape[1]
            comp = cv2.resize(comp, (1920, int(comp.shape[0] * s)))
        comp = comp[:comp.shape[0] // 2 * 2, :comp.shape[1] // 2 * 2]
        if writer is None:
            size = (comp.shape[1], comp.shape[0])
            writer = imageio_ffmpeg.write_frames(
                out_mp4, size, fps=out_fps, codec="libx264",
                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            writer.send(None)
        writer.send(np.ascontiguousarray(comp[:, :, ::-1]).tobytes())
        if progress and done % 10 == 0:
            progress(done, max(1, n_frames // stride))
            log(f"[detect] {done}/{n_frames // stride} frames processed")
    cap.release()
    if writer:
        writer.close()
    with open(os.path.join(out_dir, "detections.json"), "w") as f:
        json.dump({"georef": georef, "frames": frames_json}, f)
    n_veh = len([k for k in tracks if k >= 0])
    speeds = [t["speed"] for t in tracks.values() if t.get("speed")]
    in_zone = sum(len(f["vehicles"]) for f in frames_json)
    stats = {"frames": done, "unique_vehicles": n_veh,
             "detections": n_det, "in_zone": in_zone,
             "per_frame": round(n_det / max(done, 1), 1),
             "avg_speed_mph": round(float(np.mean(speeds)), 1) if speeds else None}
    log(f"[detect] done: {done} frames, {n_det} detections "
        f"({stats['per_frame']}/frame), {in_zone} inside the trusted zone, "
        f"{n_veh} tracked vehicles"
        + (f", avg speed {stats['avg_speed_mph']} mph" if speeds else ""))
    if n_untracked > n_det * 0.3:
        log("[detect] many detections were not tracked — use a smaller frame "
            "stride (1 is best on a GPU) so the tracker can follow vehicles")
    return stats
