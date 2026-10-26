"""Offline harness for the automatic calibration (development / validation).

  python test_autocal.py observe     # watch the clip, cache the observation
  python test_autocal.py sat         # fetch the wide satellite + OSM
  python test_autocal.py solve       # run the search and score it
"""
import json
import os
import sys
import time

import cv2
import numpy as np

import autocal
import pipeline

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
OBS = os.path.join(W, "_obs.npz")
WIDE = os.path.join(W, "_wide.png")
WIDEJ = os.path.join(W, "_wide.json")
OSMJ = os.path.join(W, "_osm.json")

# FL511's listed (wrong) position for camera 817 and the centre the user
# actually calibrated against -- ~580 m apart.
FL511 = (28.487401, -81.448087)
TRUE_CENTRE = (28.482868581884752, -81.44511195475695)


def do_observe():
    pipeline.set_model("yolo11s.pt", imgsz=960)
    t0 = time.time()
    obs = autocal.observe(os.path.join(W, "clip.mp4"), W, stride=2)
    np.savez(OBS, bg=obs["bg"], drive=obs["drive"], road=obs["road"],
             paint=obs["paint"], vp=np.float32(obs["vp"]),
             horizon=obs["horizon"], recede=obs["recede"],
             w=obs["w"], h=obs["h"], n_tracks=obs["n_tracks"],
             cars=obs["cars"], rays=np.float32(obs["rays"] or []))
    print("observe took %.1fs" % (time.time() - t0))


def load_obs():
    z = np.load(OBS)
    return {"bg": z["bg"], "drive": z["drive"], "road": z["road"],
            "paint": z["paint"], "vp": [float(z["vp"][0]), float(z["vp"][1])],
            "horizon": float(z["horizon"]), "recede": bool(z["recede"]),
            "w": int(z["w"]), "h": int(z["h"]), "cars": z["cars"],
            "rays": [(float(a), float(b)) for a, b in z["rays"]],
            "n_tracks": int(z["n_tracks"])}


def do_sat(zoom=18, grid=13):
    els = autocal.fetch_osm(FL511[0], FL511[1], 1400)
    json.dump(els, open(OSMJ, "w"))
    clat, clon = autocal.plan_area(els, FL511[0], FL511[1], "I-4", "Westbound")
    g = pipeline.fetch_satellite(clat, clon, zoom, grid, WIDE)
    json.dump(g, open(WIDEJ, "w"))


def user_H_in_wide(georef_wide):
    """The manual homography, expressed in the wide mosaic's pixels."""
    st = json.load(open(os.path.join(W, "state.json")))
    if not st.get("H"):
        return None
    gu = st["sat"]["georef"]
    s = 2.0 ** (georef_wide["zoom"] - gu["zoom"])
    T = np.array([[s, 0, gu["gx0"] * s - georef_wide["gx0"]],
                  [0, s, gu["gy0"] * s - georef_wide["gy0"]],
                  [0, 0, 1.0]])
    return T @ np.float64(st["H"])


def do_solve():
    obs = load_obs()
    sat = cv2.imread(WIDE)
    georef = json.load(open(WIDEJ))
    els = json.load(open(OSMJ))
    roads = autocal.roads_to_pixels(els, georef)
    cam_px = autocal.latlon_to_satpx(georef, *FL511)
    print(f"vp={obs['vp']}, recede={obs['recede']}, tracks={obs['n_tracks']}, "
          f"mpp={georef['mpp']:.3f}, sat={sat.shape}")

    t0 = time.time()
    cands = autocal.search(obs, sat, georef, roads, cam_px, "I-4", "Westbound")
    print("search %.1fs" % (time.time() - t0))
    for c in cands[:6]:
        print("   %.3f  %-12s fov %2d h %2d  bearing %6.1f  cam_px %s"
              % (c["score"], c["road"], c["fov"], c["height"],
                 c["road_bearing"], np.round(c["cam_px"], 1)))

    best = None
    for c in cands[:5]:
        p = autocal.polish(c, obs, sat, georef, roads)
        if best is None or p["score"] > best["score"]:
            best = p
    H, info = autocal.pose_to_H(best, obs, georef["mpp"])
    lat, lon = pipeline.pixel_to_latlon(georef, *best["cam_px"])
    print("BEST score %.3f  fov %.1f  h %.1f  roll %.1f  pan %.1f  tilt %.1f"
          % (best["score"], best["fov"], best["height"], best["roll"],
             info["pan"], info["tilt"]))
    print("camera estimated at %.6f, %.6f" % (lat, lon))
    print("  FL511 says        %.6f, %.6f  (%.0f m away)"
          % (FL511[0], FL511[1], _dist(FL511, (lat, lon))))

    Hu = user_H_in_wide(georef)
    if Hu is not None:
        pts = np.float32([[x, y] for x in np.linspace(100, obs["w"] - 100, 9)
                          for y in np.linspace(obs["h"] * 0.55, obs["h"] - 20, 7)])
        a = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.float64(H)).reshape(-1, 2)
        b = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hu).reshape(-1, 2)
        d = np.linalg.norm(a - b, axis=1) * georef["mpp"]
        print("vs MANUAL homography over the lower image: median %.1f m, "
              "p90 %.1f m" % (np.median(d), np.percentile(d, 90)))

    ov = pipeline.make_overlay(obs["bg"], sat, H, [], None)
    cv2.circle(ov, (int(best["cam_px"][0]), int(best["cam_px"][1])), 7,
               (0, 0, 255), -1)
    rr = autocal.road_raster(roads, sat.shape[:2], georef["mpp"])
    ov[:, :, 1] = np.maximum(ov[:, :, 1], (rr * 90).astype(np.uint8))
    cv2.imwrite(os.path.join(W, "_autocal_overlay.png"), ov)
    print("wrote work/_autocal_overlay.png")
    json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist())
               for k, v in best.items()},
              open(os.path.join(W, "_autocal_pose.json"), "w"))


def _dist(a, b):
    return ((a[0] - b[0]) * 111320) ** 2 + \
           ((a[1] - b[1]) * 111320 * np.cos(np.radians(a[0]))) ** 2


def _dist(a, b):  # noqa: F811
    dy = (a[0] - b[0]) * 111320
    dx = (a[1] - b[1]) * 111320 * np.cos(np.radians(a[0]))
    return float(np.hypot(dx, dy))


if __name__ == "__main__":
    {"observe": do_observe, "sat": do_sat, "solve": do_solve}[sys.argv[1]]()
