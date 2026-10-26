"""FLTwin pilot dashboard server.

Run:  .venv/bin/python pilot/app.py   (serves on 0.0.0.0:8511)
"""
import base64
import collections
import datetime
import json
import math
import os
import subprocess
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import autocal
import fl511
import pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(BASE, "work")
os.makedirs(WORK, exist_ok=True)
STATE_FILE = os.path.join(WORK, "state.json")
DEFAULT_CAMERA = 817  # I-4 @ MM 76.8 WB, 1280x720 HD, Orlando

app = Flask(__name__, static_folder="static")

# ---------------- logging ring buffer ----------------
LOGS = collections.deque(maxlen=2000)
LOG_LOCK = threading.Lock()
LOG_SEQ = [0]


def log(msg):
    with LOG_LOCK:
        LOG_SEQ[0] += 1
        line = f"{datetime.datetime.now().strftime('%H:%M:%S')}  {msg}"
        LOGS.append((LOG_SEQ[0], line))
    print(line, flush=True)


# ---------------- state ----------------
def default_state():
    return {"camera": None, "pairs": [], "H": None, "H_prev": None,
            "rms_px": None, "sat": None, "hull": None, "cal": None}


STATE = default_state()
if os.path.exists(STATE_FILE):
    try:
        STATE.update(json.load(open(STATE_FILE)))
        log("[state] restored previous session state")
    except Exception as e:
        log(f"[state] could not restore state: {e}")


def save_state():
    json.dump(STATE, open(STATE_FILE, "w"))


# ---------------- live frame grabber ----------------
LIVE = {"on": False, "proc": None}


def live_worker():
    while True:
        if LIVE["on"] and STATE["camera"]:
            p = LIVE["proc"]
            if p is None or p.poll() is not None:
                if p is not None:
                    err = (p.stderr.read() or "")[-300:]
                    if err.strip():
                        log(f"[live] ffmpeg exited: {err.strip()}")
                try:
                    LIVE["proc"] = fl511.live_proc(
                        STATE["camera"]["id"], os.path.join(WORK, "latest.jpg"),
                        log)
                except Exception as e:
                    log(f"[live] failed to start stream: {e}")
                    time.sleep(5)
        else:
            p = LIVE["proc"]
            if p is not None and p.poll() is None:
                p.terminate()
                LIVE["proc"] = None
                log("[live] paused")
        time.sleep(1.5)


threading.Thread(target=live_worker, daemon=True).start()

# ---------------- background jobs (local thread or SLURM GPU node) ----------
JOB = {"running": False, "progress": 0, "total": 1, "stage": "",
       "stats": None, "error": None, "view_change_px": None, "task": None,
       "where": "", "slurm_id": None}

MODEL = {"model": pipeline.MODEL_NAME, "imgsz": pipeline.IMGSZ,
         "conf": pipeline.CONF, "gpu": True}


def gpu_available():
    """A GPU we can actually submit to (this node usually has none allocated)."""
    if pipeline.device().startswith("cuda"):
        return "local"
    try:
        subprocess.run(["sbatch", "--version"], capture_output=True, timeout=10,
                       check=True)
        return "slurm"
    except Exception:
        return None


def submit_task(task, args):
    """Run run_task.py, on a GPU node when we can, otherwise right here."""
    argfile = os.path.join(WORK, f"task_{task}.json")
    json.dump(args, open(argfile, "w"))
    for f in (f"task_{task}_status.json",):
        try:
            os.remove(os.path.join(WORK, f))
        except OSError:
            pass
    where = gpu_available() if MODEL["gpu"] else None
    if where == "slurm":
        try:
            r = subprocess.run(
                ["sbatch", "--parsable", os.path.join(BASE, "gpu_job.sh"),
                 task, argfile],
                capture_output=True, text=True, timeout=30, check=True)
            jid = r.stdout.strip().split(";")[0]
            JOB.update(where=f"SLURM job {jid} (H100)", slurm_id=jid)
            log(f"[job] submitted {task} to SLURM as job {jid} "
                "(waiting for an H100 node)")
            return
        except Exception as e:
            log(f"[job] could not submit to SLURM ({e}); running locally")
    JOB.update(where="this node" + (" (GPU)" if where == "local" else " (CPU)"),
               slurm_id=None)
    log(f"[job] running {task} locally")
    env = dict(os.environ, MPLBACKEND="Agg", YOLO_CONFIG_DIR=WORK)
    subprocess.Popen([os.sys.executable, os.path.join(BASE, "run_task.py"),
                      task, argfile], cwd=BASE, env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def watch_task(task, on_done):
    """Poll the status file the job writes, mirroring it into JOB and the log."""
    path = os.path.join(WORK, f"task_{task}_status.json")
    seen = 0
    t0 = time.time()
    while True:
        time.sleep(1.5)
        if not os.path.exists(path):
            if time.time() - t0 > 1800:
                JOB.update(running=False, error="job never started")
                log("[job] timed out waiting for the job to start")
                return
            if JOB["slurm_id"]:
                JOB["stage"] = "queued on SLURM"
            continue
        try:
            st = json.load(open(path))
        except Exception:
            continue
        for line in st.get("log", [])[seen:]:
            with LOG_LOCK:
                LOG_SEQ[0] += 1
                LOGS.append((LOG_SEQ[0], line))
        seen = len(st.get("log", []))
        JOB.update(stage=st.get("stage", ""), progress=st.get("progress", 0),
                   total=max(1, st.get("total", 1)))
        if st.get("done"):
            if st.get("error"):
                JOB.update(error=st["error"], running=False)
            else:
                try:
                    on_done(st.get("result"))
                    JOB["stage"] = "done"
                except Exception as e:
                    JOB["error"] = str(e)
                    log(f"[job] finishing failed: {e}")
                JOB["running"] = False
            return


def start_job(task, args, on_done):
    JOB.update(running=True, progress=0, total=1, stage="starting", error=None,
               stats=None, task=task)
    submit_task(task, args)
    threading.Thread(target=watch_task, args=(task, on_done),
                     daemon=True).start()


def finish_autocal(res):
    if not res:
        raise RuntimeError("calibration produced no result")
    STATE["H_prev"], STATE["H"] = STATE["H"], res["H"]
    STATE["hull"] = res["zone"]
    STATE["pairs"] = []
    STATE["rms_px"] = None
    STATE["sat"] = {"source": "esri", "georef": res["georef"],
                    "center": res["center"], "zoom": res["zoom"],
                    "grid": res["grid"]}
    STATE["cal"] = {k: res[k] for k in
                    ("camera_latlon", "pan", "tilt", "score", "lane_match",
                     "n_tracks", "n_lane_lines", "vp")}
    STATE["cal"]["height"] = res["pose"].get("height")
    STATE["cal"]["range_m"] = res.get("range_m")
    STATE["cal"]["fov"] = res["pose"].get("fov")
    save_state()
    _regen_overlay()
    log("[cal] calibration stored; check panel 3 of the process image and the "
        "overlay, then adjust with the nudge controls if needed")


def finish_detect(stats):
    JOB["stats"] = stats


def detect_job(seconds, stride):
    """Kick off detection, after checking the camera has not moved."""
    calib = cv2.imread(os.path.join(WORK, "calib.jpg"))
    latest_p = os.path.join(WORK, "latest.jpg")
    if calib is not None and os.path.exists(latest_p):
        cur = cv2.imread(latest_p)
        if cur is not None:
            s = pipeline.view_change_score(calib, cur)
            JOB["view_change_px"] = round(s, 1)
            if s > 10:
                log(f"[warn] view-change score {s:.1f}px — the camera may "
                    "have MOVED since calibration; recalibrate!")
            elif s >= 0:
                log(f"[detect] view stability OK ({s:.1f}px median shift)")
    start_job("detect", {"camera": STATE["camera"], "seconds": seconds,
                         "stride": stride, "H": STATE["H"],
                         "zone": STATE["hull"],
                         "georef": (STATE["sat"] or {}).get("georef"),
                         "model": MODEL["model"], "imgsz": MODEL["imgsz"],
                         "conf": MODEL["conf"]}, finish_detect)


# ---------------- routes ----------------
@app.route("/")
def index():
    resp = send_from_directory("static", "index.html")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.route("/work/<path:p>")
def work_files(p):
    resp = send_from_directory(WORK, p)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/state")
def api_state():
    return jsonify({"camera": STATE["camera"], "pairs": STATE["pairs"],
                    "H": STATE["H"], "H_prev": STATE["H_prev"],
                    "rms_px": STATE["rms_px"], "cal": STATE["cal"],
                    "sat": STATE["sat"], "hull": STATE["hull"],
                    "live": LIVE["on"], "job": JOB, "model": MODEL,
                    "has_calib": os.path.exists(os.path.join(WORK, "calib.jpg")),
                    "has_overlay": os.path.exists(os.path.join(WORK, "overlay.png")),
                    "has_process": os.path.exists(os.path.join(WORK, "autocal_process.png")),
                    "has_result": os.path.exists(os.path.join(WORK, "result.mp4"))})


@app.route("/api/camera", methods=["POST"])
def api_camera():
    cam_id = int(request.json["id"])
    info = fl511.camera_info(cam_id)
    if info is None:
        return jsonify({"error": f"camera {cam_id} not found"}), 404
    STATE.update(default_state())
    STATE["camera"] = info
    for f in ("calib.jpg", "overlay.png", "result.mp4"):
        try:
            os.remove(os.path.join(WORK, f))
        except OSError:
            pass
    save_state()
    log(f"[camera] loaded {cam_id}: {info['name']} "
        f"({info['roadway']} {info['direction']}, {info['county']} Co.) "
        f"@ {info['lat']:.6f},{info['lon']:.6f}")
    return jsonify(info)


@app.route("/api/live", methods=["POST"])
def api_live():
    LIVE["on"] = bool(request.json["on"])
    log(f"[live] {'starting' if LIVE['on'] else 'pausing'} live view")
    return jsonify({"on": LIVE["on"]})


@app.route("/api/freeze", methods=["POST"])
def api_freeze():
    src = os.path.join(WORK, "latest.jpg")
    if not os.path.exists(src):
        try:
            fl511.snapshot(STATE["camera"]["id"], src)
            log("[freeze] live frame unavailable; used FL511 snapshot")
        except Exception as e:
            return jsonify({"error": f"no frame available: {e}"}), 400
    img = cv2.imread(src)
    cv2.imwrite(os.path.join(WORK, "calib.jpg"), img)
    log(f"[freeze] calibration frame frozen ({img.shape[1]}x{img.shape[0]})")
    return jsonify({"ok": True, "w": img.shape[1], "h": img.shape[0]})


@app.route("/api/satellite/fetch", methods=["POST"])
def api_sat_fetch():
    cam = STATE["camera"]
    zoom = int(request.json.get("zoom", 19))
    grid = int(request.json.get("grid", 5))
    lat = float(request.json.get("lat") or cam["lat"])
    lon = float(request.json.get("lon") or cam["lon"])
    try:
        georef = pipeline.fetch_satellite(lat, lon, zoom, grid,
                                          os.path.join(WORK, "sat.png"), log)
    except Exception as e:
        log(f"[sat] fetch failed: {e}")
        return jsonify({"error": str(e)}), 500
    STATE["sat"] = {"source": "esri", "georef": georef,
                    "center": [lat, lon], "zoom": zoom, "grid": grid}
    STATE["pairs"], STATE["H"], STATE["hull"] = [], None, None
    save_state()
    return jsonify(STATE["sat"])


@app.route("/api/satellite/upload", methods=["POST"])
def api_sat_upload():
    data = request.json["image"]  # dataURL
    raw = base64.b64decode(data.split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "could not decode image"}), 400
    cv2.imwrite(os.path.join(WORK, "sat.png"), img)
    STATE["sat"] = {"source": "upload", "georef": None}
    STATE["pairs"], STATE["H"], STATE["hull"] = [], None, None
    save_state()
    log(f"[sat] uploaded satellite image {img.shape[1]}x{img.shape[0]} "
        "(no scale — rectangles will be fixed-size, no speeds)")
    return jsonify(STATE["sat"])


@app.route("/api/satellite/scale", methods=["POST"])
def api_sat_scale():
    """Optional: two clicked points + real distance in metres -> m/px."""
    p = request.json
    d_px = ((p["x1"] - p["x2"]) ** 2 + (p["y1"] - p["y2"]) ** 2) ** 0.5
    if d_px < 2:
        return jsonify({"error": "points too close"}), 400
    mpp = float(p["meters"]) / d_px
    STATE["sat"]["georef"] = {"type": "manual", "mpp": mpp}
    save_state()
    log(f"[sat] manual scale set: {mpp:.3f} m/px")
    return jsonify(STATE["sat"])


@app.route("/api/points", methods=["POST"])
def api_points():
    STATE["pairs"] = request.json["pairs"]
    save_state()
    return jsonify({"n": len(STATE["pairs"])})


@app.route("/api/homography", methods=["POST"])
def api_homography():
    pairs = STATE["pairs"]
    if len(pairs) < 4:
        return jsonify({"error": "need at least 4 point pairs"}), 400
    try:
        H, rms, per_pt = pipeline.compute_homography(pairs)
    except Exception as e:
        log(f"[homography] ERROR: {e}")
        return jsonify({"error": str(e)}), 500
    STATE["H"], STATE["rms_px"] = H.tolist(), round(rms, 2)
    cam_img = cv2.imread(os.path.join(WORK, "calib.jpg"))
    sat_img = cv2.imread(os.path.join(WORK, "sat.png"))
    # Trusted zone from the camera's actual ground footprint, not the convex
    # hull of the clicked points -- that hull is only as big as wherever you
    # happened to click and was throwing away nearly every vehicle.
    hull = None
    mpp_ = ((STATE["sat"] or {}).get("georef") or {}).get("mpp")
    if cam_img is not None and mpp_:
        h_, w_ = cam_img.shape[:2]
        base = cv2.perspectiveTransform(
            np.float32([[[w_ / 2, h_ - 1]]]), np.float64(H)).reshape(2)
        try:
            hull = autocal.view_zone(H, w_, h_, base, mpp_,
                                     max_range_m=260.0)
        except Exception as e:
            log(f"[homography] could not derive the view zone ({e})")
    if hull is None:
        hull = pipeline.zone_hull(pairs)
        log("[homography] using the clicked-point hull as the trusted zone")
    STATE["hull"] = np.asarray(hull).round(1).tolist()
    save_state()
    overlay = pipeline.make_overlay(cam_img, sat_img, H, pairs, STATE["hull"])
    cv2.imwrite(os.path.join(WORK, "overlay.png"), overlay)
    mpp = ((STATE["sat"] or {}).get("georef") or {}).get("mpp")
    msg = f"[homography] computed from {len(pairs)} pairs, RMS error {rms:.1f}px"
    if mpp:
        msg += f" ({rms * mpp:.2f} m)"
    log(msg + f"; per-point px: {per_pt}")
    return jsonify({"rms_px": STATE["rms_px"],
                    "rms_m": round(rms * mpp, 2) if mpp else None,
                    "per_point_px": per_pt})


def _pair_errors(H):
    """Pixel error of each clicked pair under homography H."""
    pairs = STATE["pairs"]
    src = np.float32([[p[0], p[1]] for p in pairs]).reshape(-1, 1, 2)
    dst = np.float32([[p[2], p[3]] for p in pairs])
    proj = cv2.perspectiveTransform(src, np.float32(H)).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1).astype(float)
    return float(np.sqrt((err ** 2).mean())), np.round(err, 1).tolist()


def _rebuild_zone():
    """Re-derive the trusted zone after the homography moved or changed scale.

    Carrying the old polygon through a scale change would silently stretch the
    'trusted' range along with it.
    """
    cal = STATE.get("cal") or {}
    g = (STATE["sat"] or {}).get("georef")
    cam_img = cv2.imread(os.path.join(WORK, "calib.jpg"))
    if not g or cam_img is None or STATE["H"] is None:
        return
    h_, w_ = cam_img.shape[:2]
    if cal.get("camera_latlon"):
        base = autocal.latlon_to_satpx(g, *cal["camera_latlon"])
    else:
        base = cv2.perspectiveTransform(
            np.float32([[[w_ / 2, h_ - 1]]]), np.float64(STATE["H"])).reshape(2)
    try:
        z = autocal.view_zone(STATE["H"], w_, h_, base, g["mpp"],
                              max_range_m=float(cal.get("range_m") or 260))
        if z is not None:
            STATE["hull"] = np.asarray(z).round(1).tolist()
    except Exception as e:
        log(f"[zone] could not rebuild the trusted zone: {e}")


def _regen_overlay():
    cam_img = cv2.imread(os.path.join(WORK, "calib.jpg"))
    sat_img = cv2.imread(os.path.join(WORK, "sat.png"))
    overlay = pipeline.make_overlay(cam_img, sat_img, STATE["H"],
                                    STATE["pairs"], STATE["hull"])
    cv2.imwrite(os.path.join(WORK, "overlay.png"), overlay)


@app.route("/api/homography/refine", methods=["POST"])
def api_refine():
    if STATE["H"] is None:
        return jsonify({"error": "compute a rough homography first "
                                 "(4+ point pairs)"}), 400
    cam_img = cv2.imread(os.path.join(WORK, "calib.jpg"))
    sat_img = cv2.imread(os.path.join(WORK, "sat.png"))
    try:
        Hn, cc = pipeline.refine_homography(cam_img, sat_img, STATE["H"],
                                            STATE["hull"], log)
    except cv2.error:
        log("[refine] ECC did not converge — clicks may be too far off, or "
            "lane markings not visible enough; add/adjust a pair and retry")
        return jsonify({"error": "auto-refine did not converge; adjust your "
                                 "rough points and try again"}), 500
    STATE["H_prev"], STATE["H"] = STATE["H"], Hn.tolist()
    rms, per = _pair_errors(STATE["H"])
    STATE["rms_px"] = round(rms, 2)
    save_state()
    _regen_overlay()
    mpp = ((STATE["sat"] or {}).get("georef") or {}).get("mpp")
    log(f"[refine] refined H saved; your clicks were off by {rms:.1f}px RMS "
        f"({per} px per point) relative to the refined fit")
    return jsonify({"ecc": round(cc, 3), "clicks_rms_px": round(rms, 1),
                    "clicks_rms_m": round(rms * mpp, 2) if mpp else None})


@app.route("/api/homography/undo", methods=["POST"])
def api_refine_undo():
    if STATE["H_prev"] is None:
        return jsonify({"error": "nothing to undo"}), 400
    STATE["H"], STATE["H_prev"] = STATE["H_prev"], None
    rms, _ = _pair_errors(STATE["H"])
    STATE["rms_px"] = round(rms, 2)
    save_state()
    _regen_overlay()
    log("[refine] reverted to the manually computed homography")
    return jsonify({"rms_px": STATE["rms_px"]})


@app.route("/api/autocal", methods=["POST"])
def api_autocal():
    """Fully automatic calibration: no clicking, no correspondences."""
    if JOB["running"]:
        return jsonify({"error": "a job is already running"}), 409
    if not STATE["camera"]:
        return jsonify({"error": "load a camera first"}), 400
    j = request.json or {}
    for f in ("autocal_process.png", "overlay.png"):
        try:
            os.remove(os.path.join(WORK, f))
        except OSError:
            pass
    log("[cal] automatic calibration requested — this records video, finds the "
        "road in the camera and in OpenStreetMap, and solves the geometry")
    start_job("autocal", {"camera": STATE["camera"],
                          "seconds": int(j.get("seconds", 45)),
                          "zoom": int(j.get("zoom", 19)),
                          "grid": int(j.get("grid", 7)),
                          "range_m": float(j.get("range_m", 260)),
                          "model": MODEL["model"], "imgsz": MODEL["imgsz"],
                          "conf": MODEL["conf"]}, finish_autocal)
    return jsonify({"ok": True})


@app.route("/api/model", methods=["POST"])
def api_model():
    j = request.json or {}
    MODEL.update({k: j[k] for k in ("model", "imgsz", "conf") if k in j})
    if "gpu" in j:
        MODEL["gpu"] = bool(j["gpu"])
    info = pipeline.set_model(MODEL["model"], MODEL["imgsz"], MODEL["conf"])
    info["gpu_mode"] = gpu_available() if MODEL["gpu"] else "off"
    log(f"[model] {info['model']} @ {info['imgsz']}px, conf {info['conf']}, "
        f"compute: {info['gpu_mode'] or 'cpu only'}")
    return jsonify(info)


@app.route("/api/homography/nudge", methods=["POST"])
def api_nudge():
    """Manual override: shift / rotate / scale the calibration by hand."""
    if STATE["H"] is None:
        return jsonify({"error": "no homography to adjust"}), 400
    j = request.json or {}
    dx, dy = float(j.get("dx", 0)), float(j.get("dy", 0))
    deg, sc = float(j.get("rot", 0)), float(j.get("scale", 1.0))
    H = np.float64(STATE["H"])
    cam_img = cv2.imread(os.path.join(WORK, "calib.jpg"))
    h, w = cam_img.shape[:2]
    c = cv2.perspectiveTransform(
        np.float32([[[w / 2, h * 0.8]]]), H).reshape(2)
    A = autocal._similarity(float(c[0]), float(c[1]), deg, sc)
    A[0, 2] += dx
    A[1, 2] += dy
    Hn = A @ H
    STATE["H_prev"], STATE["H"] = STATE["H"], (Hn / Hn[2, 2]).tolist()
    if STATE["hull"] is not None:
        STATE["hull"] = cv2.perspectiveTransform(
            np.float32(STATE["hull"]).reshape(-1, 1, 2), A).reshape(-1, 2).tolist()
    _rebuild_zone()
    save_state()
    _regen_overlay()
    mpp = ((STATE["sat"] or {}).get("georef") or {}).get("mpp") or 0
    log(f"[nudge] moved {math.hypot(dx, dy) * mpp:.1f} m, rotated {deg:+.1f}°, "
        f"scaled {sc:.3f}")
    return jsonify({"ok": True})


@app.route("/api/scale_from_speed", methods=["POST"])
def api_scale_speed():
    """Fix the scale using traffic speed as the ruler.

    Everything except scale comes out of the geometry reliably; scale is the
    one quantity an overlap score is blind to.  But we know roughly how fast
    free-flowing traffic on this road moves, and measured speed is exactly
    proportional to the scale -- so one number the user knows fixes it.
    """
    if STATE["H"] is None:
        return jsonify({"error": "calibrate first"}), 400
    det = os.path.join(WORK, "detections.json")
    if not os.path.exists(det):
        return jsonify({"error": "run a detection first so there are speeds "
                                 "to compare against"}), 400
    target = float((request.json or {}).get("target_mph", 65))
    d = json.load(open(det))
    sp = [v["speed_mph"] for f in d.get("frames", []) for v in f["vehicles"]
          if v.get("speed_mph")]
    if len(sp) < 20:
        return jsonify({"error": "not enough moving vehicles to measure "
                                 "speed; record a longer clip"}), 400
    med = float(np.median(sp))
    if med <= 1:
        return jsonify({"error": "measured speeds are ~0"}), 400
    k = target / med
    if not (0.2 < k < 5):
        return jsonify({"error": f"measured median {med:.0f} mph is too far "
                                 f"from {target:.0f} to be a scale error"}), 400
    H = np.float64(STATE["H"])
    cal = STATE.get("cal") or {}
    g = (STATE["sat"] or {}).get("georef")
    if cal.get("camera_latlon") and g:
        cx, cy = autocal.latlon_to_satpx(g, *cal["camera_latlon"])
    else:
        cam_img = cv2.imread(os.path.join(WORK, "calib.jpg"))
        hh, ww = cam_img.shape[:2]
        cx, cy = cv2.perspectiveTransform(
            np.float32([[[ww / 2, hh - 1]]]), H).reshape(2)
    A = autocal._similarity(float(cx), float(cy), 0.0, k)
    Hn = A @ H
    STATE["H_prev"], STATE["H"] = STATE["H"], (Hn / Hn[2, 2]).tolist()
    if cal.get("height"):
        STATE["cal"]["height"] = cal["height"] * k
    _rebuild_zone()
    save_state()
    _regen_overlay()
    log(f"[scale] median measured speed {med:.0f} mph vs {target:.0f} expected "
        f"-> scaled the ground plane by {k:.3f} "
        f"(camera now {STATE['cal'].get('height', 0):.1f} m above the road). "
        "Re-run detection to confirm.")
    return jsonify({"measured_mph": round(med, 1), "factor": round(k, 3),
                    "height_m": round(STATE["cal"].get("height") or 0, 1)})


@app.route("/api/detect", methods=["POST"])
def api_detect():
    if JOB["running"]:
        return jsonify({"error": "a job is already running"}), 409
    if STATE["H"] is None:
        return jsonify({"error": "calibrate first"}), 400
    seconds = int(request.json.get("seconds", 20))
    stride = int(request.json.get("stride", 1))
    try:
        os.remove(os.path.join(WORK, "result.mp4"))
    except OSError:
        pass
    JOB.update(running=True, progress=0, total=1, stage="starting", error=None)
    threading.Thread(target=detect_job, args=(seconds, stride),
                     daemon=True).start()
    log(f"[detect] job started: {seconds}s of video, stride {stride}")
    return jsonify({"ok": True})


@app.route("/api/logs")
def api_logs():
    since = int(request.args.get("since", 0))
    with LOG_LOCK:
        lines = [(i, s) for i, s in LOGS if i > since]
    return jsonify({"lines": [s for _, s in lines],
                    "next": lines[-1][0] if lines else since})


if __name__ == "__main__":
    if STATE["camera"] is None:
        STATE["camera"] = fl511.camera_info(DEFAULT_CAMERA)
        save_state()
    log(f"FLTwin pilot dashboard starting on port 8511 "
        f"(default camera {STATE['camera']['id']}: {STATE['camera']['name']})")
    app.run(host="0.0.0.0", port=8511, debug=False, threaded=True)
