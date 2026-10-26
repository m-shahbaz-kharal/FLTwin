"""FL511 helpers: camera lookup, tokenized stream URL, snapshot, clip recording."""
import json
import os
import re
import subprocess

import requests
import imageio_ffmpeg

BASE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(BASE, "work")
CAMS_FILE = os.path.join(BASE, "data", "cameras_all.json")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

_cams = None


def load_cameras():
    global _cams
    if _cams is None:
        with open(CAMS_FILE) as f:
            _cams = {c["id"]: c for c in json.load(f)}
    return _cams


def camera_info(cam_id):
    """Return a compact dict for one camera, or None."""
    c = load_cameras().get(int(cam_id))
    if c is None:
        return None
    m = re.search(r"POINT \(([-0-9.]+) ([-0-9.]+)\)",
                  c["latLng"]["geography"]["wellKnownText"])
    lon, lat = float(m.group(1)), float(m.group(2))
    img = c["images"][0] if c.get("images") else {}
    return {
        "id": c["id"],
        "name": c.get("location") or img.get("description", ""),
        "roadway": c.get("roadway"),
        "direction": c.get("direction"),
        "county": c.get("county"),
        "region": c.get("region"),
        "lat": lat,
        "lon": lon,
        "videoUrl": img.get("videoUrl") or "",
        "videoAuth": bool(img.get("isVideoAuthRequired")),
        "snapshotUrl": f"https://fl511.com/map/Cctv/{c['id']}",
    }


def stream_url(cam_id, log=print):
    """Resolve the playable (tokenized) HLS URL for a camera."""
    info = camera_info(cam_id)
    if not info or not info["videoUrl"]:
        raise RuntimeError(f"camera {cam_id} has no video URL")
    url = info["videoUrl"]
    if not info["videoAuth"]:
        return url
    r = requests.get(f"https://fl511.com/Camera/GetVideoUrl?imageId={cam_id}",
                     headers={"User-Agent": UA, "Referer": "https://fl511.com/map"},
                     timeout=15)
    r.raise_for_status()
    tok = r.json()
    if isinstance(tok, dict):
        r2 = requests.post(
            "https://divas.cloud/VDS-API/SecureTokenUri/GetSecureTokenUriBySourceId",
            json=tok,
            headers={"User-Agent": UA, "Origin": "https://fl511.com",
                     "Referer": "https://fl511.com/"},
            timeout=15)
        r2.raise_for_status()
        suffix = r2.json()  # a string like "?token=..."
        url = url + suffix
    else:
        url = url + str(tok)
    log(f"[fl511] resolved stream url for camera {cam_id}")
    return url


def snapshot(cam_id, out_path):
    """Save the FL511 static snapshot JPEG (fallback when stream is down)."""
    r = requests.get(f"https://fl511.com/map/Cctv/{cam_id}",
                     headers={"User-Agent": UA}, timeout=15)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


def _hls_args(url):
    return ["-user_agent", UA, "-headers", "Referer: https://fl511.com/\r\n",
            "-i", url]


def live_proc(cam_id, out_jpg, log=print):
    """Start ffmpeg writing ~1 fps JPEG frames of the live stream to out_jpg."""
    url = stream_url(cam_id, log)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "warning", *_hls_args(url),
           "-vf", "fps=1", "-update", "1", "-y", out_jpg]
    log(f"[fl511] starting live frame grabber (camera {cam_id})")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)


def record_clip(cam_id, seconds, out_path, log=print):
    """Record `seconds` of the live stream to out_path (stream copy, no re-encode)."""
    url = stream_url(cam_id, log)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "warning", *_hls_args(url),
           "-t", str(int(seconds)), "-c", "copy", "-movflags", "+faststart",
           "-y", out_path]
    log(f"[fl511] recording {seconds}s clip from camera {cam_id} ...")
    p = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=int(seconds) + 90)
    if p.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg record failed: {p.stderr[-400:]}")
    log(f"[fl511] clip saved: {out_path} "
        f"({os.path.getsize(out_path) // 1024} KB)")
    return out_path
