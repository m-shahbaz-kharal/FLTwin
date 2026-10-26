# FLTwin

A digital twin of Florida's live traffic cameras: detect vehicles in an
[FL511](https://fl511.com) camera stream, project them onto georeferenced
satellite imagery, and read off each vehicle's position, heading and speed in
real-world units.

The hard part is calibration. FL511 publishes no camera poses, and its camera
coordinates are often hundreds of metres out. FLTwin **calibrates each camera
automatically** by matching the road it sees against OpenStreetMap roads on the
satellite tile. Nobody has to click point pairs.

## Status

One camera, end to end: the pilot in [`pilot/`](pilot). It's a Flask dashboard
that loads a camera, auto-calibrates it in 3–5 minutes on a GPU node, then
records and runs YOLO11 + ByteTrack detection projected onto the map. Camera
817 (I-4 @ MM 76.8 WB, Orlando) is the default and the reference case.

[`pilot/README.md`](pilot/README.md) covers the workflow, how the automatic
calibration works, why scale is the weak point and how traffic speed fixes it,
and the known gotchas.

## Layout

```
pilot/app.py            Flask dashboard, state, SLURM job submission
pilot/autocal.py        automatic calibration (observe → OSM → pose search → lock-on)
pilot/pipeline.py       satellite tiles, homography, YOLO detection, rendering
pilot/fl511.py          FL511 camera list, stream tokens, recording
pilot/run_task.py       heavy jobs (calibration, detection), runnable on a GPU node
pilot/gpu_job.sh        sbatch wrapper for run_task.py
pilot/test_autocal.py   offline harness for calibration development
pilot/static/           dashboard front end
pilot/data/             FL511 camera catalogue
```

## Quick start

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

cd pilot
MPLBACKEND=Agg YOLO_CONFIG_DIR=work ../.venv/bin/python app.py   # serves on :8511
```

YOLO11 weights download on first use. Runtime output (recorded clips, overlays,
logs, job state) goes to `pilot/work/`, which is not tracked.
