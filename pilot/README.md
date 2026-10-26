# FLTwin pilot — one camera, automatically calibrated

A digital twin of one FL511 traffic camera: detect vehicles, project them onto
georeferenced satellite imagery, and read off positions, headings and speeds.

## Run it

```bash
cd pilot
MPLBACKEND=Agg YOLO_CONFIG_DIR=work ../.venv/bin/python app.py
```

Then tunnel from your laptop and open <http://localhost:8511>:

```bash
ssh -L 8511:evc21:8511 <login-node>       # replace evc21 with the node it runs on
```

## Workflow

1. **Load camera** (defaults to 817, I-4 @ MM 76.8 WB) and press ▶ Live.
2. **🤖 Auto-calibrate.** No clicking. It records ~45 s, then works out where the
   camera is and where it points. Takes 3–5 minutes, mostly on a GPU node.
3. **Check it.** Expand *"show how the calibration was worked out"* and look at
   panel 3 — the camera view projected onto the satellite with the trusted zone.
   The camera position is also linked into Google Maps.
4. **Fix the scale.** Run a detection, then press **📏 Fix scale from traffic
   speed** with the road's free-flow speed. See "About scale" below.
5. **⏺ Record & detect.** Keep frame stride at 1 — the tracker needs
   consecutive frames, and a GPU makes it cheap.

If the automatic result is wrong, the manual override is right there: shift /
rotate / scale buttons, or section 2b for the old click-4-pairs workflow.

## How the automatic calibration works

The camera and the satellite share one thing that is easy to find in both: the
road. So it matches road structure, not appearance — appearance matching across
a 90° viewpoint change does not work.

1. **Watch the camera** (`autocal.observe`). A median background removes traffic
   and leaves clean pavement. Vehicle tracks give the road's **vanishing point**;
   they also label pavement for free, which seeds a colour model that segments
   the whole road surface. A top-hat filter extracts the lane markings.
2. **Sharpen the vanishing point** on the lane paint. Every lane line is radial
   from it, so the right point is the one that makes the angle histogram most
   peaked. Lane lines then fall out as peaks of that histogram — dashes included,
   which is why this beats running a line detector over broken markings.
3. **Read the map.** OpenStreetMap gives the roads around the camera, rasterised
   into the satellite tile. FL511's own coordinates are often hundreds of metres
   out (camera 817's are 585 m out, in a housing estate), so the search is
   centred on the nearest matching road instead.
4. **Solve the pose.** The vanishing point plus the road's OSM bearing fix pan
   and tilt *exactly*, leaving only position, height and focal length. Moving a
   camera over flat ground just translates its ground projection, so position is
   solved for every offset at once by normalised cross-correlation instead of
   being searched. Nelder–Mead then refines, scored in both directions so that
   shrinking the view onto a convenient patch of tarmac cannot win.
5. **Check the direction.** A road looks much the same from either end, so the
   travel direction from the tracks is checked against FL511's stated direction.

`autocal_process.png` shows all of this in four panels.

## About scale

Everything above is reliable except **scale**, and it is worth knowing why: an
overlap score is nearly blind to it, because a slightly smaller footprint always
sits a little more neatly inside the road. Scale has to come from something
measured in metres. Two sources are used — the tracked cars (a ~1.85 m ruler
lying on the road) and the lane grid (12 ft lanes) — but both are shaky at
night, when vehicles are headlight blobs.

The dependable fix is traffic speed: measured speed is exactly proportional to
scale. Run a detection, enter the road's free-flow speed, press **📏 Fix scale
from traffic speed**. On camera 817 this took a run reading 27 mph to 61.6 mph
in one press. Re-run detection to confirm.

## GPU

Detection and calibration run through `run_task.py`, submitted to SLURM by
`gpu_job.sh`. The dashboard stays on its own node and watches a status file.

The job asks for `--gres=gpu:nvidia_h100_pcie:1` **on purpose**: the installed
torch is a cu128 build with no kernels for the cluster's V100s (compute
capability 7.0), so a plain `--gres=gpu:1` can land on a node where CUDA dies at
model load with `no kernel image is available`.

YOLO11x is ~13.6 ms/frame on an H100 versus **68 s/frame on one CPU core**.

Untick "use GPU" to run locally on CPU instead (slow, but no queue).

## Files

| file | what it does |
| --- | --- |
| `app.py` | Flask dashboard, state, job submission |
| `autocal.py` | automatic calibration (observe → OSM → pose search → lock-on) |
| `pipeline.py` | satellite tiles, homography, YOLO detection, rendering |
| `fl511.py` | FL511 camera list, stream tokens, recording |
| `run_task.py` | heavy jobs, runnable on a GPU node |
| `gpu_job.sh` | sbatch wrapper |
| `test_autocal.py` | offline harness for calibration development |

## Gotchas

- **FL511 coordinates lie.** Treat them as a hint, not a position.
- **Frame stride > 1 breaks tracking.** ByteTrack needs consecutive frames;
  at stride 3 most detections never get confirmed and used to be discarded.
- **The trusted zone is derived from the camera's footprint**, not from clicked
  points. The old click-hull was ~470 m² and threw away 25 of 26 vehicles.
- **PTZ cameras move.** Each detection run reports a view-change score; over
  ~10 px, recalibrate.
