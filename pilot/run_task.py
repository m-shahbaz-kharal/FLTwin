"""Heavy jobs (auto-calibration, detection), runnable locally or on a GPU node.

    python run_task.py autocal work/task_autocal.json
    python run_task.py detect  work/task_detect.json

The dashboard writes the input JSON, launches this (directly or through
sbatch), and watches the matching *_status.json for progress.  Keeping it a
separate script is what lets the same code run on a compute node with an H100
while the dashboard stays put on the login/CPU node.
"""
import json
import os
import sys
import time
import traceback

BASE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(BASE, "work")


def status_path(task):
    return os.path.join(WORK, f"task_{task}_status.json")


def make_writer(task):
    state = {"stage": "starting", "progress": 0, "total": 1, "done": False,
             "error": None, "log": [], "result": None,
             "started": time.time()}

    def write():
        tmp = status_path(task) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, status_path(task))

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        state["log"].append(line)
        del state["log"][:-400]
        print(line, flush=True)
        write()

    return state, log, write


def main():
    task, argfile = sys.argv[1], sys.argv[2]
    args = json.load(open(argfile))
    state, log, write = make_writer(task)
    try:
        import pipeline
        pipeline.set_model(args.get("model"), args.get("imgsz"),
                           args.get("conf"))
        log(f"[job] {task} starting on {os.uname().nodename}, "
            f"device {pipeline.device()}, model {pipeline.MODEL_NAME}")
        if task == "autocal":
            run_autocal(args, state, log, write)
        elif task == "detect":
            run_detect(args, state, log, write)
        else:
            raise ValueError(f"unknown task {task}")
        state["stage"] = "done"
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
        log(f"[job] ERROR: {state['error']}")
        traceback.print_exc()
    finally:
        state["done"] = True
        write()


def run_autocal(args, state, log, write):
    import autocal
    import fl511
    clip = os.path.join(WORK, "autocal_clip.mp4")
    if args.get("record", True):
        state["stage"] = "recording video"
        write()
        fl511.record_clip(args["camera"]["id"], int(args.get("seconds", 45)),
                          clip, log)
    if not os.path.exists(clip):
        raise RuntimeError("no video captured for calibration")

    def prog(stage):
        state["stage"] = stage
        write()

    out = autocal.auto_calibrate(clip, args["camera"], WORK,
                                 zoom=int(args.get("zoom", 19)),
                                 grid=int(args.get("grid", 7)),
                                 max_range_m=float(args.get("range_m", 260)),
                                 log=log, progress=prog)
    state["result"] = out
    write()


def run_detect(args, state, log, write):
    import pipeline
    import fl511
    clip = os.path.join(WORK, "clip.mp4")
    if args.get("record", True):
        state["stage"] = "recording video"
        write()
        fl511.record_clip(args["camera"]["id"], int(args.get("seconds", 20)),
                          clip, log)
    state["stage"] = "detecting"
    write()

    def prog(done, total):
        state["progress"], state["total"] = done, total
        write()

    stats = pipeline.run_detection(
        clip, args["H"], args.get("zone"), os.path.join(WORK, "sat.png"),
        args.get("georef"), WORK, stride=int(args.get("stride", 3)),
        log=log, progress=prog)
    state["result"] = stats
    write()


if __name__ == "__main__":
    main()
