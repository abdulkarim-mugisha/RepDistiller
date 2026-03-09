#!/usr/bin/env python
"""
Export TensorBoard scalar data from save/student_tensorboards/ to JSON and CSV.

Usage:
    python scripts/export_tensorboard.py [--logdir DIR] [--outdir DIR]

Output:
    - experiments.json: all runs, {run_name: {tag: [{step, value}, ...]}}
    - experiments.csv: long format, run_name, step, tag, value
    - experiments/: one CSV per run (step, tag, value)
"""

from __future__ import print_function

import argparse
import json
import os
import struct
import sys

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorboard_logger.tensorboard_logger import event_pb2


def read_events(logdir):
    """Parse all event files in logdir. Returns {tag: [(step, value), ...]}."""
    data = {}
    for fn in sorted(os.listdir(logdir)):
        if not fn.startswith("events.out.tfevents"):
            continue
        path = os.path.join(logdir, fn)
        with open(path, "rb") as f:
            while True:
                header = f.read(12)
                if len(header) < 12:
                    break
                data_len = struct.unpack("Q", header[:8])[0]
                d = f.read(data_len)
                _ = f.read(4)
                if len(d) < data_len:
                    break
                event = event_pb2.Event()
                event.ParseFromString(d)
                if event.HasField("summary"):
                    for v in event.summary.value:
                        tag = v.tag
                        step = int(event.step)
                        val = float(v.simple_value)
                        if tag not in data:
                            data[tag] = []
                        data[tag].append((step, val))
    for tag in data:
        data[tag] = sorted(data[tag])
    return data


def main():
    parser = argparse.ArgumentParser(description="Export TensorBoard data to JSON/CSV")
    parser.add_argument(
        "--logdir",
        default="save/student_tensorboards",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--outdir",
        default="save/tensorboard_export",
        help="Output directory for JSON and CSV",
    )
    args = parser.parse_args()

    logdir = args.logdir
    outdir = args.outdir

    if not os.path.isdir(logdir):
        print("Error: logdir not found:", logdir)
        sys.exit(1)

    os.makedirs(outdir, exist_ok=True)

    all_runs = {}
    rows = []

    for run_name in sorted(os.listdir(logdir)):
        run_path = os.path.join(logdir, run_name)
        if not os.path.isdir(run_path):
            continue
        try:
            data = read_events(run_path)
        except Exception as e:
            print("Warning: failed to read {}: {}".format(run_name, e))
            continue
        if not data:
            print("Warning: no events in", run_name)
            continue

        all_runs[run_name] = {
            tag: [{"step": s, "value": v} for s, v in pairs]
            for tag, pairs in data.items()
        }

        for tag, pairs in data.items():
            for step, value in pairs:
                rows.append({"run": run_name, "step": step, "tag": tag, "value": value})

    json_path = os.path.join(outdir, "experiments.json")
    with open(json_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print("Wrote", json_path)

    csv_path = os.path.join(outdir, "experiments.csv")
    with open(csv_path, "w") as f:
        f.write("run,step,tag,value\n")
        for r in rows:
            f.write("{},{},{},{}\n".format(r["run"], r["step"], r["tag"], r["value"]))
    print("Wrote", csv_path)

    per_run_dir = os.path.join(outdir, "runs")
    os.makedirs(per_run_dir, exist_ok=True)
    for run_name, data in all_runs.items():
        safe_name = run_name.replace("/", "_").replace(":", "_")
        run_csv = os.path.join(per_run_dir, "{}.csv".format(safe_name))
        with open(run_csv, "w") as f:
            f.write("step,tag,value\n")
            for tag, pairs in data.items():
                for p in pairs:
                    f.write("{},{},{}\n".format(p["step"], tag, p["value"]))
        print("Wrote", run_csv)

    print("Done. {} runs exported.".format(len(all_runs)))


if __name__ == "__main__":
    main()
