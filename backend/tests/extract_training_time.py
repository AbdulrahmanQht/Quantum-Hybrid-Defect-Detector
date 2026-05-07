"""
tests/extract_training_time.py
===================================
Parses logger output files to extract per-epoch wall times for
PR-3.1 (GPU faster than CPU) and PR-3.2 (≤ 4 hours per run).

The fit() methods in all three models log:
    "=== Epoch {n}/{N} ===\\n Time: {dt:.1f}s\\n"

This script reads those log files and produces a summary table.

Usage:
    python tests/extract_training_time.py \\
        --cnn    data/cnn_training.log \\
        --cpu    data/qnn_cpu_training.log \\
        --gpu    data/qnn_gpu_training.log

Output:
    Prints a per-model summary and a PR-3.1 / PR-3.2 verdict.
    Saves results/training_time/training_times.json for inclusion in the audit artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Parser
# ═══════════════════════════════════════════════════════════════════════════
# Pattern matches the logger line: " Time: 123.4s"
_TIME_RE = re.compile(r"Time:\s*([\d.]+)s")
_EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")


def parse_log(path: str) -> Dict:
    """
    Parse a training log file and extract per-epoch times.

    Returns a dict with:
        epoch_times_s  : list of per-epoch wall times (float)
        total_s        : sum of all epoch times
        n_epochs       : number of completed epochs
        mean_epoch_s   : mean epoch time
        total_hours    : total / 3600
    """
    log = Path(path)
    if not log.exists():
        return {"error": f"File not found: {path}"}

    text = log.read_text(encoding="utf-8", errors="replace")
    times = [float(m.group(1)) for m in _TIME_RE.finditer(text)]

    if not times:
        return {"error": f"No 'Time: Xs' entries found in {path}"}

    total_s = sum(times)
    return {
        "epoch_times_s": [round(t, 1) for t in times],
        "n_epochs": len(times),
        "mean_epoch_s": round(total_s / len(times), 1),
        "total_s": round(total_s, 1),
        "total_minutes": round(total_s / 60, 1),
        "total_hours": round(total_s / 3600, 2),
        "min_epoch_s": round(min(times), 1),
        "max_epoch_s": round(max(times), 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PR verdicts
# ═══════════════════════════════════════════════════════════════════════════
def pr31_verdict(cnn: Dict, cpu: Dict, gpu: Dict) -> str:
    """PR-3.1: GPU training must be faster than CPU training."""
    if "error" in gpu or "error" in cpu:
        return "UNVERIFIED (log parse error)"
    gpu_total = gpu["total_s"]
    cpu_total = cpu["total_s"]
    if gpu_total < cpu_total:
        speedup = cpu_total / gpu_total
        return f"PASS — GPU total {gpu['total_minutes']:.1f} min < CPU total {cpu['total_minutes']:.1f} min (×{speedup:.1f} speedup)"
    else:
        return f"FAIL — GPU total {gpu['total_minutes']:.1f} min ≥ CPU total {cpu['total_minutes']:.1f} min"


def pr32_verdict(results: Dict[str, Dict], limit_hours: float = 4.0) -> Dict[str, str]:
    """PR-3.2: Each model's total training must be ≤ ~4 hours."""
    verdicts = {}
    for name, r in results.items():
        if "error" in r:
            verdicts[name] = f"UNVERIFIED ({r['error']})"
        elif r["total_hours"] <= limit_hours:
            verdicts[name] = f"PASS — {r['total_hours']:.2f} h ≤ {limit_hours:.0f} h"
        else:
            verdicts[name] = f"FAIL — {r['total_hours']:.2f} h > {limit_hours:.0f} h"
    return verdicts


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════
def print_report(results: Dict[str, Dict], pr31: str, pr32: Dict[str, str]) -> None:
    print("\n" + "=" * 64)
    print("TRAINING TIME ANALYSIS — PR-3.1 and PR-3.2")
    print("=" * 64)

    for model_name, r in results.items():
        print(f"\n  {model_name}:")
        if "error" in r:
            print(f"    ERROR: {r['error']}")
        else:
            print(f"    Epochs         : {r['n_epochs']}")
            print(f"    Mean epoch time: {r['mean_epoch_s']:.1f} s")
            print(f"    Min / Max epoch: {r['min_epoch_s']:.1f} s / {r['max_epoch_s']:.1f} s")
            print(f"    Total          : {r['total_minutes']:.1f} min  ({r['total_hours']:.2f} h)")

    print(f"\n  PR-3.1 (GPU faster than CPU): {pr31}")
    print("\n  PR-3.2 (each model ≤ 4 hours):")
    for name, v in pr32.items():
        print(f"    {name}: {v}")
    print("=" * 64 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract per-epoch training times for PR-3 compliance."
    )
    parser.add_argument("--cnn", default=None, help="Path to CNN training log file.")
    parser.add_argument("--cpu", default=None, help="Path to QNN-CPU training log file.")
    parser.add_argument("--gpu", default=None, help="Path to QNN-GPU training log file.")
    parser.add_argument(
        "--out",
        default="results/training_time/training_times.json",
        help="Path to save JSON results.",
    )
    args = parser.parse_args()

    if not any([args.cnn, args.cpu, args.gpu]):
        parser.print_help()
        print(
            "\nExample:\n"
            "python tests/extract_training_time.py \\"
                "--cnn    data/cnn_training.log \\"
                "--cpu    data/qnn_cpu_training.log \\"
                "--gpu    data/qnn_gpu_training.log"
        )
        sys.exit(0)

    results: Dict[str, Dict] = {}
    if args.cnn:
        results["CNN"] = parse_log(args.cnn)
    if args.cpu:
        results["QNN_CPU"] = parse_log(args.cpu)
    if args.gpu:
        results["QNN_GPU"] = parse_log(args.gpu)

    cnn_r = results.get("CNN", {"error": "not provided"})
    cpu_r = results.get("QNN_CPU", {"error": "not provided"})
    gpu_r = results.get("QNN_GPU", {"error": "not provided"})

    verdict_31 = pr31_verdict(cnn_r, cpu_r, gpu_r)
    verdicts_32 = pr32_verdict(results)

    print_report(results, verdict_31, verdicts_32)

    # Save to JSON artifact
    output = {
        "model_times": results,
        "pr3_1_verdict": verdict_31,
        "pr3_2_verdicts": verdicts_32,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"Results saved to {args.out}")