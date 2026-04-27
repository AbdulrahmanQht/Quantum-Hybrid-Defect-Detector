"""
tests/monitor_gpu_memory.py
==============================
GPU memory utilization monitor for PR-5.2.
Requirement: GPU memory must not exceed 90% during training.

Usage modes:

1. Hook into a training run (recommended):
   Add to your training loop:
       from tests.monitor_gpu_memory import GpuMemoryMonitor
       monitor = GpuMemoryMonitor(threshold=0.90, log_path="data/results_tests/gpu_memory.csv")
       # In epoch loop:
       monitor.sample(label=f"epoch_{epoch}_start")
       # ... train ...
       monitor.sample(label=f"epoch_{epoch}_end")
   After training:
       monitor.report()
       monitor.assert_pr52()   # raises AssertionError if any sample exceeded 90%

2. Standalone sidecar process (monitors externally, no code changes needed):
       python tests/monitor_gpu_memory.py --interval 5 --duration 14400
   (samples every 5 s for 4 hours)

3. Single snapshot:
       python tests/monitor_gpu_memory.py --snapshot
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Detect torch / CUDA availability at import time
# ---------------------------------------------------------------------------
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import subprocess

    _NVIDIASMI_AVAILABLE = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
        ).returncode
        == 0
    )
except Exception:
    _NVIDIASMI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class GpuSample:
    timestamp: str
    label: str
    device_index: int
    reserved_gb: float
    allocated_gb: float
    total_gb: float
    utilisation_pct: float  # reserved / total × 100
    allocated_pct: float    # allocated / total × 100
    exceeds_threshold: bool


@dataclass
class GpuMemoryReport:
    samples: List[GpuSample] = field(default_factory=list)
    peak_reserved_gb: float = 0.0
    peak_utilisation_pct: float = 0.0
    threshold_violations: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Core monitor class
# ═══════════════════════════════════════════════════════════════════════════
class GpuMemoryMonitor:
    """
    Collects per-sample GPU memory snapshots during training.
    Attach to the training loop via `sample()` calls.

    Parameters
    ----------
    threshold : float
        Fraction of total GPU memory (PR-5.2: 0.90 = 90%).
    device_index : int
        CUDA device index to monitor.
    log_path : str or None
        If provided, every sample is appended to this CSV file in real-time.
    """

    def __init__(
        self,
        threshold: float = 0.90,
        device_index: int = 0,
        log_path: Optional[str] = None,
    ) -> None:
        self.threshold = threshold
        self.device_index = device_index
        self.log_path = log_path
        self._report = GpuMemoryReport()
        self._csv_initialized = False

        if not _TORCH_AVAILABLE:
            print("[GpuMemoryMonitor] WARNING: torch not available; monitoring disabled.")
        elif not torch.cuda.is_available():
            print("[GpuMemoryMonitor] WARNING: CUDA not available; monitoring disabled.")

    def _get_snapshot(self, label: str) -> Optional[GpuSample]:
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return None

        props = torch.cuda.get_device_properties(self.device_index)
        total_bytes = props.total_memory
        reserved_bytes = torch.cuda.memory_reserved(self.device_index)
        allocated_bytes = torch.cuda.memory_allocated(self.device_index)

        utilisation = reserved_bytes / total_bytes
        allocated_pct = allocated_bytes / total_bytes

        return GpuSample(
            timestamp=datetime.utcnow().isoformat(),
            label=label,
            device_index=self.device_index,
            reserved_gb=round(reserved_bytes / 1e9, 3),
            allocated_gb=round(allocated_bytes / 1e9, 3),
            total_gb=round(total_bytes / 1e9, 3),
            utilisation_pct=round(utilisation * 100, 2),
            allocated_pct=round(allocated_pct * 100, 2),
            exceeds_threshold=utilisation > self.threshold,
        )

    def _write_csv_row(self, sample: GpuSample) -> None:
        if not self.log_path:
            return
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or not self._csv_initialized
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=asdict(sample).keys())
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(sample))
        self._csv_initialized = True

    def sample(self, label: str = "") -> Optional[GpuSample]:
        """Take a single memory snapshot. Call from inside your training loop."""
        snap = self._get_snapshot(label)
        if snap is None:
            return None

        self._report.samples.append(snap)
        self._write_csv_row(snap)

        # Update rolling stats
        if snap.reserved_gb > self._report.peak_reserved_gb:
            self._report.peak_reserved_gb = snap.reserved_gb
        if snap.utilisation_pct > self._report.peak_utilisation_pct:
            self._report.peak_utilisation_pct = snap.utilisation_pct
        if snap.exceeds_threshold:
            self._report.threshold_violations += 1

        # Real-time console warning
        flag = "WARN" if snap.exceeds_threshold else "    "
        print(
            f"[GPU{snap.device_index}|{flag}] {label or 'snapshot'}: "
            f"reserved={snap.reserved_gb:.2f} GB "
            f"({snap.utilisation_pct:.1f}%) / "
            f"total={snap.total_gb:.2f} GB"
        )
        return snap

    def report(self) -> GpuMemoryReport:
        """Print a summary table and return the report object."""
        r = self._report
        print("\n" + "=" * 60)
        print("GPU MEMORY MONITOR — PR-5.2 REPORT")
        print("=" * 60)
        print(f"  Samples collected : {len(r.samples)}")
        print(f"  Peak reserved     : {r.peak_reserved_gb:.3f} GB")
        print(f"  Peak utilisation  : {r.peak_utilisation_pct:.1f}%")
        print(f"  PR-5.2 threshold  : {self.threshold * 100:.0f}%")
        print(f"  Violations        : {r.threshold_violations}")
        if r.threshold_violations == 0:
            print("  [PASS] PR-5.2: GPU memory stayed below threshold")
        else:
            print(f"  [FAIL] PR-5.2: {r.threshold_violations} samples exceeded threshold")
        print("=" * 60 + "\n")
        return r

    def assert_pr52(self) -> None:
        """Raise AssertionError if any sample exceeded the configured threshold."""
        if self._report.threshold_violations > 0:
            raise AssertionError(
                f"PR-5.2 FAIL: GPU memory exceeded {self.threshold * 100:.0f}% "
                f"threshold on {self._report.threshold_violations} / "
                f"{len(self._report.samples)} samples. "
                f"Peak: {self._report.peak_utilisation_pct:.1f}%. "
                f"See {self.log_path or 'console output'} for details."
            )


# ═══════════════════════════════════════════════════════════════════════════
# Sidecar mode — external continuous monitor via nvidia-smi
# ═══════════════════════════════════════════════════════════════════════════
def _sidecar_monitor(
    interval_s: int = 5,
    duration_s: int = 3600,
    threshold_pct: float = 90.0,
    log_path: str = "data/results_tests/gpu_sidecar.csv",
) -> None:
    """
    External monitor: reads GPU stats from nvidia-smi at `interval_s` intervals
    for `duration_s` seconds total.  Does not require torch.
    """
    import subprocess

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    violations = 0
    samples = 0
    start = time.time()

    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["timestamp", "index", "name", "used_mb", "total_mb",
             "util_gpu_pct", "used_pct", "exceeds_threshold"]
        )

        print(f"[SIDECAR] Monitoring GPU every {interval_s}s for {duration_s}s → {log_path}")
        while time.time() - start < duration_s:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 5:
                        continue
                    ts, idx, name, used, total, util = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5] if len(parts) > 5 else "0"
                    used_pct = round(float(used) / max(float(total), 1) * 100, 1)
                    exceeds = used_pct > threshold_pct
                    writer.writerow([ts, idx, name, used, total, util, used_pct, exceeds])
                    fh.flush()
                    samples += 1
                    if exceeds:
                        violations += 1
                        print(
                            f"[WARN] GPU{idx} memory: {used_pct:.1f}% "
                            f"({used} / {total} MB) — exceeds {threshold_pct:.0f}%"
                        )
            except Exception as exc:
                print(f"[SIDECAR] nvidia-smi error: {exc}")

            time.sleep(interval_s)

    print(f"\n[SIDECAR] Done. {samples} samples, {violations} threshold violations.")
    if violations == 0:
        print(f"[PASS] PR-5.2: GPU memory never exceeded {threshold_pct:.0f}%")
    else:
        print(f"[FAIL] PR-5.2: {violations} violations — see {log_path}")


# ═══════════════════════════════════════════════════════════════════════════
# pytest fixture — use this in test_suite_extended.py for GAP-15
# ═══════════════════════════════════════════════════════════════════════════
try:
    import pytest

    @pytest.fixture
    def gpu_memory_monitor(tmp_path):
        """
        Pytest fixture providing a GpuMemoryMonitor.
        Usage:
            def test_training_step_memory(gpu_memory_monitor):
                monitor = gpu_memory_monitor
                monitor.sample("before_forward")
                # run your forward pass
                monitor.sample("after_backward")
                monitor.assert_pr52()
        """
        log = str(tmp_path / "gpu_memory.csv")
        return GpuMemoryMonitor(threshold=0.90, log_path=log)

except ImportError:
    pass  # pytest not installed; skip fixture registration


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU memory monitor for PR-5.2 compliance."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--snapshot",
        action="store_true",
        help="Take a single snapshot and exit.",
    )
    group.add_argument(
        "--sidecar",
        action="store_true",
        help="Run external nvidia-smi sidecar monitor (no torch needed).",
    )
    parser.add_argument(
        "--interval", type=int, default=5, help="Sampling interval in seconds (sidecar mode)."
    )
    parser.add_argument(
        "--duration", type=int, default=3600, help="Total monitoring duration in seconds."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90, help="Violation threshold (0.0–1.0)."
    )
    parser.add_argument(
        "--log", default="data/results_tests/gpu_memory.csv", help="CSV output path."
    )

    args = parser.parse_args()

    if args.snapshot:
        monitor = GpuMemoryMonitor(threshold=args.threshold, log_path=args.log)
        snap = monitor.sample("manual_snapshot")
        if snap is None:
            print("No CUDA device available or torch not installed.")
            sys.exit(1)
        monitor.report()

    elif args.sidecar:
        if not _NVIDIASMI_AVAILABLE:
            print("ERROR: nvidia-smi not found. Install NVIDIA drivers.")
            sys.exit(1)
        _sidecar_monitor(
            interval_s=args.interval,
            duration_s=args.duration,
            threshold_pct=args.threshold * 100,
            log_path=args.log,
        )

    else:
        # Default: in-process monitoring example
        print("Usage: python tests/monitor_gpu_memory.py --snapshot")
        print("       python tests/monitor_gpu_memory.py --sidecar --interval 5")
        print("       python tests/monitor_gpu_memory.py --help")
        parser.print_help()