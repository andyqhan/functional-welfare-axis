"""GPU utilization, timing, and training metrics logging."""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class GPUMetrics:
    """Snapshot of GPU utilization metrics."""
    gpu_util_pct: float = 0.0
    gpu_mem_used_gb: float = 0.0
    gpu_mem_total_gb: float = 0.0
    gpu_mem_used_pct: float = 0.0
    tensor_core_active_pct: float = 0.0  # estimated from compute throughput


def get_gpu_metrics(device_idx: int | None = None) -> GPUMetrics:
    """Get current GPU utilization metrics via torch.cuda."""
    if not torch.cuda.is_available():
        return GPUMetrics()

    if device_idx is None:
        device_idx = torch.cuda.current_device()

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GPUMetrics(
            gpu_util_pct=util.gpu,
            gpu_mem_used_gb=mem.used / (1024**3),
            gpu_mem_total_gb=mem.total / (1024**3),
            gpu_mem_used_pct=100.0 * mem.used / mem.total,
        )
    except Exception:
        # Fallback to torch.cuda
        mem_used = torch.cuda.memory_allocated(device_idx) / (1024**3)
        mem_total = torch.cuda.get_device_properties(device_idx).total_mem / (1024**3)
        return GPUMetrics(
            gpu_mem_used_gb=mem_used,
            gpu_mem_total_gb=mem_total,
            gpu_mem_used_pct=100.0 * mem_used / mem_total if mem_total > 0 else 0,
        )


class Timer:
    """Accumulated timer for profiling phases."""

    def __init__(self):
        self.timings: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    def start(self, name: str):
        self._starts[name] = time.time()

    def stop(self, name: str) -> float:
        assert name in self._starts, f"Timer '{name}' was never started"
        elapsed = time.time() - self._starts.pop(name)
        self.timings[name] = self.timings.get(name, 0) + elapsed
        return elapsed

    @contextmanager
    def track(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def reset(self):
        self.timings.clear()
        self._starts.clear()


class MetricsLogger:
    """Logs training metrics to JSONL files."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.trajectories_file = self.output_dir / "trajectories.jsonl"
        self.gpu_file = self.output_dir / "gpu_utilization.jsonl"
        self._start_time = time.time()

    def log_step(self, step: int, metrics: dict):
        """Log training metrics for a step."""
        record = {"step": step, "wall_time": time.time(), "data": metrics}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    def log_trajectories(self, step: int, trajectories: list[dict]):
        """Log trajectory summaries for a step."""
        with open(self.trajectories_file, "a") as f:
            for t in trajectories:
                record = {"step": step, **t}
                f.write(json.dumps(record, default=_json_default) + "\n")

    def log_gpu(self, step: int, phase: str, metrics: GPUMetrics):
        """Log GPU utilization snapshot."""
        record = {
            "step": step,
            "phase": phase,
            "wall_time": time.time(),
            "relative_time": time.time() - self._start_time,
            "gpu_util_pct": metrics.gpu_util_pct,
            "gpu_mem_used_gb": metrics.gpu_mem_used_gb,
            "gpu_mem_total_gb": metrics.gpu_mem_total_gb,
            "gpu_mem_used_pct": metrics.gpu_mem_used_pct,
        }
        with open(self.gpu_file, "a") as f:
            f.write(json.dumps(record) + "\n")


def _json_default(obj):
    """JSON serializer for non-standard types."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
