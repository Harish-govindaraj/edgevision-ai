"""System hardware and resource utilization metrics for edge performance profiling."""

from dataclasses import dataclass
import shutil
import subprocess
from typing import Dict, Optional
import psutil
import torch


@dataclass
class SystemMetrics:
    """Snapshot of hardware resource utilization."""

    cpu_utilization_percent: float
    ram_used_mb: float
    ram_total_mb: float
    ram_utilization_percent: float
    gpu_name: str = "N/A"
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None


def get_gpu_metrics() -> Dict[str, Optional[float]]:
    """
    Attempt to read GPU metrics via torch.cuda or nvidia-smi.

    Returns dict with keys: 'name', 'utilization', 'memory_used', 'memory_total'.
    Falls back gracefully to N/A when unavailable.
    """
    metrics = {
        "name": "N/A",
        "utilization": None,
        "memory_used": None,
        "memory_total": None,
    }

    # 1. Try PyTorch CUDA if active
    if torch.cuda.is_available():
        try:
            metrics["name"] = torch.cuda.get_device_name(0)
            mem_alloc = torch.cuda.memory_allocated(0) / (1024.0 * 1024.0)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024.0 * 1024.0)
            metrics["memory_used"] = float(mem_alloc)
            metrics["memory_total"] = float(mem_total)
            return metrics
        except Exception:
            pass

    # 2. Try nvidia-smi CLI query
    if shutil.which("nvidia-smi") is not None:
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            output = subprocess.check_output(cmd, encoding="utf-8", timeout=2).strip()
            if output:
                parts = [p.strip() for p in output.split(",")]
                if len(parts) >= 4:
                    metrics["name"] = parts[0]
                    metrics["utilization"] = float(parts[1]) if parts[1] != "[Not Supported]" else None
                    metrics["memory_used"] = float(parts[2])
                    metrics["memory_total"] = float(parts[3])
                    return metrics
        except Exception:
            pass

    return metrics


def get_system_metrics() -> SystemMetrics:
    """Capture current system CPU, RAM, and GPU resource utilization."""
    cpu_pct = float(psutil.cpu_percent(interval=None))
    vm = psutil.virtual_memory()
    ram_used = float(vm.used / (1024.0 * 1024.0))
    ram_total = float(vm.total / (1024.0 * 1024.0))
    ram_pct = float(vm.percent)

    gpu = get_gpu_metrics()

    return SystemMetrics(
        cpu_utilization_percent=cpu_pct,
        ram_used_mb=ram_used,
        ram_total_mb=ram_total,
        ram_utilization_percent=ram_pct,
        gpu_name=str(gpu["name"]),
        gpu_utilization_percent=gpu["utilization"],
        gpu_memory_used_mb=gpu["memory_used"],
        gpu_memory_total_mb=gpu["memory_total"],
    )
