"""perf_monitor.py — Theo dõi hiệu suất CPU / RAM / GPU theo thời gian thực.

Dùng psutil để đọc CPU và RAM.
Dùng GPUtil để đọc NVIDIA GPU (nếu có), fallback graceful nếu không có GPU.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GpuStats:
    index: int
    name: str
    load_percent: float       # 0–100
    mem_used_mb: float
    mem_total_mb: float
    temperature: float        # Celsius, 0 nếu không đọc được


@dataclass
class PerfSnapshot:
    cpu_percent: float        # 0–100
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float        # 0–100
    gpus: list[GpuStats] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)

    # Tiện ích
    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def primary_gpu(self) -> Optional[GpuStats]:
        return self.gpus[0] if self.gpus else None


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def _read_cpu_ram() -> tuple[float, float, float, float]:
    """Trả về (cpu_percent, ram_used_gb, ram_total_gb, ram_percent)."""
    try:
        import psutil  # type: ignore
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used = mem.used / (1024 ** 3)
        ram_total = mem.total / (1024 ** 3)
        ram_pct = mem.percent
        return cpu, ram_used, ram_total, ram_pct
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def _read_gpus() -> list[GpuStats]:
    """Trả về danh sách GPU NVIDIA. Trả về [] nếu không có hoặc không hỗ trợ."""
    try:
        import GPUtil  # type: ignore
        gpus = GPUtil.getGPUs()
        result: list[GpuStats] = []
        for gpu in gpus:
            result.append(
                GpuStats(
                    index=gpu.id,
                    name=gpu.name,
                    load_percent=round(gpu.load * 100, 1),
                    mem_used_mb=round(gpu.memoryUsed, 1),
                    mem_total_mb=round(gpu.memoryTotal, 1),
                    temperature=round(gpu.temperature or 0.0, 1),
                )
            )
        return result
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Background sampler
# ---------------------------------------------------------------------------

class PerfSampler:
    """Thread nền lấy mẫu CPU/RAM/GPU mỗi `interval` giây.

    Sử dụng:
        sampler = PerfSampler(interval=0.5)
        sampler.start()
        ...
        snap = sampler.snapshot()
        print(snap.cpu_percent, snap.primary_gpu)
        ...
        sampler.stop()
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._snapshot: PerfSnapshot = PerfSnapshot(
            cpu_percent=0.0, ram_used_gb=0.0, ram_total_gb=0.0, ram_percent=0.0
        )
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Khởi tạo psutil trước để lần đầu cpu_percent() không trả về 0
        try:
            import psutil  # type: ignore
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bắt đầu thread nền."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Dừng thread nền."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def snapshot(self) -> PerfSnapshot:
        """Trả về snapshot hiệu suất mới nhất (thread-safe)."""
        with self._lock:
            return self._snapshot

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            cpu, ram_used, ram_total, ram_pct = _read_cpu_ram()
            gpus = _read_gpus()
            snap = PerfSnapshot(
                cpu_percent=cpu,
                ram_used_gb=ram_used,
                ram_total_gb=ram_total,
                ram_percent=ram_pct,
                gpus=gpus,
            )
            with self._lock:
                self._snapshot = snap
            self._stop_event.wait(self.interval)
