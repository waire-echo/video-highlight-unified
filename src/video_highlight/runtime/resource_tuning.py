"""Choose conservative runtime settings from current CPU, RAM and GPU capacity."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from threading import Lock

_cpu_sample = None
_cpu_lock = Lock()


def cpu_capacity() -> int:
    count = max(1, os.cpu_count() or 1)
    if hasattr(os, 'sched_getaffinity'):
        count = min(count, len(os.sched_getaffinity(0)))
    quota_file = Path('/sys/fs/cgroup/cpu.max')
    if quota_file.exists():
        quota, period = quota_file.read_text().split()
        if quota != 'max':
            count = min(count, max(1, int(int(quota) / int(period))))
    return max(1, count)


def cpu_usage(count):
    global _cpu_sample
    path = Path('/sys/fs/cgroup/cpu.stat')
    if path.exists():
        values = dict(line.split() for line in path.read_text().splitlines())
        if 'usage_usec' in values:
            current = time.monotonic(), int(values['usage_usec'])
            with _cpu_lock:
                previous, _cpu_sample = _cpu_sample, current
            if previous and current[0] - previous[0] > 0.01:
                return round(min(100, max(0, (current[1] - previous[1]) / 1e6 /
                                                       (current[0] - previous[0]) / count * 100)), 1)
    try:
        return round(min(100, os.getloadavg()[0] / count * 100), 1)
    except (AttributeError, OSError):
        return None


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_count: int
    cpu_load_percent: float | None
    ram_available_gib: float | None
    gpu_name: str | None
    gpu_total_gib: float | None
    gpu_free_gib: float | None
    gpu_utilization_percent: int | None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TuningPlan:
    profile: str
    yolo_batch_size: int
    pose_batch_size: int
    batch_prefetch: int
    frame_prefetch: int
    tracking_workers: int
    threads_per_tracking_worker: int
    reason: str

    def to_dict(self):
        return asdict(self)


def _memory_available_gib() -> float | None:
    path = Path('/proc/meminfo')
    if not path.is_file():
        return None
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.strip().split()[0])
    return round(values.get('MemAvailable', 0) / 1024 / 1024, 2)


def capture_resources() -> ResourceSnapshot:
    cpu_count = cpu_capacity()
    cpu_load = cpu_usage(cpu_count)
    gpu_name = None
    total = free = utilization = None
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits',
        ], check=True, capture_output=True, text=True, timeout=8)
        gpu_name, total_mib, free_mib, utilization_value = [part.strip() for part in result.stdout.splitlines()[0].split(',', 3)]
        total = round(int(total_mib) / 1024, 2)
        free = round(int(free_mib) / 1024, 2)
        utilization = int(utilization_value)
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        pass
    return ResourceSnapshot(cpu_count, cpu_load, _memory_available_gib(), gpu_name, total, free, utilization)


def choose_plan(snapshot: ResourceSnapshot, profile: str = 'auto') -> TuningPlan:
    if profile == 'compat':
        return TuningPlan('compat', 8, 4, 1, 4, 1, max(1, min(4, snapshot.cpu_count)),
                          'Original slow settings for compatibility comparison.')
    if profile != 'auto':
        raise ValueError('profile must be auto or compat')
    available = snapshot.gpu_free_gib if snapshot.gpu_free_gib is not None else snapshot.gpu_total_gib
    memory = min(snapshot.gpu_total_gib or 0, available or 0)
    if memory < 10:
        yolo, pose, prefetch, workers = 8, 4, 1, 1
    elif memory < 18:
        yolo, pose, prefetch, workers = 24, 12, 2, 2
    elif memory < 28:
        yolo, pose, prefetch, workers = 64, 32, 2, 3
    else:
        yolo, pose, prefetch, workers = 96, 48, 3, 4
    workers = min(workers, max(1, snapshot.cpu_count // 4))
    if snapshot.ram_available_gib is not None and snapshot.ram_available_gib < 16:
        workers = 1
        yolo = min(yolo, 16)
        pose = min(pose, 8)
    threads = max(1, min(4, snapshot.cpu_count // workers))
    return TuningPlan('auto', yolo, pose, prefetch, prefetch * 8, workers, threads,
                      'Selected from GPU memory, available RAM and CPU count; live usage is recorded during the run.')


def concurrency_limit(snapshot: ResourceSnapshot, maximum: int) -> int:
    """Admission control for new video shards; running tracks are never interrupted."""
    maximum = max(1, maximum)
    if snapshot.cpu_load_percent is not None and snapshot.cpu_load_percent >= 90:
        return 1
    if snapshot.ram_available_gib is not None and snapshot.ram_available_gib < 8:
        return 1
    if snapshot.gpu_free_gib is not None and snapshot.gpu_free_gib < 1.5:
        return 1
    if snapshot.gpu_utilization_percent is not None and snapshot.gpu_utilization_percent >= 95:
        return max(1, maximum - 1)
    return maximum


class ResourceMonitor:
    def __init__(self, output: Path, interval: float = 5.0):
        from threading import Event, Thread
        self.output = output
        self.interval = interval
        self.stop = Event()
        self.thread = Thread(target=self._run, name='resource-monitor', daemon=True)

    def _run(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open('a', encoding='utf-8', buffering=1) as stream:
            while not self.stop.is_set():
                stream.write(json.dumps({'time_unix': time.time(), **capture_resources().to_dict()}) + '\n')
                self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)
