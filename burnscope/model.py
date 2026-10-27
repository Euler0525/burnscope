from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
import os


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def data_dir():
    path = Path(os.environ.get(
        "BURNSCOPE_DATA_DIR", Path.home() / ".burnscope"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Plan:
    mode: str = "cpu"
    duration: float = 300
    load: float = 1.0
    threads: int = 0
    gpu: dict | None = None
    memory_mib: int = 0

    def validate(self):
        if self.mode not in {"cpu", "gpu", "combined", "ram", "vram"}:
            raise ValueError("请选择 CPU、GPU、双烤、内存或显存测试")
        if not math.isfinite(self.duration) or not 0 < self.duration <= 86400:
            raise ValueError("测试时长须在 0～86400 秒之间")
        if not 0 < self.load <= 1 or self.threads < 0 or self.memory_mib < 0:
            raise ValueError("负载、线程数或内存预算不合法")
        if self.mode in {"gpu", "combined", "vram"} and self.gpu is None:
            raise ValueError("没有可用 GPU，请安装 GPU 依赖和显卡的 OpenCL 驱动")
        return self

    def plain(self):
        return asdict(self)
