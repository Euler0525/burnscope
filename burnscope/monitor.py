"""Portable psutil + optional NVML; deliberately no CPU temperature API."""
import math
import time

import psutil


def metric(device, name, value, unit, source, reason=""):
    valid = value is not None and math.isfinite(value)
    return {"device": device, "metric": name, "value": value if valid else None,
            "unit": unit, "source": source, "status": "ok" if valid else "unavailable",
            "reason": reason, "monotonic": time.monotonic()}


class Monitor:
    def __init__(self, gpu=None):
        self.gpu, self.nv, self.handle = gpu, None, None
        self.cpu_ready = False
        if gpu:
            try:
                import pynvml as nv
                nv.nvmlInit()
                self.nv = nv
                if gpu.get("pci"):
                    self.handle = nv.nvmlDeviceGetHandleByPciBusId(gpu["pci"])
                elif gpu.get("uuid"):
                    raw = gpu["uuid"]
                    uuid = "GPU-" + "-".join([raw[:8], raw[8:12], raw[12:16], raw[16:20], raw[20:]])
                    self.handle = nv.nvmlDeviceGetHandleByUUID(uuid)
            except Exception:
                self.handle = None

    def collect(self):
        readings = []
        try:
            usage = psutil.cpu_percent(percpu=True)
            if self.cpu_ready and usage:
                readings.append(metric("cpu", "utilization", sum(usage) / len(usage), "%", "psutil"))
                readings.extend(metric(f"cpu:{i}", "utilization", value, "%", "psutil")
                                for i, value in enumerate(usage))
            else:
                readings.append(metric("cpu", "utilization", None, "%", "psutil", "正在初始化采样"))
            self.cpu_ready = True
        except (OSError, RuntimeError):
            readings.append(metric("cpu", "utilization", None, "%", "psutil", "系统未提供利用率计数器"))
        # psutil can return a fixed or nominal clock: never label it effective clock.
        try:
            frequency = psutil.cpu_freq()
            clock = frequency.current if frequency and frequency.current > 0 else None
        except (AttributeError, OSError, RuntimeError, NotImplementedError):
            clock = None
        readings.append(metric("cpu", "reported_clock", clock, "MHz", "psutil", "系统报告频率，可能是标称值"))
        memory = psutil.virtual_memory()
        readings += [metric("system", "memory_used", memory.used, "B", "psutil"),
                     metric("system", "memory_percent", memory.percent, "%", "psutil"),
                     metric("system", "memory_available", memory.available, "B", "psutil")]
        if self.gpu:
            nv, handle = self.nv, self.handle
            queries = {}
            if handle is not None:
                queries = {
                    "temperature": ("℃", lambda: nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)),
                    "critical_temperature": ("℃", lambda: nv.nvmlDeviceGetTemperatureThreshold(
                        handle, nv.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)),
                    "utilization": ("%", lambda: nv.nvmlDeviceGetUtilizationRates(handle).gpu),
                    "clock": ("MHz", lambda: nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_GRAPHICS)),
                    "power": ("W", lambda: nv.nvmlDeviceGetPowerUsage(handle) / 1000),
                    "memory_free": ("B", lambda: nv.nvmlDeviceGetMemoryInfo(handle).free),
                    "memory_used": ("B", lambda: nv.nvmlDeviceGetMemoryInfo(handle).used),
                    "throttle_mask": ("bits", lambda: nv.nvmlDeviceGetCurrentClocksEventReasons(handle)),
                }
            for name, unit in [("temperature", "℃"), ("critical_temperature", "℃"), ("utilization", "%"),
                               ("clock", "MHz"), ("power", "W"), ("memory_used", "B"),
                               ("memory_free", "B"), ("throttle_mask", "bits")]:
                value = None
                if name in queries:
                    try:
                        value = queries[name][1]()
                    except Exception:
                        pass
                readings.append(metric("gpu", name, value, unit, "NVML" if handle is not None else "capabilities",
                    "" if value is not None else "通用接口未提供此指标，或不能可靠关联显卡"))
        return readings

    def close(self):
        if self.nv:
            self.nv.nvmlShutdown()


def run_monitor(config, stop, emit):
    monitor = Monitor(config.get("gpu"))
    try:
        while not stop.is_set():
            started = time.monotonic()
            emit({"type": "samples", "samples": monitor.collect()})
            stop.wait(max(0, 1 - (time.monotonic() - started)))
    finally:
        monitor.close()
