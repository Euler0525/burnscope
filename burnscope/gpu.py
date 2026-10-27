"""Optional OpenCL workload. No OS DLL calls, platform tools or name-based NVML matching."""
from pathlib import Path
import os
import time

from .model import data_dir


def signature(device):
    return [device.platform.vendor.strip(), device.platform.name.strip(), device.vendor.strip(),
            device.name.strip(), device.driver_version]


def identity(device):
    result = {}
    try:
        info = device.pci_bus_info_khr
        result["pci"] = f"{info.pci_domain:04x}:{info.pci_bus:02x}:{info.pci_device:02x}.{info.pci_function}"
    except Exception:
        pass
    if "pci" not in result:
        try:
            bus, slot, domain = device.pci_bus_id_nv, device.pci_slot_id_nv, device.pci_domain_id_nv
            result["pci"] = f"{domain:04x}:{bus:02x}:{slot >> 3:02x}.{slot & 7}"
        except Exception:
            pass
    try:
        result["uuid"] = bytes(device.uuid_khr).hex()
    except Exception:
        pass
    return result


def discover():
    devices, issues = [], []
    try:
        import pyopencl as cl
        for pi, platform in enumerate(cl.get_platforms()):
            for di, device in enumerate(platform.get_devices(device_type=cl.device_type.GPU)):
                devices.append({"index": len(devices), "platform": pi, "device": di,
                    "name": device.name.strip(), "vendor": device.vendor.strip(),
                    "signature": signature(device), "memory_bytes": int(device.global_mem_size),
                    "max_allocation": int(device.max_mem_alloc_size),
                    "shared_memory": bool(device.host_unified_memory), **identity(device)})
        if not devices:
            issues.append("驱动没有提供 OpenCL GPU；CPU 和内存测试仍可使用")
    except ImportError:
        issues.append("GPU 功能未安装，按 requirements-gpu.txt 安装后可启用")
    except Exception as exc:
        issues.append(f"GPU 不可用：{exc}")
    return {"devices": devices, "issues": issues}


def memory_pattern(index, pass_):
    if pass_ < 4:
        return (0, 0xffffffff, 0xaaaaaaaa, 0x55555555)[pass_]
    if pass_ < 36:
        return 1 << (pass_ - 4)
    x = (index ^ (pass_ * 0x9e3779b9)) & 0xffffffff
    x = ((x ^ (x >> 16)) * 0x7feb352d) & 0xffffffff
    x = ((x ^ (x >> 15)) * 0x846ca68b) & 0xffffffff
    return (x ^ (x >> 16)) & 0xffffffff


def float_reference(indices, iterations):
    import numpy as np
    seed = np.float32(.125) + (np.asarray(indices) & 15).astype(np.float32) * np.float32(.03125)
    values = seed[:, None] + np.arange(8, dtype=np.float32)[None, :] * np.float32(.125)
    a, b = np.float64(np.float32(1.0001)), np.float64(np.float32(.0001))
    for _ in range(iterations):
        values = (values.astype(np.float64) * a + b).astype(np.float32)
    return values


def run_gpu(config, stop, emit):
    # The worker needs no persistent Python wrapper cache or system cache directory.
    os.environ["PYOPENCL_NO_CACHE"] = "1"
    import numpy as np
    import pyopencl as cl
    selected = config["gpu"]
    platform = cl.get_platforms()[selected["platform"]]
    device = platform.get_devices(device_type=cl.device_type.GPU)[selected["device"]]
    if signature(device) != selected["signature"] or identity(device) != {
            key: selected[key] for key in ("pci", "uuid") if key in selected}:
        raise RuntimeError("显卡列表已变化，请重新打开程序检测")
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    cache = data_dir() / "opencl-cache"
    cache.mkdir(exist_ok=True)
    program = cl.Program(context, (Path(__file__).parent / "kernels/stress.cl").read_text()).build(
        ["-cl-std=CL1.2"], cache_dir=str(cache))
    compute, fill, verify = (cl.Kernel(program, name) for name in ("compute", "fill", "verify"))
    count = max(256, min(65536, device.max_compute_units * 256))
    output = cl.Buffer(context, cl.mem_flags.READ_WRITE, count * 32)
    allocated = count * 32
    buffers = []
    if config["mode"] == "vram":
        remaining = config["memory_mib"] * 1048576
        chunk = min(8 * 1048576, int(device.max_mem_alloc_size) // 2)
        while remaining and not stop.is_set():
            size = min(chunk, remaining)
            buffers.append((cl.Buffer(context, cl.mem_flags.READ_WRITE, size), size // 4))
            allocated += size
            remaining -= size
    emit({"type": "ready", "allocated_bytes": allocated})
    blocks, checked, iterations, pass_, last, last_check, maximum_ms = 0, 0, 32, 0, 0, 0, 0

    def wait(event, profile=True):
        nonlocal maximum_ms
        while event.command_execution_status != cl.command_execution_status.COMPLETE:
            if stop.is_set():
                raise InterruptedError
            if event.command_execution_status < 0:
                raise RuntimeError("GPU 任务失败或显卡驱动重置")
            time.sleep(.001)
        ms = (event.profile.end - event.profile.start) / 1e6 if profile else 0
        maximum_ms = max(maximum_ms, ms)
        return ms

    try:
        while not stop.is_set():
            start = time.monotonic()
            if config["mode"] == "vram":
                for buffer, words in buffers:
                    if stop.is_set():
                        break
                    wait(fill(queue, (words,), None, buffer, np.uint32(words), np.uint32(pass_)))
                    counters = np.asarray([0, 0xffffffff], np.uint32)
                    errbuf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                       hostbuf=counters)
                    wait(verify(queue, (words,), None, buffer, np.uint32(words), np.uint32(pass_), errbuf))
                    wait(cl.enqueue_copy(queue, counters, errbuf), False)
                    errbuf.release()
                    actual = np.empty(min(64, words), np.uint32)
                    wait(cl.enqueue_copy(queue, actual, buffer), False)
                    expected = np.asarray([memory_pattern(i, pass_) for i in range(len(actual))], np.uint32)
                    if counters[0] or np.any(actual != expected):
                        raise RuntimeError(f"显存校验失败，首个设备错误字 {int(counters[1])}，错误数 {int(counters[0])}")
                    checked += words * 4
                    blocks += 1
                pass_ = (pass_ + 1) % 40
            else:
                ms = wait(compute(queue, (count,), None, output, np.uint32(iterations)))
                if time.monotonic() - last_check >= 1:
                    actual = np.empty((count, 8), np.float32)
                    wait(cl.enqueue_copy(queue, actual, output), False)
                    indices = np.unique(np.linspace(0, count - 1, 32, dtype=int))
                    if not np.allclose(actual[indices], float_reference(indices, iterations), rtol=1e-5, atol=1e-6):
                        raise RuntimeError("GPU 计算校验失败")
                    last_check = time.monotonic()
                blocks += count * iterations * 8
                if ms:
                    iterations = max(1, min(2048, int(iterations * min(2, 40 / ms))))
            if time.monotonic() - last >= .5:
                emit({"type": "stats", "blocks": blocks, "checked_bytes": checked, "errors": 0,
                      "allocated_bytes": allocated, "max_kernel_ms": maximum_ms})
                last = time.monotonic()
            if config["load"] < 1:
                stop.wait((time.monotonic() - start) * (1 / config["load"] - 1))
    except InterruptedError:
        pass
    emit({"type": "finished", "blocks": blocks, "checked_bytes": checked, "errors": 0,
          "allocated_bytes": allocated, "max_kernel_ms": maximum_ms})
