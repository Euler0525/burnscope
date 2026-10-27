import time

from .model import Plan, now
from .processes import Child, stop_children
from .storage import Store, new_report
from .workloads import available_cpus, ram_budget


class Halt(Exception):
    def __init__(self, message, state="failed", code=3):
        self.message, self.state, self.code = message, state, code


class GpuGuard:
    def __init__(self, required=False):
        self.required, self.last, self.stamp, self.over = required, time.monotonic(), None, 0

    def inspect(self, samples, clock=None):
        clock = time.monotonic() if clock is None else clock
        critical = next((s["value"] for s in samples if s["device"] == "gpu" and
                         s["metric"] == "critical_temperature" and s["status"] == "ok"), None)
        limit = min(85, critical - 5) if critical is not None else 85
        for sample in samples:
            if sample["device"] != "gpu" or sample["metric"] != "temperature" or sample["status"] != "ok":
                continue
            if sample["value"] is None or not -1 <= clock - sample["monotonic"] < 5:
                continue
            self.required = True
            self.last = sample["monotonic"]
            if self.stamp != self.last:
                self.stamp = self.last
                self.over = self.over + 1 if sample["value"] >= limit else 0
            if self.over >= 2 or (critical is not None and sample["value"] >= critical):
                raise Halt("GPU 温度达到停止阈值", "protected_stop", 2)
        if self.required and clock - self.last >= 5:
            raise Halt("GPU 温度连续 5 秒不可读，已停止测试", "protected_stop", 2)


def run_engine(config, stop, emit):
    plan = Plan(**config).validate()
    store = Store()
    store.recover()
    report = new_report(plan)
    store.save(report)
    started, active_start, last_collector = time.monotonic(), None, time.monotonic()
    collector, workers, latest, stats, ready = None, [], {}, {}, set()
    gpu_active = plan.mode in {"gpu", "combined", "vram"}
    guard = GpuGuard()
    thermal_flags = set()

    def record(packet):
        packet["elapsed"] = time.monotonic() - started
        store.append(report["id"], packet["elapsed"], packet)
        emit(packet)

    def note(message):
        record({"type": "event", "message": message})

    def pump():
        nonlocal last_collector
        if stop.is_set():
            raise Halt("测试已停止，记录已保存", "cancelled", 130)
        collector.ping()
        for packet in collector.drain():
            if packet["type"] == "error":
                raise Halt("监控组件异常：" + packet["message"], "protected_stop", 2)
            if packet["type"] == "samples":
                last_collector = time.monotonic()
                for sample in packet["samples"]:
                    latest[sample["device"], sample["metric"]] = sample
                    if sample["metric"] == "throttle_mask" and sample["status"] == "ok":
                        thermal_flags.add(int(sample["value"]))
                if gpu_active:
                    guard.inspect(packet["samples"])
                record(packet)
        if not collector.alive() or time.monotonic() - last_collector >= 5:
            raise Halt("监控组件退出或连续 5 秒无响应，已停止测试", "protected_stop", 2)
        if gpu_active:
            guard.inspect([])
        for name, worker in workers:
            worker.ping()
            for packet in worker.drain():
                if packet["type"] == "error":
                    raise Halt(name + "：" + packet["message"])
                if packet["type"] == "ready":
                    ready.add(name)
                if packet["type"] in {"ready", "stats", "finished"}:
                    stats.setdefault(name, {}).update(packet)
            if not worker.alive():
                raise Halt(name + " 压力进程意外退出")

    try:
        emit({"type": "started", "id": report["id"]})
        if plan.mode in {"cpu", "combined", "ram"}:
            note("CPU 温度功能已移除；本次不提供 CPU 温度保护或散热诊断。")
        collector = Child("monitor", {"gpu": plan.gpu})
        deadline = time.monotonic() + 15
        while not latest and time.monotonic() < deadline:
            pump()
            stop.wait(.05)
        if not latest:
            raise Halt("无法取得系统监控数据", "unavailable", 4)
        duration = plan.duration
        if gpu_active and not guard.required:
            duration = min(duration, 60)
            note("GPU 温度不可用，本次测试最多运行 60 秒；其他 GPU 指标按驱动能力显示。")
        count = plan.threads or available_cpus()
        if count > available_cpus():
            raise Halt("线程数超过当前可用逻辑核数", "unavailable", 4)
        budget = ram_budget(plan.memory_mib) if plan.mode == "ram" else 0
        if plan.mode == "ram":
            count = min(count, budget)
        active_start = time.monotonic()
        if plan.mode in {"cpu", "combined", "ram"}:
            for i in range(count):
                pump()
                memory = budget // count + (i < budget % count)
                workers.append((f"cpu:{i}", Child("cpu", {
                    "mode": "ram" if plan.mode == "ram" else "cpu", "load": plan.load, "memory_mib": memory})))
        if gpu_active:
            total = plan.gpu["memory_bytes"]
            free = latest.get(("gpu", "memory_free"), {}).get("value")
            safe = int(min(total * .25, 256 * 1048576) if free is None else free * .6) // 1048576
            if plan.gpu["shared_memory"]:
                safe = min(safe, ram_budget())
            memory = plan.memory_mib or safe
            if memory < 1 or memory > safe:
                raise Halt("显存预算超过当前安全余量或显存不足", "unavailable", 4)
            workers.append(("gpu", Child("gpu", {"gpu": plan.gpu, "mode": plan.mode,
                "memory_mib": memory, "load": plan.load})))
        deadline = time.monotonic() + 45
        while len(ready) < len(workers):
            pump()
            if time.monotonic() >= deadline:
                raise Halt("压力进程未能及时初始化", "unavailable", 4)
            if gpu_active and not guard.required and time.monotonic() - active_start >= 60:
                raise Halt("GPU 无温度监测运行时限已到", "protected_stop", 2)
            stop.wait(.05)
        test_start = time.monotonic()
        test_end = test_start + duration
        if gpu_active and not guard.required:
            test_end = min(test_end, active_start + 60)
        note("开始测试")
        previous_blocks, previous_time, next_stats = 0, test_start, test_start
        while time.monotonic() < test_end:
            pump()
            clock = time.monotonic()
            if clock >= next_stats:
                blocks = sum(s.get("blocks", 0) for s in stats.values())
                cpu_blocks = sum(s.get("blocks", 0) for name, s in stats.items() if name.startswith("cpu:"))
                gpu_blocks = stats.get("gpu", {}).get("blocks", 0)
                # Keep CPU and GPU units separate; the sum is never a performance score.
                emit({"type": "progress", "seconds": clock - test_start, "duration": test_end - test_start,
                      "cpu_blocks": cpu_blocks, "gpu_operations": gpu_blocks,
                      "updates": blocks - previous_blocks, "interval": clock - previous_time})
                previous_blocks, previous_time, next_stats = blocks, clock, clock + 1
            stop.wait(.05)
        report.update(state="completed", exit_code=0)
        report["test_seconds"] = time.monotonic() - test_start
    except Halt as exc:
        report.update(state=exc.state, exit_code=exc.code)
        note(exc.message)
    except Exception as exc:
        report.update(state="failed", exit_code=3)
        note(str(exc))
    finally:
        stop_children([w for _, w in workers])
        for name, worker in workers:
            for packet in worker.drain():
                if packet["type"] == "error":
                    report.update(state="failed", exit_code=3)
                    note(name + "：" + packet["message"])
                elif packet["type"] in {"stats", "finished"}:
                    stats.setdefault(name, {}).update(packet)
            if report["state"] == "completed" and (not stats.get(name, {}).get("blocks") or
                    worker.process.exitcode != 0):
                report.update(state="failed", exit_code=3)
                note(name + " 未完成有效校验或结束异常")
            worker.pipe.close()
        if collector:
            collector.close()
            collector.pipe.close()
        report.update(finished=now(), elapsed_seconds=time.monotonic() - started, workers=stats)
        if plan.mode in {"cpu", "combined", "ram"}:
            report["findings"].append("CPU 温度监测已移除，本报告无法判断 CPU 散热或硅脂状态。")
        if gpu_active and not guard.required:
            report["findings"].append("GPU 温度不可用，本次已限制为最多 60 秒，无法判断 GPU 散热状态。")
        if any(flags & (0x20 | 0x40) for flags in thermal_flags):
            report["findings"].append("显卡驱动报告了热限制标记；原始标记已保存在采样记录中。")
        if any(flags & 0x4 for flags in thermal_flags):
            report["findings"].append("显卡驱动报告过功耗限制。")
        if report["state"] == "completed":
            report["findings"].append("测试已完成，在本次已完成的计算与内存校验中未发现错误。")
        elif not report["findings"]:
            report["findings"].append("测试未完整完成，请查看事件记录。")
        store.save(report)
        emit({"type": "finished", "report": report})
        store.close()
