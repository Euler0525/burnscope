"""One portable spawn/heartbeat path for every supported desktop system."""
import multiprocessing as mp
import os
import threading
import time


CTX = mp.get_context("spawn")


def dispatch(kind, config, stop, heartbeat, output):
    finished = threading.Event()

    def watchdog():
        while not finished.wait(.2):
            parent = mp.parent_process()
            if time.monotonic() - heartbeat.value > 3 or (parent and not parent.is_alive()):
                stop.set()
                if not finished.wait(1.5):
                    os._exit(130)
                return

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        if kind == "controller":
            from .engine import run_engine
            run_engine(config, stop, output.send)
        elif kind == "monitor":
            from .monitor import run_monitor
            run_monitor(config, stop, output.send)
        elif kind == "probe":
            from .gpu import discover
            output.send({"type": "devices", **discover()})
        elif kind == "gpu":
            from .gpu import run_gpu
            run_gpu(config, stop, output.send)
        else:
            from .workloads import run_cpu
            run_cpu(config, stop, output.send)
    except Exception as exc:
        try:
            output.send({"type": "error", "message": str(exc)})
        except (BrokenPipeError, OSError):
            pass
    finally:
        finished.set()
        output.close()


class Child:
    def __init__(self, kind, config):
        self.stop = CTX.Event()
        self.heartbeat = CTX.Value("d", time.monotonic())
        self.pipe, writer = CTX.Pipe(duplex=False)
        self.process = CTX.Process(target=dispatch, args=(kind, config, self.stop, self.heartbeat, writer))
        self.process.start()
        writer.close()
        self.closed = False

    def ping(self):
        self.heartbeat.value = time.monotonic()

    def drain(self):
        messages = []
        try:
            while self.pipe.poll():
                messages.append(self.pipe.recv())
        except (EOFError, OSError):
            pass
        return messages

    def alive(self):
        return self.process.is_alive()

    def close(self):
        if not self.closed:
            stop_children([self])


def stop_children(children):
    children = [c for c in children if not c.closed]
    for child in children:
        child.stop.set()
    deadline = time.monotonic() + .8
    for child in children:
        child.process.join(max(0, deadline - time.monotonic()))
    for child in children:
        if child.alive():
            child.process.terminate()
    deadline = time.monotonic() + .5
    for child in children:
        child.process.join(max(0, deadline - time.monotonic()))
        if child.alive():
            child.process.kill()
    deadline = time.monotonic() + .3
    for child in children:
        child.process.join(max(0, deadline - time.monotonic()))
        child.closed = True


def probe_gpus(timeout=15):
    child = Child("probe", {})
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            child.ping()
            for packet in child.drain():
                if packet["type"] == "devices":
                    return packet
                if packet["type"] == "error":
                    return {"devices": [], "issues": [packet["message"]]}
            if not child.alive():
                break
            time.sleep(.05)
        return {"devices": [], "issues": ["GPU 检测未完成；CPU 和内存测试仍可使用"]}
    finally:
        child.close()
