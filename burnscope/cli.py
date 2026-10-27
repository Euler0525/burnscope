import argparse
import json
import sys
import time

from . import __version__
from .model import Plan
from .processes import Child, probe_gpus
from .reports import export
from .storage import Store


def follow(child, duration=None):
    started, finished = time.monotonic(), False
    try:
        while True:
            child.ping()
            for packet in child.drain():
                print(json.dumps(packet, ensure_ascii=False, allow_nan=False), flush=True)
                if packet["type"] == "finished":
                    finished = True
                    return packet["report"]["exit_code"]
                if packet["type"] == "error":
                    return 3
            if duration is not None and time.monotonic() - started >= duration:
                return 0
            if not child.alive():
                # Drain once more after process exit: a finished packet may have raced the prior poll.
                for packet in child.drain():
                    print(json.dumps(packet, ensure_ascii=False, allow_nan=False), flush=True)
                    if packet["type"] == "finished":
                        finished = True
                        return packet["report"]["exit_code"]
                return 3
            time.sleep(.1)
    except KeyboardInterrupt:
        child.stop.set()
        deadline = time.monotonic() + 2
        while child.alive() and time.monotonic() < deadline:
            child.ping()
            for packet in child.drain():
                print(json.dumps(packet, ensure_ascii=False, allow_nan=False), flush=True)
                if packet["type"] == "finished":
                    finished = True
            time.sleep(.05)
        return 130
    finally:
        child.close()
        child.pipe.close()
        if duration is None and not finished:
            print(json.dumps({"type": "event", "message": "测试未正常结束，可在记录中查看已保存数据"}, ensure_ascii=False))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="BurnScope Python · 简洁烤机")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    gui = commands.add_parser("gui")
    gui.add_argument("--demo", action="store_true")
    gui.add_argument("--preview")
    commands.add_parser("devices")
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=["cpu", "gpu", "combined", "ram", "vram"], default="cpu")
    run.add_argument("--duration", type=float, default=300, help="秒")
    run.add_argument("--load", type=float, default=100, help="百分比")
    run.add_argument("--threads", type=int, default=0, help="0 为全部可用逻辑核")
    run.add_argument("--gpu", type=int, default=0)
    run.add_argument("--memory-mib", type=int, default=0)
    monitor = commands.add_parser("monitor")
    monitor.add_argument("--duration", type=float, default=60)
    monitor.add_argument("--gpu", type=int, default=0)
    commands.add_parser("history")
    report = commands.add_parser("report")
    report.add_argument("id")
    report.add_argument("--output", default="report.html")
    args = parser.parse_args()
    try:
        if args.command in {None, "gui"}:
            if getattr(args, "preview", None) and not args.demo:
                parser.error("截图预览只用于 --demo，不会启动测试")
            from .gui import main as gui_main
            return gui_main(getattr(args, "demo", False), getattr(args, "preview", None))
        if args.command == "devices":
            print(json.dumps(probe_gpus(), ensure_ascii=False, indent=2))
        elif args.command in {"run", "monitor"}:
            gpu = None
            if args.command == "monitor" or args.mode in {"gpu", "combined", "vram"}:
                found = probe_gpus()
                if found["devices"]:
                    gpu = next((g for g in found["devices"] if g["index"] == args.gpu), None)
                    if gpu is None:
                        raise ValueError("GPU 序号无效，先运行 devices 查看")
            if args.command == "monitor":
                return follow(Child("monitor", {"gpu": gpu}), args.duration)
            plan = Plan(args.mode, args.duration, args.load / 100, args.threads, gpu, args.memory_mib).validate()
            return follow(Child("controller", plan.plain()))
        elif args.command in {"history", "report"}:
            store = Store()
            try:
                store.recover()
                if args.command == "history":
                    print(json.dumps(store.history(), ensure_ascii=False, indent=2))
                else:
                    print(export(store, args.id, args.output).resolve())
            finally:
                store.close()
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False))
        return 4
