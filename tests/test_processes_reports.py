import time

import pytest

from burnscope.model import Plan
from burnscope.processes import Child, stop_children
from burnscope.reports import export
from burnscope.storage import Store, new_report


def test_python_worker_runs_checks_and_stops():
    child = Child("cpu", {"mode": "cpu", "load": .01, "memory_mib": 0})
    packets = []
    deadline = time.monotonic() + 8
    try:
        while time.monotonic() < deadline:
            child.ping()
            packets.extend(child.drain())
            if any(p.get("blocks", 0) for p in packets):
                break
            time.sleep(.05)
        assert any(p.get("blocks", 0) for p in packets)
        assert not any(p["type"] == "error" for p in packets)
        start = time.monotonic()
        child.close()
        assert time.monotonic() - start < 2
        assert not child.alive()
        assert child.process.exitcode == 0
    finally:
        child.close()
        child.pipe.close()


def test_stale_parent_heartbeat_stops_worker():
    child = Child("cpu", {"mode": "cpu", "load": .01, "memory_mib": 0})
    child.heartbeat.value = time.monotonic() - 10
    try:
        child.process.join(7)
        assert not child.alive()
    finally:
        child.close()
        child.pipe.close()


def test_parallel_stop_deadline():
    children = [Child("cpu", {"mode": "cpu", "load": .01, "memory_mib": 0}) for _ in range(3)]
    try:
        start = time.monotonic()
        stop_children(children)
        assert time.monotonic() - start < 2
        assert not any(c.alive() for c in children)
    finally:
        stop_children(children)
        for child in children:
            child.pipe.close()


def test_history_export_and_recovery(tmp_path):
    store = Store(tmp_path / "history.sqlite3")
    report = new_report(Plan())
    report.update(pid=99999999, process_created=0)
    store.save(report)
    store.append(report["id"], 1, {"type": "event", "message": "<script>alert(1)</script>"})
    store.recover()
    assert store.get(report["id"])["state"] == "interrupted"
    output = export(store, report["id"], tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<script>" not in output and "&lt;script&gt;" in output
    assert "CPU 温度" in output
    store.close()


@pytest.mark.parametrize("mode", ["cpu", "ram"])
def test_controller_without_cpu_temperature(mode, tmp_path, monkeypatch):
    monkeypatch.setenv("BURNSCOPE_DATA_DIR", str(tmp_path))
    child = Child("controller", Plan(mode=mode, duration=.3, load=.01, threads=1, memory_mib=1).plain())
    packets = []
    deadline = time.monotonic() + 15
    try:
        while time.monotonic() < deadline:
            child.ping()
            packets.extend(child.drain())
            if any(p["type"] == "finished" for p in packets):
                break
            if not child.alive():
                packets.extend(child.drain())
                break
            time.sleep(.05)
        report = next(p["report"] for p in packets if p["type"] == "finished")
        assert report["state"] == "completed", packets
        assert not report["cpu_temperature_monitoring"]
        assert report["workers"]["cpu:0"]["blocks"] > 0
        samples = [s for p in packets for s in p.get("samples", [])]
        assert not any(s["metric"] == "temperature" and s["device"].startswith("cpu") for s in samples)
    finally:
        child.close()
        child.pipe.close()
