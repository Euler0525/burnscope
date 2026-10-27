import pytest

from burnscope.engine import GpuGuard, Halt
from burnscope.monitor import Monitor, metric


def sample(value=50, stamp=0, device="gpu", name="temperature"):
    result = metric(device, name, value, "℃", "fixture")
    result["monotonic"] = stamp
    return result


def test_cpu_has_no_temperature_dependency():
    guard = GpuGuard()
    guard.last = 0
    guard.inspect([sample(120, 10, "cpu")], clock=10)
    guard.inspect([], clock=100)
    assert not guard.required


def test_duplicate_sample_does_not_trigger_twice():
    guard = GpuGuard()
    guard.inspect([sample(85)], clock=0)
    guard.inspect([sample(85)], clock=.5)
    with pytest.raises(Halt) as alert:
        guard.inspect([sample(85, 1)], clock=1)
    assert alert.value.state == "protected_stop"


def test_gpu_temperature_loss_stops():
    guard = GpuGuard()
    guard.inspect([sample()], clock=0)
    with pytest.raises(Halt) as alert:
        guard.inspect([sample(None, 5)], clock=5)
    assert "不可读" in alert.value.message


def test_gpu_critical_minus_five():
    guard = GpuGuard()
    guard.inspect([sample(76), sample(80, name="critical_temperature")], clock=0)
    with pytest.raises(Halt):
        guard.inspect([sample(76, 1), sample(80, 1, name="critical_temperature")], clock=1)


def test_monitor_does_not_read_cpu_sensors(monkeypatch):
    import psutil
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU temperature API must never be called")
    monkeypatch.setattr(psutil, "sensors_temperatures", forbidden, raising=False)
    monkeypatch.setattr(psutil, "cpu_percent", lambda **kwargs: [12, 34])
    monitor = Monitor()
    readings = monitor.collect() + monitor.collect()
    assert not any(s["metric"] == "temperature" for s in readings)
    assert any(s["metric"] == "utilization" and s["value"] == 23 for s in readings)
