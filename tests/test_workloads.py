from array import array
import math

import pytest

from burnscope.model import Plan
from burnscope.workloads import (check_compute, float_block, integer_block, integer_reference,
                                pattern, ram_budget, verify_memory)


@pytest.mark.parametrize("seed,rounds", [(0, 0), (0, 1), (0x12345678, 2048), (12345, 4096)])
def test_integer_independent_reference(seed, rounds):
    assert integer_block(seed, rounds) == integer_reference(seed, rounds)


def test_compute_error_injection():
    check_compute(integer_block(), float_block())
    with pytest.raises(RuntimeError, match="校验失败"):
        check_compute(integer_block() ^ 1, float_block())
    with pytest.raises(RuntimeError, match="校验失败"):
        check_compute(integer_block(), math.nan)


@pytest.mark.parametrize("pass_", [0, 1, 2, 3, 4, 35, 36, 39])
def test_ram_patterns_and_corruption(pass_):
    expected = pattern(pass_)
    buffer = array("I", [expected]) * 1024
    verify_memory(buffer, expected)
    buffer[17] ^= 1
    with pytest.raises(RuntimeError, match="字节 68"):
        verify_memory(buffer, expected)


def test_memory_budget_reserves_2_gib(monkeypatch):
    class Memory:
        available = 3 * 1024**3
    monkeypatch.setattr("burnscope.workloads.psutil.virtual_memory", lambda: Memory())
    assert ram_budget() == 1024
    with pytest.raises(ValueError):
        ram_budget(1025)


def test_random_memory_corruption():
    import random
    expected = array("I")
    expected.frombytes(random.Random(37).randbytes(4096))
    buffer = expected[:]
    verify_memory(buffer, expected)
    buffer[100] ^= 1
    with pytest.raises(RuntimeError, match="字节 400"):
        verify_memory(buffer, expected)


def test_gpu_requires_real_device():
    with pytest.raises(ValueError, match="没有可用 GPU"):
        Plan(mode="gpu").validate()
    assert Plan().validate().mode == "cpu"
