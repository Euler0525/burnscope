from array import array
import math
import random
import time

import psutil


def available_cpus():
    process = psutil.Process()
    try:
        return len(process.cpu_affinity())
    except (AttributeError, psutil.Error, OSError):
        return psutil.cpu_count() or 1


def ram_budget(requested=0):
    free = psutil.virtual_memory().available
    safe = max(0, min(free // 2, free - 2 * 1024**3)) // 1048576
    budget = requested or min(1024, safe)
    if budget < 1 or budget > safe:
        raise ValueError("可用内存不足，或预算超过保留 2 GiB 后的安全余量")
    return int(budget)


def integer_block(seed=0x12345678, rounds=2048):
    value = seed
    for _ in range(rounds):
        value = (1664525 * value + 1013904223) & 0xffffffff
    return value


def integer_reference(seed=0x12345678, rounds=2048):
    # Closed form in a larger modulus, independent of the loop under test.
    a, c, m = 1664525, 1013904223, 1 << 32
    power = pow(a, rounds, m * (a - 1))
    return (pow(a, rounds, m) * seed + c * ((power - 1) // (a - 1))) % m


def float_block():
    value = .25
    for _ in range(1024):
        value = value * 1.0001 + .0001
    return value


def check_compute(integer, floating):
    expected = 1.25 * 1.0001**1024 - 1
    if integer != integer_reference() or not math.isclose(floating, expected, rel_tol=1e-10):
        raise RuntimeError("CPU 计算校验失败")


def pattern(pass_):
    if pass_ < 4:
        return (0, 0xffffffff, 0xaaaaaaaa, 0x55555555)[pass_]
    if pass_ < 36:
        return 1 << (pass_ - 4)
    return random.Random(pass_).getrandbits(32)


def verify_memory(buffer, expected, offset=0):
    if isinstance(expected, array):
        if buffer == expected:
            return
        for index, (value, reference) in enumerate(zip(buffer, expected)):
            if value != reference:
                raise RuntimeError(f"内存校验失败：字节 {offset + index * buffer.itemsize}，预期 {reference}，实际 {value}")
        raise RuntimeError("内存校验失败：长度不匹配")
    for index, value in enumerate(buffer):
        if value != expected:
            raise RuntimeError(f"内存校验失败：字节 {offset + index * buffer.itemsize}，预期 {expected}，实际 {value}")


def run_cpu(config, stop, emit):
    mode, load = config["mode"], config["load"]
    buffers = []
    allocated = 0
    if mode == "ram":
        remaining = config["memory_mib"] * 1048576
        while remaining and not stop.is_set():
            size = min(256 * 1024, remaining)
            block = array("I", [0]) * (size // array("I").itemsize)
            buffers.append(block)
            allocated += len(block) * block.itemsize
            remaining -= size
    emit({"type": "ready", "allocated_bytes": allocated})
    blocks, checked, pass_, last = 0, 0, 0, 0
    while not stop.is_set():
        start = time.monotonic()
        if mode == "ram":
            expected = pattern(pass_)
            for index, buffer in enumerate(buffers):
                if stop.is_set():
                    break
                if pass_ >= 36:
                    reference = array("I")
                    reference.frombytes(random.Random((pass_ << 32) | index).randbytes(len(buffer) * buffer.itemsize))
                else:
                    reference = array("I", [expected]) * len(buffer)
                buffer[:] = reference
                verify_memory(buffer, reference, index * 256 * 1024)
                checked += len(buffer) * buffer.itemsize
                blocks += 1
            pass_ = (pass_ + 1) % 40
        else:
            check_compute(integer_block(), float_block())
            blocks += 1
        if time.monotonic() - last >= .5:
            emit({"type": "stats", "blocks": blocks, "checked_bytes": checked,
                  "allocated_bytes": allocated, "errors": 0})
            last = time.monotonic()
        if load < 1:
            stop.wait((time.monotonic() - start) * (1 / load - 1))
    emit({"type": "finished", "blocks": blocks, "checked_bytes": checked,
          "allocated_bytes": allocated, "errors": 0})
