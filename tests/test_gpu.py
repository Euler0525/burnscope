from pathlib import Path
import os

import pytest

from burnscope.gpu import float_reference, identity, memory_pattern


def test_optional_identity_fields_do_not_disable_gpu():
    class Device:
        def __getattr__(self, key):
            raise ValueError("unsupported device info")
    assert identity(Device()) == {}


@pytest.mark.skipif(os.environ.get("BURNSCOPE_HARDWARE_TESTS") != "1", reason="opt-in GPU microtasks")
def test_real_gpu_reference_and_injected_error(tmp_path):
    np = pytest.importorskip("numpy")
    cl = pytest.importorskip("pyopencl")
    import burnscope.gpu
    devices = [d for p in cl.get_platforms() for d in p.get_devices(device_type=cl.device_type.GPU)]
    if not devices:
        pytest.skip("No OpenCL GPU")
    for device in devices:
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        source = (Path(burnscope.gpu.__file__).parent / "kernels/stress.cl").read_text()
        program = cl.Program(context, source).build(["-cl-std=CL1.2"], cache_dir=str(tmp_path))
        buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE, 256 * 8 * 4)
        actual = np.empty((256, 8), np.float32)
        cl.Kernel(program, "compute")(queue, (256,), None, buffer, np.uint32(16)).wait()
        cl.enqueue_copy(queue, actual, buffer).wait()
        assert np.allclose(actual, float_reference(np.arange(256), 16), rtol=1e-5, atol=1e-6)
        fill, verify = cl.Kernel(program, "fill"), cl.Kernel(program, "verify")
        for pass_ in (0, 1, 4, 35, 36, 39):
            fill(queue, (256,), None, buffer, np.uint32(256), np.uint32(pass_)).wait()
            words = np.empty(256, np.uint32)
            cl.enqueue_copy(queue, words, buffer).wait()
            assert np.array_equal(words, np.array([memory_pattern(i, pass_) for i in range(256)], np.uint32))
            words[17] ^= np.uint32(1)
            cl.enqueue_copy(queue, buffer, words).wait()
            counters = np.array([0, 0xffffffff], np.uint32)
            errbuf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=counters)
            verify(queue, (256,), None, buffer, np.uint32(256), np.uint32(pass_), errbuf).wait()
            cl.enqueue_copy(queue, counters, errbuf).wait()
            assert tuple(counters) == (1, 17)
