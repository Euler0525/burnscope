// OpenCL 1.2: bounded FP32 FMA workload and deterministic memory checks.
uint mix32(uint x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    return x ^ (x >> 16);
}
uint pattern(uint index, uint pass) {
    if (pass == 0) return 0;
    if (pass == 1) return 0xffffffffU;
    if (pass == 2) return 0xaaaaaaaaU;
    if (pass == 3) return 0x55555555U;
    if (pass < 36) return 1U << (pass - 4);
    return mix32(index ^ (pass * 0x9e3779b9U));
}
__kernel void compute(__global float4* output, uint iterations) {
    uint i = get_global_id(0);
    float seed = 0.125f + (float)(i & 15U) * 0.03125f;
    float4 a = (float4)(seed, seed + .125f, seed + .25f, seed + .375f);
    float4 b = a + .5f;
    for (uint k = 0; k < iterations; ++k) {
        a = fma(a, (float4)(1.0001f), (float4)(.0001f));
        b = fma(b, (float4)(1.0001f), (float4)(.0001f));
    }
    output[2*i] = a;
    output[2*i+1] = b;
}
__kernel void fill(__global uint* output, uint words, uint pass) {
    uint i = get_global_id(0);
    if (i < words) output[i] = pattern(i, pass);
}
__kernel void verify(__global const uint* input, uint words, uint pass, __global uint* errors) {
    uint i = get_global_id(0);
    if (i < words && input[i] != pattern(i, pass)) {
        atomic_inc(errors);
        atomic_cmpxchg(errors + 1, 0xffffffffU, i);
    }
}
__kernel void bandwidth(__global uint* input, uint words) {
    uint i = get_global_id(0);
    if (i < words) input[i] ^= 0xa5a5a5a5U;
}

