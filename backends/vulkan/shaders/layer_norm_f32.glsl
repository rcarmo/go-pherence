#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) readonly buffer X { float x[]; };
layout(set=0, binding=1) readonly buffer Weight { float weight[]; };
layout(set=0, binding=2) readonly buffer Bias { float bias[]; };
layout(set=0, binding=3) buffer Out { float out_data[]; };
layout(push_constant) uniform P { uint rows; uint width; float eps; };
shared float scratch[256];
void main() {
    uint row = gl_WorkGroupID.x;
    // Uniform workgroup condition: every lane must reach the same barriers.
    if (row >= rows) return;
    uint tid = gl_LocalInvocationID.x;
    uint base = row * width;
    float sum = 0.0;
    for (uint col = tid; col < width; col += 256) sum += x[base + col];
    scratch[tid] = sum;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        barrier();
    }
    float mean = scratch[0] / float(width);
    // Every lane must load mean before any lane reuses shared scratch.
    barrier();
    float squared = 0.0;
    for (uint col = tid; col < width; col += 256) {
        float delta = x[base + col] - mean;
        squared += delta * delta;
    }
    scratch[tid] = squared;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        barrier();
    }
    float inv = inversesqrt(scratch[0] / float(width) + eps);
    // All variance input reads are complete. Exact X/Out alias is safe: each
    // row/column is now read and written by its own invocation only. Output
    // must not overlap shared weight/bias or partially overlap X.
    for (uint col = tid; col < width; col += 256) {
        out_data[base + col] = ((x[base + col] - mean) * inv) * weight[col] + bias[col];
    }
}
