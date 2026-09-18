#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) readonly buffer X { float x[]; };
layout(set=0, binding=1) readonly buffer Scale { float scale[]; };
layout(set=0, binding=2) readonly buffer Shift { float shift[]; };
layout(set=0, binding=3) buffer Out { float out_data[]; };
layout(push_constant) uniform P { uint channels; uint spatial; uint relu; };

// Channel-major affine transform for inference BatchNorm coefficients prepared
// by the caller. Explicit fma defines one-rounding F32 arithmetic.
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint count = channels * spatial;
    if (i < count) {
        uint channel = i / spatial;
        float value = fma(x[i], scale[channel], shift[channel]);
        if (relu != 0 && !(value > 0.0)) value = 0.0;
        out_data[i] = value;
    }
}
