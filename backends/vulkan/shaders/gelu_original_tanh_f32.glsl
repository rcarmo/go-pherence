#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) readonly buffer X { float x[]; };
layout(set=0, binding=1) buffer Out { float out_data[]; };
layout(push_constant) uniform P { uint count; };

// Original whisper.cpp/GGML Vulkan op_gelu tanh-form expression in its exact
// operation order: 0.5*x*(2-2/(exp(2*v)+1)), v=sqrt(2/pi)*x*(1+0.044715*x*x).
// Do not substitute erf, tanh(), clamps or further approximation.
float original_gelu(float v) {
    const float a = 0.044715;
    const float c = 0.79788456080286535587989211986876;
    const float val = c * v * (1.0 + a * v * v);
    return 0.5 * v * (2.0 - 2.0 / (exp(2.0 * val) + 1.0));
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < count) {
        out_data[i] = original_gelu(x[i]);
    }
}
