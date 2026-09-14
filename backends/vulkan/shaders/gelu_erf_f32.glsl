#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) readonly buffer X { float x[]; };
layout(set=0, binding=1) buffer Out { float out_data[]; };
layout(push_constant) uniform P { uint count; };

// Erf-based GELU via the A&S 7.1.26 erfc approximation (p=0.3275911).
// See benchmark docs for the formula source and measured offline error.
// Saturating at |v| >= 10 also bypasses a*a for extreme finite F32 inputs;
// 10 is a tail cutoff, not the F32 overflow threshold. GPU Exp rounding and
// subnormal handling still require qualification.
float gelu_erf(float v) {
    if (v >= 10.0) return v;
    if (v <= -10.0) return 0.0;
    precise float a = abs(v) * 0.7071067811865475244;
    precise float t = 1.0 / (1.0 + 0.3275911 * a);
    precise float p = 1.061405429;
    p = p * t - 1.453152027;
    p = p * t + 1.421413741;
    p = p * t - 0.284496736;
    p = p * t + 0.254829592;
    precise float tail = (p * t) * exp(-a * a);
    // For negative v, avoid cancellation in 1 + erf(v/sqrt(2)).
    precise float result;
    if (v < 0.0) result = (0.5 * v) * tail;
    else result = v * (1.0 - 0.5 * tail);
    return result;
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < count) out_data[i] = gelu_erf(x[i]);
}
