#version 450
// X[M,K] * transpose(WeightF16[N,K]) + Bias[N] -> Out[M,N].
// Two row-major IEEE-F16 weights are packed little-half-first in each uint.
// Storage and arithmetic use only core 32-bit scalar types. Weights widen to
// F32 before multiplication; activations, accumulation, bias and output are F32.
layout(local_size_x = 16, local_size_y = 16) in;
layout(set=0, binding=0) readonly buffer X { float x[]; };
layout(set=0, binding=1) readonly buffer WeightF16 { uint weight_packed[]; };
layout(set=0, binding=2) readonly buffer Bias { float bias[]; };
layout(set=0, binding=3) buffer Out { float out_data[]; };
layout(push_constant) uniform P { uint rows; uint inDim; uint outDim; };
shared float tileX[256];
shared float tileW[256];

float finite_f16_to_f32(uint value) {
    uint sign = (value & 0x8000u) << 16;
    uint exponent = (value >> 10) & 0x1fu;
    uint mantissa = value & 0x03ffu;
    if (exponent == 0u) {
        float subnormal = float(mantissa) * 5.9604644775390625e-8;
        return sign == 0u ? subnormal : -subnormal;
    }
    // Host packing rejects Inf/NaN, so exponent 31 cannot reach this shader.
    return uintBitsToFloat(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

float load_weight(uint index) {
    uint pair = weight_packed[index >> 1];
    uint bits = (index & 1u) == 0u ? (pair & 0xffffu) : (pair >> 16);
    return finite_f16_to_f32(bits);
}

void main() {
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint row = gl_WorkGroupID.y * 16 + ty;
    uint col = gl_WorkGroupID.x * 16 + tx;
    uint weightRow = gl_WorkGroupID.x * 16 + ty;
    uint slot = ty * 16 + tx;
    float sum = 0.0;
    for (uint base = 0; base < inDim; base += 16) {
        float xv = 0.0;
        float wv = 0.0;
        if (base + tx < inDim) {
            if (row < rows) xv = x[row * inDim + base + tx];
            if (weightRow < outDim) wv = load_weight(weightRow * inDim + base + tx);
        }
        tileX[slot] = xv;
        tileW[slot] = wv;
        barrier();
        for (uint k = 0; k < 16; k++) {
            sum += tileX[ty * 16 + k] * tileW[tx * 16 + k];
        }
        barrier();
    }
    if (row < rows && col < outDim) {
        out_data[row * outDim + col] = sum + bias[col];
    }
}
