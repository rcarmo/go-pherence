#version 450
// X[M,K] * transpose(dequantize(Q8Weight[N,K])) + Bias[N] -> Out[M,N].
// Four row-major signed-int8 weights are packed little-byte-first per uint.
// One F32 symmetric scale belongs to each complete output row.
layout(local_size_x = 16, local_size_y = 16) in;
layout(set=0, binding=0) readonly buffer X { float x[]; };
layout(set=0, binding=1) readonly buffer WeightQ8 { uint weight_packed[]; };
layout(set=0, binding=2) readonly buffer Scale { float scale[]; };
layout(set=0, binding=3) readonly buffer Bias { float bias[]; };
layout(set=0, binding=4) buffer Out { float out_data[]; };
layout(push_constant) uniform P { uint rows; uint inDim; uint outDim; };
shared float tileX[256];
shared float tileW[256];

float load_weight(uint index, uint row) {
    uint word = weight_packed[index >> 2];
    uint byte_value = (word >> (8u * (index & 3u))) & 0xffu;
    float signed_value = byte_value < 128u ? float(byte_value) : float(byte_value) - 256.0;
    return signed_value * scale[row];
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
            if (weightRow < outDim) wv = load_weight(weightRow * inDim + base + tx, weightRow);
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
