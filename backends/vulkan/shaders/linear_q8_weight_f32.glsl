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

// 32x32 output tile and K tile. Each lane owns four output elements. For the
// common word-aligned geometry, every lane loads exactly one packed word and
// expands four adjacent Q8 values into the complete 32x32 shared weight tile.
// The odd-K fallback preserves the format's general shape contract.
shared float tileX[1024];
shared float tileW[1024];

float signed_byte(uint word, uint lane) {
    uint byte_value = (word >> (8u * lane)) & 0xffu;
    return byte_value < 128u ? float(byte_value) : float(byte_value) - 256.0;
}

float load_weight(uint index) {
    return signed_byte(weight_packed[index >> 2], index & 3u);
}

void main() {
    uint cx = gl_LocalInvocationID.x;
    uint ry = gl_LocalInvocationID.y;
    uint lane = ry * 16u + cx;
    uint rowBase = gl_WorkGroupID.y * 32u;
    uint colBase = gl_WorkGroupID.x * 32u;
    float s00 = 0.0;
    float s01 = 0.0;
    float s10 = 0.0;
    float s11 = 0.0;

    for (uint base = 0u; base < inDim; base += 32u) {
        // Four coalesced activation loads per lane cover 32 rows x 32 K.
        for (uint i = lane; i < 1024u; i += 256u) {
            uint r = i / 32u;
            uint k = base + i % 32u;
            tileX[i] = (k < inDim && rowBase + r < rows) ? x[(rowBase + r) * inDim + k] : 0.0;
        }

        if ((inDim & 3u) == 0u) {
            // 32 rows x eight packed words: one word per invocation.
            uint wr = lane / 8u;
            uint wordColumn = lane & 7u;
            uint first = base + 4u * wordColumn;
            uint word = 0u;
            if (first < inDim && colBase + wr < outDim) {
                word = weight_packed[((colBase + wr) * inDim + first) >> 2];
            }
            uint tileBase = wr * 32u + 4u * wordColumn;
            for (uint q = 0u; q < 4u; q++) {
                tileW[tileBase + q] = first + q < inDim ? signed_byte(word, q) : 0.0;
            }
        } else {
            for (uint i = lane; i < 1024u; i += 256u) {
                uint wr = i / 32u;
                uint k = base + i % 32u;
                tileW[i] = (k < inDim && colBase + wr < outDim) ? load_weight((colBase + wr) * inDim + k) : 0.0;
            }
        }
        barrier();
        for (uint k = 0u; k < 32u; k++) {
            float a0 = tileX[ry * 32u + k];
            float a1 = tileX[(ry + 16u) * 32u + k];
            float b0 = tileW[cx * 32u + k];
            float b1 = tileW[(cx + 16u) * 32u + k];
            s00 += a0 * b0;
            s01 += a0 * b1;
            s10 += a1 * b0;
            s11 += a1 * b1;
        }
        barrier();
    }

    uint r = rowBase + ry;
    uint c = colBase + cx;
    if (r < rows) {
        if (c < outDim) out_data[r * outDim + c] = s00 * scale[c] + bias[c];
        if (c + 16u < outDim) out_data[r * outDim + c + 16u] = s01 * scale[c + 16u] + bias[c + 16u];
    }
    if (r + 16u < rows) {
        if (c < outDim) out_data[(r + 16u) * outDim + c] = s10 * scale[c] + bias[c];
        if (c + 16u < outDim) out_data[(r + 16u) * outDim + c + 16u] = s11 * scale[c + 16u] + bias[c + 16u];
    }
}
