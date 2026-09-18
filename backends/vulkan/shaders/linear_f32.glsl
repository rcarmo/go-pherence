#version 450
// X[M,K] * transpose(Weight[N,K]) + Bias[N] -> Out[M,N].
// One invocation/output, with coalesced loads into two 16x16 shared tiles.
layout(local_size_x = 16, local_size_y = 16) in;
layout(set=0, binding=0) readonly buffer X { float x[]; };
layout(set=0, binding=1) readonly buffer Weight { float weight[]; };
layout(set=0, binding=2) readonly buffer Bias { float bias[]; };
layout(set=0, binding=3) buffer Out { float out_data[]; };
layout(push_constant) uniform P { uint rows; uint inDim; uint outDim; };
shared float tileX[256];
shared float tileW[256];
void main() {
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint row = gl_WorkGroupID.y * 16 + ty;
    uint col = gl_WorkGroupID.x * 16 + tx;
    uint weightRow = gl_WorkGroupID.x * 16 + ty;
    uint slot = ty * 16 + tx;
    float sum = 0.0;
    // No per-lane return before barriers: edge lanes zero-pad shared inputs.
    for (uint base = 0; base < inDim; base += 16) {
        float xv = 0.0;
        float wv = 0.0;
        if (base + tx < inDim) {
            if (row < rows) xv = x[row * inDim + base + tx];
            if (weightRow < outDim) wv = weight[weightRow * inDim + base + tx];
        }
        tileX[slot] = xv;
        tileW[slot] = wv;
        barrier();
        for (uint k = 0; k < 16; k++) {
            sum += tileX[ty * 16 + k] * tileW[tx * 16 + k];
        }
        // Every reader finishes before the next tile overwrites shared data.
        barrier();
    }
    if (row < rows) {
        if (col < outDim) out_data[row * outDim + col] = sum + bias[col];
    }
}
