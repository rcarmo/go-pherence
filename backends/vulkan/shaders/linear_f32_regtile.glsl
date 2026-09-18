#version 450
layout(local_size_x=16,local_size_y=16) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float weight[]; };
layout(set=0,binding=2) readonly buffer B { float bias[]; };
layout(set=0,binding=3) buffer O { float output_data[]; };
layout(push_constant) uniform P { uint rows; uint inDim; uint outDim; };
// 32x32 output tile, K tile32. 16x16 lanes each own four output elements.
// Cooperatively loaded row-major X/W tiles; W is consumed transposed. Each
// reduction step shares two activation and two weight loads among four FMAs.
// Invalid row/column/K lanes zero-fill but participate in both barriers.
shared float xt[1024];
shared float wt[1024];
void main() {
    uint cx=gl_LocalInvocationID.x, ry=gl_LocalInvocationID.y;
    uint lane=ry*16+cx;
    uint rowBase=gl_WorkGroupID.y*32, colBase=gl_WorkGroupID.x*32;
    float s00=0.0, s01=0.0, s10=0.0, s11=0.0;
    for (uint base=0; base<inDim; base+=32) {
        for (uint i=lane; i<1024; i+=256) {
            uint r=i/32, k=base+i%32;
            float a=0.0, b=0.0;
            if (k<inDim) {
                if (rowBase+r<rows) a=x[(rowBase+r)*inDim+k];
                if (colBase+r<outDim) b=weight[(colBase+r)*inDim+k];
            }
            xt[i]=a;
            wt[i]=b;
        }
        barrier();
        for (uint k=0; k<32; k++) {
            float a0=xt[ry*32+k], a1=xt[(ry+16)*32+k];
            float b0=wt[cx*32+k], b1=wt[(cx+16)*32+k];
            s00+=a0*b0;
            s01+=a0*b1;
            s10+=a1*b0;
            s11+=a1*b1;
        }
        barrier();
    }
    uint r=rowBase+ry, c=colBase+cx;
    if (r<rows) {
        if (c<outDim) output_data[r*outDim+c]=s00+bias[c];
        if (c+16<outDim) output_data[r*outDim+c+16]=s01+bias[c+16];
    }
    if (r+16<rows) {
        if (c<outDim) output_data[(r+16)*outDim+c]=s10+bias[c];
        if (c+16<outDim) output_data[(r+16)*outDim+c+16]=s11+bias[c+16];
    }
}
