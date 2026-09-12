#version 450
layout(local_size_x=16,local_size_y=16) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float weight[]; };
layout(set=0,binding=2) readonly buffer B { float bias[]; };
layout(set=0,binding=3) buffer O { float output_data[]; };
layout(push_constant) uniform P { uint inLen; uint inCh; uint outCh; uint outLen; uint stride; uint inputLayout; };
// Fixed kernel3, symmetric zero pad1, stride1/2, no dilation/groups.
// Implicit im2col tiles: no materialised patches or host transpose. Input is
// channel-first(layout0) or time-major(layout1); output is always time-major.
shared float xt[256];
shared float wt[256];
void main() {
    uint cx=gl_LocalInvocationID.x, ry=gl_LocalInvocationID.y;
    uint row=gl_WorkGroupID.y*16+ry;
    uint col=gl_WorkGroupID.x*16+cx;
    uint reduction=inCh*3;
    precise float sum=0.0;
    for (uint base=0; base<reduction; base+=16) {
        uint ix=base+cx;
        float value=0.0;
        if (row<outLen) {
            if (ix<reduction) {
                uint channel=ix/3, tap=ix%3;
                uint padded=row*stride+tap;
                // Skip the left pad before unsigned subtraction.
                if (padded>0) {
                    uint pos=padded-1;
                    if (pos<inLen) {
                        if (inputLayout==0) value=x[channel*inLen+pos];
                        else value=x[pos*inCh+channel];
                    }
                }
            }
        }
        xt[ry*16+cx]=value;
        uint oc=gl_WorkGroupID.x*16+ry;
        value=0.0;
        if (oc<outCh) {
            if (ix<reduction) value=weight[oc*reduction+ix];
        }
        wt[ry*16+cx]=value;
        barrier();
        for (uint j=0; j<16; j++) sum+=xt[ry*16+j]*wt[cx*16+j];
        barrier();
    }
    if (row<outLen) {
        if (col<outCh) output_data[row*outCh+col]=sum+bias[col];
    }
}
