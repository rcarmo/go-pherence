#version 450
layout(local_size_x=16, local_size_y=16) in;
layout(set=0,binding=0) readonly buffer Q { float q[]; };
layout(set=0,binding=1) readonly buffer K { float k[]; };
layout(set=0,binding=2) readonly buffer V { float v[]; };
layout(set=0,binding=3) buffer Out { float output_data[]; };
layout(push_constant) uniform P { uint seqQ; uint seqKV; uint heads; uint headDim; float scale; };

// Non-causal F32 attention. One group owns 16 queries of one head; lanes
// cooperate on 16-key tiles and retain four output columns each in registers.
// K and V reuse the same shared tile, separated by workgroup barriers.
// Online max/sum rescaling avoids storing the full [heads,seqQ,seqKV] matrix.
// headDim <= 64. Invalid query/key/channel lanes zero-pad and reach barriers.
shared float qt[1024];
shared float kv[1024];
shared float probabilities[256];
shared float rowMax[16];
shared float rowSum[16];
shared float rescale[16];

void main() {
    uint x = gl_LocalInvocationID.x;
    uint y = gl_LocalInvocationID.y;
    uint lane = y*16+x;
    uint queryBase = gl_WorkGroupID.x*16;
    uint headBase = gl_WorkGroupID.y*headDim;
    uint width = heads*headDim;
    precise float acc[4];
    for (uint i=0; i<4; i++) acc[i]=0.0;
    for (uint i=lane; i<1024; i+=256) {
        uint r=i/64, d=i%64;
        float value=0.0;
        if (queryBase+r < seqQ) {
            if (d < headDim) value=q[(queryBase+r)*width+headBase+d];
        }
        qt[i]=value;
    }
    if (x==0) { rowMax[y]=0.0; rowSum[y]=0.0; }
    barrier();
    for (uint base=0; base<seqKV; base+=16) {
        for (uint i=lane; i<1024; i+=256) {
            uint r=i/64, d=i%64;
            float value=0.0;
            if (base+r < seqKV) {
                if (d < headDim) value=k[(base+r)*width+headBase+d];
            }
            kv[i]=value;
        }
        barrier();
        precise float score=0.0;
        for (uint d=0; d<headDim; d++) score+=qt[y*64+d]*kv[x*64+d];
        probabilities[lane]=score*scale;
        barrier();
        if (x==0) {
            float m=probabilities[y*16]; // base < seqKV: key zero is always valid.
            for (uint j=1; j<16; j++) {
                if (base+j < seqKV) {
                    if (m < probabilities[y*16+j]) m=probabilities[y*16+j];
                }
            }
            precise float alpha=0.0;
            if (base > 0) {
                if (m < rowMax[y]) m=rowMax[y];
                alpha=exp(rowMax[y]-m);
            }
            precise float sum=0.0;
            for (uint j=0; j<16; j++) {
                precise float p=0.0;
                if (base+j < seqKV) p=exp(probabilities[y*16+j]-m);
                probabilities[y*16+j]=p;
                sum+=p;
            }
            rowSum[y]=rowSum[y]*alpha+sum;
            rowMax[y]=m;
            rescale[y]=alpha;
        }
        // All K readers and probability writers finish before K becomes V.
        barrier();
        for (uint i=lane; i<1024; i+=256) {
            uint r=i/64, d=i%64;
            float value=0.0;
            if (base+r < seqKV) {
                if (d < headDim) value=v[(base+r)*width+headBase+d];
            }
            kv[i]=value;
        }
        barrier();
        for (uint i=0; i<4; i++) {
            uint d=x+i*16;
            acc[i]*=rescale[y];
            if (d < headDim) {
                for (uint j=0; j<16; j++) acc[i]+=probabilities[y*16+j]*kv[j*64+d];
            }
        }
        barrier(); // All V/probability readers finish before the next key tile.
    }
    if (queryBase+y < seqQ) {
        for (uint i=0; i<4; i++) {
            uint d=x+i*16;
            if (d < headDim) output_data[(queryBase+y)*width+headBase+d]=acc[i]/rowSum[y];
        }
    }
}
