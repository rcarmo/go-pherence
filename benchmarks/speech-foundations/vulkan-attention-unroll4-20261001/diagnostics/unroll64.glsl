#version 450
layout(local_size_x=16, local_size_y=16) in;
layout(set=0,binding=0) readonly buffer Q { float q[]; };
layout(set=0,binding=1) readonly buffer K { float k[]; };
layout(set=0,binding=2) readonly buffer V { float v[]; };
layout(set=0,binding=3) buffer Out { float output_data[]; };
layout(push_constant) uniform P { uint seqQ; uint seqKV; uint heads; uint headDim; float scale; };

// Non-causal F32 attention. One group owns 16 queries of one head; lanes
// cooperate on 32-key tiles and retain four output columns each in registers.
// K and V reuse the same shared tile, separated by workgroup barriers.
// Online max/sum rescaling avoids storing the full [heads,seqQ,seqKV] matrix.
// headDim <= 64. Invalid query/key/channel lanes zero-pad and reach barriers.
shared float qt[1024];
shared float kv[2048];
shared float probabilities[512];
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
    for (uint base=0; base<seqKV; base+=32) {
        for (uint i=lane; i<2048; i+=256) {
            uint r=i/64, d=i%64;
            float value=0.0;
            if (base+r < seqKV) {
                if (d < headDim) value=k[(base+r)*width+headBase+d];
            }
            kv[i]=value;
        }
        barrier();
        precise float score0=0.0,score1=0.0;
        {float qvalue=qt[y*64+0];score0=fma(qvalue,kv[x*64+0],score0);score1=fma(qvalue,kv[(x+16)*64+0],score1);}
        {float qvalue=qt[y*64+1];score0=fma(qvalue,kv[x*64+1],score0);score1=fma(qvalue,kv[(x+16)*64+1],score1);}
        {float qvalue=qt[y*64+2];score0=fma(qvalue,kv[x*64+2],score0);score1=fma(qvalue,kv[(x+16)*64+2],score1);}
        {float qvalue=qt[y*64+3];score0=fma(qvalue,kv[x*64+3],score0);score1=fma(qvalue,kv[(x+16)*64+3],score1);}
        {float qvalue=qt[y*64+4];score0=fma(qvalue,kv[x*64+4],score0);score1=fma(qvalue,kv[(x+16)*64+4],score1);}
        {float qvalue=qt[y*64+5];score0=fma(qvalue,kv[x*64+5],score0);score1=fma(qvalue,kv[(x+16)*64+5],score1);}
        {float qvalue=qt[y*64+6];score0=fma(qvalue,kv[x*64+6],score0);score1=fma(qvalue,kv[(x+16)*64+6],score1);}
        {float qvalue=qt[y*64+7];score0=fma(qvalue,kv[x*64+7],score0);score1=fma(qvalue,kv[(x+16)*64+7],score1);}
        {float qvalue=qt[y*64+8];score0=fma(qvalue,kv[x*64+8],score0);score1=fma(qvalue,kv[(x+16)*64+8],score1);}
        {float qvalue=qt[y*64+9];score0=fma(qvalue,kv[x*64+9],score0);score1=fma(qvalue,kv[(x+16)*64+9],score1);}
        {float qvalue=qt[y*64+10];score0=fma(qvalue,kv[x*64+10],score0);score1=fma(qvalue,kv[(x+16)*64+10],score1);}
        {float qvalue=qt[y*64+11];score0=fma(qvalue,kv[x*64+11],score0);score1=fma(qvalue,kv[(x+16)*64+11],score1);}
        {float qvalue=qt[y*64+12];score0=fma(qvalue,kv[x*64+12],score0);score1=fma(qvalue,kv[(x+16)*64+12],score1);}
        {float qvalue=qt[y*64+13];score0=fma(qvalue,kv[x*64+13],score0);score1=fma(qvalue,kv[(x+16)*64+13],score1);}
        {float qvalue=qt[y*64+14];score0=fma(qvalue,kv[x*64+14],score0);score1=fma(qvalue,kv[(x+16)*64+14],score1);}
        {float qvalue=qt[y*64+15];score0=fma(qvalue,kv[x*64+15],score0);score1=fma(qvalue,kv[(x+16)*64+15],score1);}
        {float qvalue=qt[y*64+16];score0=fma(qvalue,kv[x*64+16],score0);score1=fma(qvalue,kv[(x+16)*64+16],score1);}
        {float qvalue=qt[y*64+17];score0=fma(qvalue,kv[x*64+17],score0);score1=fma(qvalue,kv[(x+16)*64+17],score1);}
        {float qvalue=qt[y*64+18];score0=fma(qvalue,kv[x*64+18],score0);score1=fma(qvalue,kv[(x+16)*64+18],score1);}
        {float qvalue=qt[y*64+19];score0=fma(qvalue,kv[x*64+19],score0);score1=fma(qvalue,kv[(x+16)*64+19],score1);}
        {float qvalue=qt[y*64+20];score0=fma(qvalue,kv[x*64+20],score0);score1=fma(qvalue,kv[(x+16)*64+20],score1);}
        {float qvalue=qt[y*64+21];score0=fma(qvalue,kv[x*64+21],score0);score1=fma(qvalue,kv[(x+16)*64+21],score1);}
        {float qvalue=qt[y*64+22];score0=fma(qvalue,kv[x*64+22],score0);score1=fma(qvalue,kv[(x+16)*64+22],score1);}
        {float qvalue=qt[y*64+23];score0=fma(qvalue,kv[x*64+23],score0);score1=fma(qvalue,kv[(x+16)*64+23],score1);}
        {float qvalue=qt[y*64+24];score0=fma(qvalue,kv[x*64+24],score0);score1=fma(qvalue,kv[(x+16)*64+24],score1);}
        {float qvalue=qt[y*64+25];score0=fma(qvalue,kv[x*64+25],score0);score1=fma(qvalue,kv[(x+16)*64+25],score1);}
        {float qvalue=qt[y*64+26];score0=fma(qvalue,kv[x*64+26],score0);score1=fma(qvalue,kv[(x+16)*64+26],score1);}
        {float qvalue=qt[y*64+27];score0=fma(qvalue,kv[x*64+27],score0);score1=fma(qvalue,kv[(x+16)*64+27],score1);}
        {float qvalue=qt[y*64+28];score0=fma(qvalue,kv[x*64+28],score0);score1=fma(qvalue,kv[(x+16)*64+28],score1);}
        {float qvalue=qt[y*64+29];score0=fma(qvalue,kv[x*64+29],score0);score1=fma(qvalue,kv[(x+16)*64+29],score1);}
        {float qvalue=qt[y*64+30];score0=fma(qvalue,kv[x*64+30],score0);score1=fma(qvalue,kv[(x+16)*64+30],score1);}
        {float qvalue=qt[y*64+31];score0=fma(qvalue,kv[x*64+31],score0);score1=fma(qvalue,kv[(x+16)*64+31],score1);}
        {float qvalue=qt[y*64+32];score0=fma(qvalue,kv[x*64+32],score0);score1=fma(qvalue,kv[(x+16)*64+32],score1);}
        {float qvalue=qt[y*64+33];score0=fma(qvalue,kv[x*64+33],score0);score1=fma(qvalue,kv[(x+16)*64+33],score1);}
        {float qvalue=qt[y*64+34];score0=fma(qvalue,kv[x*64+34],score0);score1=fma(qvalue,kv[(x+16)*64+34],score1);}
        {float qvalue=qt[y*64+35];score0=fma(qvalue,kv[x*64+35],score0);score1=fma(qvalue,kv[(x+16)*64+35],score1);}
        {float qvalue=qt[y*64+36];score0=fma(qvalue,kv[x*64+36],score0);score1=fma(qvalue,kv[(x+16)*64+36],score1);}
        {float qvalue=qt[y*64+37];score0=fma(qvalue,kv[x*64+37],score0);score1=fma(qvalue,kv[(x+16)*64+37],score1);}
        {float qvalue=qt[y*64+38];score0=fma(qvalue,kv[x*64+38],score0);score1=fma(qvalue,kv[(x+16)*64+38],score1);}
        {float qvalue=qt[y*64+39];score0=fma(qvalue,kv[x*64+39],score0);score1=fma(qvalue,kv[(x+16)*64+39],score1);}
        {float qvalue=qt[y*64+40];score0=fma(qvalue,kv[x*64+40],score0);score1=fma(qvalue,kv[(x+16)*64+40],score1);}
        {float qvalue=qt[y*64+41];score0=fma(qvalue,kv[x*64+41],score0);score1=fma(qvalue,kv[(x+16)*64+41],score1);}
        {float qvalue=qt[y*64+42];score0=fma(qvalue,kv[x*64+42],score0);score1=fma(qvalue,kv[(x+16)*64+42],score1);}
        {float qvalue=qt[y*64+43];score0=fma(qvalue,kv[x*64+43],score0);score1=fma(qvalue,kv[(x+16)*64+43],score1);}
        {float qvalue=qt[y*64+44];score0=fma(qvalue,kv[x*64+44],score0);score1=fma(qvalue,kv[(x+16)*64+44],score1);}
        {float qvalue=qt[y*64+45];score0=fma(qvalue,kv[x*64+45],score0);score1=fma(qvalue,kv[(x+16)*64+45],score1);}
        {float qvalue=qt[y*64+46];score0=fma(qvalue,kv[x*64+46],score0);score1=fma(qvalue,kv[(x+16)*64+46],score1);}
        {float qvalue=qt[y*64+47];score0=fma(qvalue,kv[x*64+47],score0);score1=fma(qvalue,kv[(x+16)*64+47],score1);}
        {float qvalue=qt[y*64+48];score0=fma(qvalue,kv[x*64+48],score0);score1=fma(qvalue,kv[(x+16)*64+48],score1);}
        {float qvalue=qt[y*64+49];score0=fma(qvalue,kv[x*64+49],score0);score1=fma(qvalue,kv[(x+16)*64+49],score1);}
        {float qvalue=qt[y*64+50];score0=fma(qvalue,kv[x*64+50],score0);score1=fma(qvalue,kv[(x+16)*64+50],score1);}
        {float qvalue=qt[y*64+51];score0=fma(qvalue,kv[x*64+51],score0);score1=fma(qvalue,kv[(x+16)*64+51],score1);}
        {float qvalue=qt[y*64+52];score0=fma(qvalue,kv[x*64+52],score0);score1=fma(qvalue,kv[(x+16)*64+52],score1);}
        {float qvalue=qt[y*64+53];score0=fma(qvalue,kv[x*64+53],score0);score1=fma(qvalue,kv[(x+16)*64+53],score1);}
        {float qvalue=qt[y*64+54];score0=fma(qvalue,kv[x*64+54],score0);score1=fma(qvalue,kv[(x+16)*64+54],score1);}
        {float qvalue=qt[y*64+55];score0=fma(qvalue,kv[x*64+55],score0);score1=fma(qvalue,kv[(x+16)*64+55],score1);}
        {float qvalue=qt[y*64+56];score0=fma(qvalue,kv[x*64+56],score0);score1=fma(qvalue,kv[(x+16)*64+56],score1);}
        {float qvalue=qt[y*64+57];score0=fma(qvalue,kv[x*64+57],score0);score1=fma(qvalue,kv[(x+16)*64+57],score1);}
        {float qvalue=qt[y*64+58];score0=fma(qvalue,kv[x*64+58],score0);score1=fma(qvalue,kv[(x+16)*64+58],score1);}
        {float qvalue=qt[y*64+59];score0=fma(qvalue,kv[x*64+59],score0);score1=fma(qvalue,kv[(x+16)*64+59],score1);}
        {float qvalue=qt[y*64+60];score0=fma(qvalue,kv[x*64+60],score0);score1=fma(qvalue,kv[(x+16)*64+60],score1);}
        {float qvalue=qt[y*64+61];score0=fma(qvalue,kv[x*64+61],score0);score1=fma(qvalue,kv[(x+16)*64+61],score1);}
        {float qvalue=qt[y*64+62];score0=fma(qvalue,kv[x*64+62],score0);score1=fma(qvalue,kv[(x+16)*64+62],score1);}
        {float qvalue=qt[y*64+63];score0=fma(qvalue,kv[x*64+63],score0);score1=fma(qvalue,kv[(x+16)*64+63],score1);}
        probabilities[y*32+x]=score0*scale;
        probabilities[y*32+x+16]=score1*scale;
        barrier();
        if (x==0) {
            float m=probabilities[y*32]; // base < seqKV: key zero is always valid.
            for (uint j=1; j<32; j++) {
                if (base+j < seqKV) {
                    if (m < probabilities[y*32+j]) m=probabilities[y*32+j];
                }
            }
            precise float alpha=0.0;
            if (base > 0) {
                if (m < rowMax[y]) m=rowMax[y];
                alpha=exp(rowMax[y]-m);
            }
            rowMax[y]=m;
            rescale[y]=alpha;
        }
        barrier();
        for(uint j=x; j<32; j+=16) {
            precise float p=0.0;
            if(base+j<seqKV)p=exp(probabilities[y*32+j]-rowMax[y]);
            probabilities[y*32+j]=p;
        }
        barrier();
        if(x==0) {
            precise float sum=0.0;
            for(uint j=0;j<32;j++)sum+=probabilities[y*32+j];
            rowSum[y]=rowSum[y]*rescale[y]+sum;
        }
        // All K readers and probability writers finish before K becomes V.
        barrier();
        for (uint i=lane; i<2048; i+=256) {
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
                for (uint j=0; j<32; j++) acc[i]=fma(probabilities[y*32+j],kv[j*64+d],acc[i]);
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
