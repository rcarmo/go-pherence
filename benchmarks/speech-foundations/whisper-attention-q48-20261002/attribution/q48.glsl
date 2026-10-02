#version 450
layout(local_size_x=16, local_size_y=16) in;
layout(set=0,binding=0) readonly buffer Q { float q[]; };
layout(set=0,binding=1) readonly buffer K { float k[]; };
layout(set=0,binding=2) readonly buffer V { float v[]; };
layout(set=0,binding=3) buffer Out { float output_data[]; };
layout(push_constant) uniform P { uint seqQ; uint seqKV; uint heads; uint headDim; float scale; };
// Same per-output arithmetic as attention_f32_key32_padded_extent; 48 queries per group,
// lane (x,y) owns queries y+16n (n<3), keys x and x+16, columns x+16i.
shared float qt[3072];
shared float kv[2048];
shared float probabilities[1536];
shared float rowMax[48];
shared float rowSum[48];
shared float rescale[48];
void main() {
    uint x = gl_LocalInvocationID.x;
    uint y = gl_LocalInvocationID.y;
    uint lane = y*16+x;
    uint keyExtent=((seqKV+255u)/256u)*256u;
    uint queryBase = gl_WorkGroupID.x*48;
    uint headBase = gl_WorkGroupID.y*headDim;
    uint width = heads*headDim;
    precise float acc0[4];
    precise float acc1[4];
    precise float acc2[4];
    for (uint i=0; i<4; i++) { acc0[i]=0.0; acc1[i]=0.0; acc2[i]=0.0; }
    for (uint i=lane; i<3072; i+=256) {
        uint r=i/64, d=i%64;
        float value=0.0;
        if (queryBase+r < seqQ) {
            if (d < headDim) value=q[(queryBase+r)*width+headBase+d];
        }
        qt[i]=value;
    }
    if (x<3) { rowMax[y+16*x]=0.0; rowSum[y+16*x]=0.0; }
    barrier();
    uint y1=y+16;
    uint y2=y+32;
    for (uint base=0; base<keyExtent; base+=32) {
        for (uint i=lane; i<2048; i+=256) {
            uint r=i/64, d=i%64;
            float value=0.0;
            if (base+r < seqKV) {
                if (d < headDim) value=k[(base+r)*width+headBase+d];
            }
            kv[i]=value;
        }
        barrier();
        precise float s00=0.0,s01=0.0,s10=0.0,s11=0.0,s20=0.0,s21=0.0;
        for(uint channel=0;channel<64;channel+=4){
        {float k0=kv[x*64+(channel+0)];float k1=kv[(x+16)*64+(channel+0)];float a0=qt[y*64+(channel+0)];s00=fma(a0,k0,s00);s01=fma(a0,k1,s01);float a1=qt[y1*64+(channel+0)];s10=fma(a1,k0,s10);s11=fma(a1,k1,s11);float a2=qt[y2*64+(channel+0)];s20=fma(a2,k0,s20);s21=fma(a2,k1,s21);}
        {float k0=kv[x*64+(channel+1)];float k1=kv[(x+16)*64+(channel+1)];float a0=qt[y*64+(channel+1)];s00=fma(a0,k0,s00);s01=fma(a0,k1,s01);float a1=qt[y1*64+(channel+1)];s10=fma(a1,k0,s10);s11=fma(a1,k1,s11);float a2=qt[y2*64+(channel+1)];s20=fma(a2,k0,s20);s21=fma(a2,k1,s21);}
        {float k0=kv[x*64+(channel+2)];float k1=kv[(x+16)*64+(channel+2)];float a0=qt[y*64+(channel+2)];s00=fma(a0,k0,s00);s01=fma(a0,k1,s01);float a1=qt[y1*64+(channel+2)];s10=fma(a1,k0,s10);s11=fma(a1,k1,s11);float a2=qt[y2*64+(channel+2)];s20=fma(a2,k0,s20);s21=fma(a2,k1,s21);}
        {float k0=kv[x*64+(channel+3)];float k1=kv[(x+16)*64+(channel+3)];float a0=qt[y*64+(channel+3)];s00=fma(a0,k0,s00);s01=fma(a0,k1,s01);float a1=qt[y1*64+(channel+3)];s10=fma(a1,k0,s10);s11=fma(a1,k1,s11);float a2=qt[y2*64+(channel+3)];s20=fma(a2,k0,s20);s21=fma(a2,k1,s21);}
        }
        probabilities[y*32+x]=s00*scale;
        probabilities[y*32+x+16]=s01*scale;
        probabilities[y1*32+x]=s10*scale;
        probabilities[y1*32+x+16]=s11*scale;
        probabilities[y2*32+x]=s20*scale;
        probabilities[y2*32+x+16]=s21*scale;
        barrier();
        if (x<3) {
            uint row=y+16*x;
            float m=probabilities[row*32];
            for (uint j=1; j<32; j++) {
                if (base+j < keyExtent) {
                    if (m < probabilities[row*32+j]) m=probabilities[row*32+j];
                }
            }
            precise float alpha=0.0;
            if (base > 0) {
                if (m < rowMax[row]) m=rowMax[row];
                alpha=exp(rowMax[row]-m);
            }
            rowMax[row]=m;
            rescale[row]=alpha;
        }
        barrier();
        for(uint j=x; j<32; j+=16) {
            {precise float p=0.0;if(base+j<keyExtent)p=exp(probabilities[y*32+j]-rowMax[y]);probabilities[y*32+j]=p;}
            {precise float p=0.0;if(base+j<keyExtent)p=exp(probabilities[y1*32+j]-rowMax[y1]);probabilities[y1*32+j]=p;}
            {precise float p=0.0;if(base+j<keyExtent)p=exp(probabilities[y2*32+j]-rowMax[y2]);probabilities[y2*32+j]=p;}
        }
        barrier();
        if(x<3) {
            uint row=y+16*x;
            precise float sum=0.0;
            for(uint j=0;j<32;j++)sum+=probabilities[row*32+j];
            rowSum[row]=rowSum[row]*rescale[row]+sum;
        }
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
        {float r=rescale[y];for(uint i=0;i<4;i++)acc0[i]*=r;}
        {float r=rescale[y1];for(uint i=0;i<4;i++)acc1[i]*=r;}
        {float r=rescale[y2];for(uint i=0;i<4;i++)acc2[i]*=r;}
        for(uint j=0;j<32;j++){
         float v0=kv[j*64+x+0],v1=kv[j*64+x+16],v2=kv[j*64+x+32],v3=kv[j*64+x+48];
         {float p=probabilities[y*32+j];acc0[0]=fma(p,v0,acc0[0]);acc0[1]=fma(p,v1,acc0[1]);acc0[2]=fma(p,v2,acc0[2]);acc0[3]=fma(p,v3,acc0[3]);}
         {float p=probabilities[y1*32+j];acc1[0]=fma(p,v0,acc1[0]);acc1[1]=fma(p,v1,acc1[1]);acc1[2]=fma(p,v2,acc1[2]);acc1[3]=fma(p,v3,acc1[3]);}
         {float p=probabilities[y2*32+j];acc2[0]=fma(p,v0,acc2[0]);acc2[1]=fma(p,v1,acc2[1]);acc2[2]=fma(p,v2,acc2[2]);acc2[3]=fma(p,v3,acc2[3]);}
        }
        barrier();
    }
    for (uint i=0; i<4; i++) {
        uint d=x+i*16;
        if (d < headDim) {
            if (queryBase+y < seqQ) output_data[(queryBase+y)*width+headBase+d]=acc0[i]/rowSum[y];
            if (queryBase+y1 < seqQ) output_data[(queryBase+y1)*width+headBase+d]=acc1[i]/rowSum[y1];
            if (queryBase+y2 < seqQ) output_data[(queryBase+y2)*width+headBase+d]=acc2[i]/rowSum[y2];
        }
    }
}
