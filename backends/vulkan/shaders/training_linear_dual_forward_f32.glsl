#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float x[];};
layout(set=0,binding=1) readonly buffer T {float xt[];};
layout(set=0,binding=2) readonly buffer W {float weight[];};
layout(set=0,binding=3) readonly buffer B {float bias[];};
layout(set=0,binding=4) writeonly buffer O {float result[];};
// mode0=primal,1=tangent. Tangent bias is zero, not trainable/sampled.
layout(push_constant) uniform Args {uint rows;uint inputDim;uint outputDim;uint mode;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=a.rows*a.outputDim)return;
 uint row=i/a.outputDim,col=i%a.outputDim;
 // DenseNTTo's blocked CPU dispatch is bounded at 576 rows. Larger
 // sequences use SgemmNT's four-vector accumulator and tail association.
 if(a.rows>576u){
  precise float v0[8],v1[8],v2[8],v3[8];
  for(uint lane=0u;lane<8u;lane++){v0[lane]=0.0;v1[lane]=0.0;v2[lane]=0.0;v3[lane]=0.0;}
  uint p=0u;
  for(;p+32u<a.inputDim+1u;p+=32u){
   for(uint lane=0u;lane<8u;lane++){
    precise float value=a.mode==0u?x[row*a.inputDim+p+lane]:xt[row*a.inputDim+p+lane];
    v0[lane]=fma(value,weight[col*a.inputDim+p+lane],v0[lane]);
    value=a.mode==0u?x[row*a.inputDim+p+8u+lane]:xt[row*a.inputDim+p+8u+lane];
    v1[lane]=fma(value,weight[col*a.inputDim+p+8u+lane],v1[lane]);
    value=a.mode==0u?x[row*a.inputDim+p+16u+lane]:xt[row*a.inputDim+p+16u+lane];
    v2[lane]=fma(value,weight[col*a.inputDim+p+16u+lane],v2[lane]);
    value=a.mode==0u?x[row*a.inputDim+p+24u+lane]:xt[row*a.inputDim+p+24u+lane];
    v3[lane]=fma(value,weight[col*a.inputDim+p+24u+lane],v3[lane]);
   }
  }
  for(uint lane=0u;lane<8u;lane++){v0[lane]=v0[lane]+v2[lane];v1[lane]=v1[lane]+v3[lane];}
  if(p+16u<a.inputDim+1u){
   for(uint lane=0u;lane<8u;lane++){
    precise float value=a.mode==0u?x[row*a.inputDim+p+lane]:xt[row*a.inputDim+p+lane];
    v0[lane]=fma(value,weight[col*a.inputDim+p+lane],v0[lane]);
    value=a.mode==0u?x[row*a.inputDim+p+8u+lane]:xt[row*a.inputDim+p+8u+lane];
    v1[lane]=fma(value,weight[col*a.inputDim+p+8u+lane],v1[lane]);
   }
   p+=16u;
  }
  for(uint lane=0u;lane<8u;lane++)v0[lane]=v0[lane]+v1[lane];
  if(p+8u<a.inputDim+1u){
   for(uint lane=0u;lane<8u;lane++){
    precise float value=a.mode==0u?x[row*a.inputDim+p+lane]:xt[row*a.inputDim+p+lane];
    v0[lane]=fma(value,weight[col*a.inputDim+p+lane],v0[lane]);
   }
   p+=8u;
  }
  precise float p0=v0[0]+v0[4],p1=v0[1]+v0[5],p2=v0[2]+v0[6],p3=v0[3]+v0[7];
  precise float total=(p0+p1)+(p2+p3);
  for(;p<a.inputDim;p++){
   precise float value=a.mode==0u?x[row*a.inputDim+p]:xt[row*a.inputDim+p];
   total=fma(value,weight[col*a.inputDim+p],total);
  }
  if(a.mode==0u)total=total+bias[col];result[i]=total;return;
 }
 // CPU multi-row dense dispatch accumulates independently reduced K=128
 // blocks. Preserve that association for its admitted shape class.
 if(a.rows>=16u){if(a.inputDim>=64u){if(a.outputDim>=64u){
  precise float total=0.0;
  for(uint block=0u;block<a.inputDim;block+=128u){
   precise float lanes[8],upper[8];
   for(uint lane=0u;lane<8u;lane++){lanes[lane]=0.0;upper[lane]=0.0;}
   uint end=block+128u;if(end>a.inputDim)end=a.inputDim;
   uint p=block;
   for(;p+16u<end+1u;p+=16u){
    for(uint lane=0u;lane<8u;lane++){
     precise float value=a.mode==0u?x[row*a.inputDim+p+lane]:xt[row*a.inputDim+p+lane];
     lanes[lane]=fma(value,weight[col*a.inputDim+p+lane],lanes[lane]);
     value=a.mode==0u?x[row*a.inputDim+p+8u+lane]:xt[row*a.inputDim+p+8u+lane];
     upper[lane]=fma(value,weight[col*a.inputDim+p+8u+lane],upper[lane]);
    }
   }
   for(uint lane=0u;lane<8u;lane++)lanes[lane]=lanes[lane]+upper[lane];
   if(p+8u<end+1u){
    for(uint lane=0u;lane<8u;lane++){
     precise float value=a.mode==0u?x[row*a.inputDim+p+lane]:xt[row*a.inputDim+p+lane];
     lanes[lane]=fma(value,weight[col*a.inputDim+p+lane],lanes[lane]);
    }
    p+=8u;
   }
   precise float p0=lanes[0]+lanes[4],p1=lanes[1]+lanes[5],p2=lanes[2]+lanes[6],p3=lanes[3]+lanes[7];
   precise float part=(p0+p1)+(p2+p3);
   for(;p<end;p++){
    precise float value=a.mode==0u?x[row*a.inputDim+p]:xt[row*a.inputDim+p];
    part=fma(value,weight[col*a.inputDim+p],part);
   }
   total=total+part;
  }
  if(a.mode==0u)total=total+bias[col];result[i]=total;return;
 }}}
 // Independent FMA lanes avoid the error growth of one serial F32 sum.
 // The balanced final reduction also preserves the CPU GEMV association.
 precise float lo[8],hi[8];
 for(uint lane=0u;lane<8u;lane++){lo[lane]=0.0;hi[lane]=0.0;}
 uint split=0u;
 if(a.inputDim>2048u)split=1u;
 if(a.outputDim>=256u)split=0u;
 // GEMV's remaining <4 output rows use the two-accumulator scalar-row kernel.
 if(col>=a.outputDim-a.outputDim%4u)split=1u;
 uint q=0u;
 for(;q+16u<a.inputDim+1u;q+=16u){
  for(uint lane=0u;lane<8u;lane++){
   precise float v=a.mode==0u?x[row*a.inputDim+q+lane]:xt[row*a.inputDim+q+lane];
   lo[lane]=fma(v,weight[col*a.inputDim+q+lane],lo[lane]);
   v=a.mode==0u?x[row*a.inputDim+q+8u+lane]:xt[row*a.inputDim+q+8u+lane];
   if(split!=0u){hi[lane]=fma(v,weight[col*a.inputDim+q+8u+lane],hi[lane]);}
   else{lo[lane]=fma(v,weight[col*a.inputDim+q+8u+lane],lo[lane]);}
  }
 }
 if(q+8u<a.inputDim+1u){
  for(uint lane=0u;lane<8u;lane++){
   precise float v=a.mode==0u?x[row*a.inputDim+q+lane]:xt[row*a.inputDim+q+lane];
   lo[lane]=fma(v,weight[col*a.inputDim+q+lane],lo[lane]);
  }
  q+=8u;
 }
 for(uint lane=0u;lane<8u;lane++)lo[lane]=lo[lane]+hi[lane];
 precise float p0=lo[0]+lo[4],p1=lo[1]+lo[5],p2=lo[2]+lo[6],p3=lo[3]+lo[7];
 precise float sum=(p0+p1)+(p2+p3);
 // Keep scalar tails after the vector reduction, as on the CPU path.
 for(;q<a.inputDim;q++){
  precise float v=a.mode==0u?x[row*a.inputDim+q]:xt[row*a.inputDim+q];
  sum=fma(v,weight[col*a.inputDim+q],sum);
 }
 if(a.mode==0u)sum=sum+bias[col];result[i]=sum;
}
