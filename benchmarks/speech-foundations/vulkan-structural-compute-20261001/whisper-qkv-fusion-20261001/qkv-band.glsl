#version 450
layout(local_size_x=16,local_size_y=16) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer WQ{float wq[];};
layout(set=0,binding=2) readonly buffer WK{float wk[];};
layout(set=0,binding=3) readonly buffer WV{float wv[];};
layout(set=0,binding=4)readonly buffer BQ{float bq[];};
layout(set=0,binding=5)readonly buffer BK{float bk[];};
layout(set=0,binding=6)readonly buffer BV{float bv[];};
layout(set=0,binding=7)buffer OQ{float oq[];};
layout(set=0,binding=8)buffer OK{float ok_data[];};
layout(set=0,binding=9)buffer OV{float ov[];};
layout(push_constant) uniform P { uint rows; uint inDim; uint outDim; };
// 64x64 output tile, K32, sixteen accumulators per lane. Same K order.
shared float xt[2048];
shared float wt[2048];
void main() {
 uint cx=gl_LocalInvocationID.x,ry=gl_LocalInvocationID.y,lane=ry*16+cx;
 uint rowBase=gl_WorkGroupID.y*64,colBase=gl_WorkGroupID.x*64;
 float sums[16];for(uint i=0;i<16;i++)sums[i]=0.0;
 for(uint base=0;base<inDim;base+=32) {
  for(uint i=lane;i<2048;i+=256) {
   uint r=i/32,k=base+i%32;float a=0.0,b=0.0;
   if(k<inDim){if(rowBase+r<rows)a=x[(rowBase+r)*inDim+k];if(colBase+r<outDim*3u){uint col=colBase+r,which=col/outDim,index=(col%outDim)*inDim+k;if(which==0u)b=wq[index];else if(which==1u)b=wk[index];else b=wv[index];}}
   xt[i]=a;wt[i]=b;
  }
  barrier();
  for(uint k=0;k<32;k++) {
   for(uint r=0;r<4;r++) { float a=xt[(ry+r*16)*32+k];
    for(uint c=0;c<4;c++)sums[r*4+c]+=a*wt[(cx+c*16)*32+k];
   }
  }
  barrier();
 }
 for(uint r=0;r<4;r++)for(uint c=0;c<4;c++) {
  uint row=rowBase+ry+r*16,col=colBase+cx+c*16;
  if(row<rows&&col<outDim*3u){uint which=col/outDim,localCol=col%outDim,index=row*outDim+localCol;if(which==0u)oq[index]=sums[r*4+c]+bq[localCol];else if(which==1u)ok_data[index]=sums[r*4+c]+bk[localCol];else ov[index]=sums[r*4+c]+bv[localCol];}
 }
}
