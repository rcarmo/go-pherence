#version 450
layout(local_size_x=16,local_size_y=16)in;
layout(set=0,binding=0)readonly buffer X{float x[];};
layout(set=0,binding=1)readonly buffer WQ{float wq[];};
layout(set=0,binding=2)readonly buffer WK{float wk[];};
layout(set=0,binding=3)readonly buffer WV{float wv[];};
layout(set=0,binding=4)readonly buffer BQ{float bq[];};
layout(set=0,binding=5)readonly buffer BK{float bk[];};
layout(set=0,binding=6)readonly buffer BV{float bv[];};
layout(set=0,binding=7)buffer OQ{float oq[];};
layout(set=0,binding=8)buffer OK{float ok_data[];};
layout(set=0,binding=9)buffer OV{float ov[];};
layout(push_constant)uniform P{uint rows;uint inDim;uint outDim;};
// Simultaneous ordered Q/K/V sums; one X tile serves three projections.
shared float xt[2048];shared float qt[2048];shared float kt[2048];shared float vt[2048];
void main(){
 uint cx=gl_LocalInvocationID.x,ry=gl_LocalInvocationID.y,lane=ry*16+cx;
 uint rowBase=gl_WorkGroupID.y*64,colBase=gl_WorkGroupID.x*64;
 float qs[16],ks[16],vs[16];for(uint i=0;i<16;i++){qs[i]=0.0;ks[i]=0.0;vs[i]=0.0;}
 for(uint base=0;base<inDim;base+=32){
  for(uint i=lane;i<2048;i+=256){
   uint r=i/32,k=base+i%32;float a=0.0,q=0.0,b=0.0,v=0.0;
   if(k<inDim){if(rowBase+r<rows)a=x[(rowBase+r)*inDim+k];if(colBase+r<outDim){uint offset=(colBase+r)*inDim+k;q=wq[offset];b=wk[offset];v=wv[offset];}}
   xt[i]=a;qt[i]=q;kt[i]=b;vt[i]=v;
  }
  barrier();
  for(uint k=0;k<32;k++){for(uint r=0;r<4;r++){float a=xt[(ry+r*16)*32+k];for(uint c=0;c<4;c++){
   uint i=r*4+c,offset=(cx+c*16)*32+k;qs[i]+=a*qt[offset];ks[i]+=a*kt[offset];vs[i]+=a*vt[offset];
  }}}
  barrier();
 }
 for(uint r=0;r<4;r++)for(uint c=0;c<4;c++){
  uint row=rowBase+ry+r*16,col=colBase+cx+c*16,i=r*4+c;
  if(row<rows&&col<outDim){uint offset=row*outDim+col;oq[offset]=qs[i]+bq[col];ok_data[offset]=ks[i]+bk[col];ov[offset]=vs[i]+bv[col];}
 }
}
