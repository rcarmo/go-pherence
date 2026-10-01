#version 450
layout(local_size_x=16,local_size_y=16) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { uint weight[]; };

float half_value(uint value){
 uint sign=(value&32768u)<<16, exponent=(value>>10)&31u, mantissa=value&1023u;
 if(exponent==0u){float subnormal=float(mantissa)*5.9604644775390625e-8;return sign==0u?subnormal:-subnormal;}
 return uintBitsToFloat(sign|((exponent+112u)<<23)|(mantissa<<13));
}
uint read32(uint offset){uint i=offset>>2,s=(offset&3u)*8u;uint a=weight[i];if(s==0u)return a;return (a>>s)|(weight[i+1]<<(32u-s));}
layout(set=0,binding=2) readonly buffer B { float bias[]; };
layout(set=0,binding=3) buffer O { float output_data[]; };
layout(push_constant) uniform P { uint rows; uint inDim; uint outDim; };
// 64x64 output tile, K32, sixteen accumulators per lane. Same K order.
shared float xt[4096];
shared float wt[1024];
void main() {
 uint cx=gl_LocalInvocationID.x,ry=gl_LocalInvocationID.y,lane=ry*16+cx;
 uint rowBase=gl_WorkGroupID.y*128,colBase=gl_WorkGroupID.x*32;
 precise float a0=0.0;precise float a1=0.0;precise float a2=0.0;precise float a3=0.0;precise float a4=0.0;precise float a5=0.0;precise float a6=0.0;precise float a7=0.0;precise float a8=0.0;precise float a9=0.0;precise float a10=0.0;precise float a11=0.0;precise float a12=0.0;precise float a13=0.0;precise float a14=0.0;precise float a15=0.0;
 for(uint base=0;base<inDim;base+=32) {
  for(uint i=lane;i<4096;i+=256) {
   uint r=i/32,k=base+i%32;float a=0.0;
   if(k<inDim&&rowBase+r<rows)a=x[(rowBase+r)*inDim+k];
   xt[i]=a;
  }
  // 8 lanes/block; explicit independent decoded stores.
  for(uint id=lane;id<256u;id+=256u){
   uint r=id/8u,chunk=id%8u,j=chunk*2u;
   float d=0.0;uint high=0u,quants=0u;
   if(colBase+r<outDim){
    uint b=(colBase+r)*(inDim/32u)+base/32u,offset=b*6u;
    d=half_value(weight[offset]&65535u);high=weight[offset+1u];quants=weight[offset+2u+j/4u];
   }
   {uint index=j+0u,packed=(quants>>((j%4u+0u)*8u))&255u;
    float lo=float((packed&15u)|(((high>>index)&1u)<<4u))-16.0;
    float hi=float((packed>>4u)|(((high>>(index+16u))&1u)<<4u))-16.0;
    wt[r*32u+index]=d*lo;wt[r*32u+index+16u]=d*hi;
   }
   {uint index=j+1u,packed=(quants>>((j%4u+1u)*8u))&255u;
    float lo=float((packed&15u)|(((high>>index)&1u)<<4u))-16.0;
    float hi=float((packed>>4u)|(((high>>(index+16u))&1u)<<4u))-16.0;
    wt[r*32u+index]=d*lo;wt[r*32u+index+16u]=d*hi;
   }
  }
  barrier();
  for(uint channel=0;channel<32;channel+=4){
  {uint k=channel+0;float x0=xt[(ry+0)*32+k];float x1=xt[(ry+16)*32+k];float x2=xt[(ry+32)*32+k];float x3=xt[(ry+48)*32+k];float x4=xt[(ry+64)*32+k];float x5=xt[(ry+80)*32+k];float x6=xt[(ry+96)*32+k];float x7=xt[(ry+112)*32+k];float w0=wt[(cx+0)*32+k];float w1=wt[(cx+16)*32+k];a0=fma(x0,w0,a0);a1=fma(x0,w1,a1);a2=fma(x1,w0,a2);a3=fma(x1,w1,a3);a4=fma(x2,w0,a4);a5=fma(x2,w1,a5);a6=fma(x3,w0,a6);a7=fma(x3,w1,a7);a8=fma(x4,w0,a8);a9=fma(x4,w1,a9);a10=fma(x5,w0,a10);a11=fma(x5,w1,a11);a12=fma(x6,w0,a12);a13=fma(x6,w1,a13);a14=fma(x7,w0,a14);a15=fma(x7,w1,a15);}
  {uint k=channel+1;float x0=xt[(ry+0)*32+k];float x1=xt[(ry+16)*32+k];float x2=xt[(ry+32)*32+k];float x3=xt[(ry+48)*32+k];float x4=xt[(ry+64)*32+k];float x5=xt[(ry+80)*32+k];float x6=xt[(ry+96)*32+k];float x7=xt[(ry+112)*32+k];float w0=wt[(cx+0)*32+k];float w1=wt[(cx+16)*32+k];a0=fma(x0,w0,a0);a1=fma(x0,w1,a1);a2=fma(x1,w0,a2);a3=fma(x1,w1,a3);a4=fma(x2,w0,a4);a5=fma(x2,w1,a5);a6=fma(x3,w0,a6);a7=fma(x3,w1,a7);a8=fma(x4,w0,a8);a9=fma(x4,w1,a9);a10=fma(x5,w0,a10);a11=fma(x5,w1,a11);a12=fma(x6,w0,a12);a13=fma(x6,w1,a13);a14=fma(x7,w0,a14);a15=fma(x7,w1,a15);}
  {uint k=channel+2;float x0=xt[(ry+0)*32+k];float x1=xt[(ry+16)*32+k];float x2=xt[(ry+32)*32+k];float x3=xt[(ry+48)*32+k];float x4=xt[(ry+64)*32+k];float x5=xt[(ry+80)*32+k];float x6=xt[(ry+96)*32+k];float x7=xt[(ry+112)*32+k];float w0=wt[(cx+0)*32+k];float w1=wt[(cx+16)*32+k];a0=fma(x0,w0,a0);a1=fma(x0,w1,a1);a2=fma(x1,w0,a2);a3=fma(x1,w1,a3);a4=fma(x2,w0,a4);a5=fma(x2,w1,a5);a6=fma(x3,w0,a6);a7=fma(x3,w1,a7);a8=fma(x4,w0,a8);a9=fma(x4,w1,a9);a10=fma(x5,w0,a10);a11=fma(x5,w1,a11);a12=fma(x6,w0,a12);a13=fma(x6,w1,a13);a14=fma(x7,w0,a14);a15=fma(x7,w1,a15);}
  {uint k=channel+3;float x0=xt[(ry+0)*32+k];float x1=xt[(ry+16)*32+k];float x2=xt[(ry+32)*32+k];float x3=xt[(ry+48)*32+k];float x4=xt[(ry+64)*32+k];float x5=xt[(ry+80)*32+k];float x6=xt[(ry+96)*32+k];float x7=xt[(ry+112)*32+k];float w0=wt[(cx+0)*32+k];float w1=wt[(cx+16)*32+k];a0=fma(x0,w0,a0);a1=fma(x0,w1,a1);a2=fma(x1,w0,a2);a3=fma(x1,w1,a3);a4=fma(x2,w0,a4);a5=fma(x2,w1,a5);a6=fma(x3,w0,a6);a7=fma(x3,w1,a7);a8=fma(x4,w0,a8);a9=fma(x4,w1,a9);a10=fma(x5,w0,a10);a11=fma(x5,w1,a11);a12=fma(x6,w0,a12);a13=fma(x6,w1,a13);a14=fma(x7,w0,a14);a15=fma(x7,w1,a15);}
  }
  barrier();
 }
 {uint row=rowBase+ry+0,col=colBase+cx+0;if(row<rows&&col<outDim)output_data[row*outDim+col]=a0+bias[col];}
 {uint row=rowBase+ry+0,col=colBase+cx+16;if(row<rows&&col<outDim)output_data[row*outDim+col]=a1+bias[col];}
 {uint row=rowBase+ry+16,col=colBase+cx+0;if(row<rows&&col<outDim)output_data[row*outDim+col]=a2+bias[col];}
 {uint row=rowBase+ry+16,col=colBase+cx+16;if(row<rows&&col<outDim)output_data[row*outDim+col]=a3+bias[col];}
 {uint row=rowBase+ry+32,col=colBase+cx+0;if(row<rows&&col<outDim)output_data[row*outDim+col]=a4+bias[col];}
 {uint row=rowBase+ry+32,col=colBase+cx+16;if(row<rows&&col<outDim)output_data[row*outDim+col]=a5+bias[col];}
 {uint row=rowBase+ry+48,col=colBase+cx+0;if(row<rows&&col<outDim)output_data[row*outDim+col]=a6+bias[col];}
 {uint row=rowBase+ry+48,col=colBase+cx+16;if(row<rows&&col<outDim)output_data[row*outDim+col]=a7+bias[col];}
 {uint row=rowBase+ry+64,col=colBase+cx+0;if(row<rows&&col<outDim)output_data[row*outDim+col]=a8+bias[col];}
 {uint row=rowBase+ry+64,col=colBase+cx+16;if(row<rows&&col<outDim)output_data[row*outDim+col]=a9+bias[col];}
 {uint row=rowBase+ry+80,col=colBase+cx+0;if(row<rows&&col<outDim)output_data[row*outDim+col]=a10+bias[col];}
 {uint row=rowBase+ry+80,col=colBase+cx+16;if(row<rows&&col<outDim)output_data[row*outDim+col]=a11+bias[col];}
 {uint row=rowBase+ry+96,col=colBase+cx+0;if(row<rows&&col<outDim)output_data[row*outDim+col]=a12+bias[col];}
 {uint row=rowBase+ry+96,col=colBase+cx+16;if(row<rows&&col<outDim)output_data[row*outDim+col]=a13+bias[col];}
 {uint row=rowBase+ry+112,col=colBase+cx+0;if(row<rows&&col<outDim)output_data[row*outDim+col]=a14+bias[col];}
 {uint row=rowBase+ry+112,col=colBase+cx+16;if(row<rows&&col<outDim)output_data[row*outDim+col]=a15+bias[col];}
}
