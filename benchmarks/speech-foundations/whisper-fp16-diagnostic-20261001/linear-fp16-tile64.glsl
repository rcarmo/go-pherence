#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x=16,local_size_y=16) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { uint weight[]; };
float load_weight(uint i) {
 uint pair=weight[i>>1], value=(pair>>(16*(i&1)))&65535u;
 uint sign=(value&0x8000u)<<16, exponent=(value>>10)&31u, mantissa=value&1023u;
 if(exponent==0u){float subnormal=float(mantissa)*5.9604644775390625e-8;return sign==0u?subnormal:-subnormal;}
 return uintBitsToFloat(sign|((exponent+112u)<<23)|(mantissa<<13));
}
layout(set=0,binding=2) readonly buffer B { float bias[]; };
layout(set=0,binding=3) buffer O { float output_data[]; };
layout(push_constant) uniform P { uint rows; uint inDim; uint outDim; };
// 64x64 output tile, K32, sixteen accumulators per lane. Same K order.
shared float16_t xt[2048];
shared float16_t wt[2048];
void main() {
 uint cx=gl_LocalInvocationID.x,ry=gl_LocalInvocationID.y,lane=ry*16+cx;
 uint rowBase=gl_WorkGroupID.y*64,colBase=gl_WorkGroupID.x*64;
 float sums[16];for(uint i=0;i<16;i++)sums[i]=0.0;
 for(uint base=0;base<inDim;base+=32) {
  for(uint i=lane;i<2048;i+=256) {
   uint r=i/32,k=base+i%32;float a=0.0,b=0.0;
   if(k<inDim){if(rowBase+r<rows)a=x[(rowBase+r)*inDim+k];if(colBase+r<outDim)b=load_weight((colBase+r)*inDim+k);}
   xt[i]=float16_t(a);wt[i]=float16_t(b);
  }
  barrier();
  for(uint k=0;k<32;k++) {
   for(uint r=0;r<4;r++) { float a=float(xt[(ry+r*16)*32+k]);
    for(uint c=0;c<4;c++)sums[r*4+c]+=a*float(wt[(cx+c*16)*32+k]);
   }
  }
  barrier();
 }
 for(uint r=0;r<4;r++)for(uint c=0;c<4;c++) {
  uint row=rowBase+ry+r*16,col=colBase+cx+c*16;
  if(row<rows&&col<outDim)output_data[row*outDim+col]=sums[r*4+c]+bias[col];
 }
}
