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
shared float xt[2048];
shared float wt[2048];
void main() {
 uint cx=gl_LocalInvocationID.x,ry=gl_LocalInvocationID.y,lane=ry*16+cx;
 uint rowBase=gl_WorkGroupID.y*64,colBase=gl_WorkGroupID.x*64;
 float sums[16];for(uint i=0;i<16;i++)sums[i]=0.0;
 for(uint base=0;base<inDim;base+=32) {
  for(uint i=lane;i<2048;i+=256) {
   uint r=i/32,k=base+i%32;float a=0.0;
   if(k<inDim&&rowBase+r<rows)a=x[(rowBase+r)*inDim+k];
   xt[i]=a;
  }
  // Prepared9words: exactF32scale plus32 signed Q5 values.
  for(uint id=lane;id<256u;id+=256u){uint r=id/4u,chunk=id%4u;float d=0.0;uint offset=0u;if(colBase+r<outDim){offset=((colBase+r)*(inDim/32u)+base/32u)*9u;d=uintBitsToFloat(weight[offset]);}
   {uint index=chunk*8u+0u;uint bits=0;if(colBase+r<outDim)bits=weight[offset+1u+index/4u];uint byteValue=(bits>>((index%4u)*8u))&255u;float q=float(byteValue&127u)-float(byteValue&128u);wt[r*32u+index]=d*q;}
   {uint index=chunk*8u+1u;uint bits=0;if(colBase+r<outDim)bits=weight[offset+1u+index/4u];uint byteValue=(bits>>((index%4u)*8u))&255u;float q=float(byteValue&127u)-float(byteValue&128u);wt[r*32u+index]=d*q;}
   {uint index=chunk*8u+2u;uint bits=0;if(colBase+r<outDim)bits=weight[offset+1u+index/4u];uint byteValue=(bits>>((index%4u)*8u))&255u;float q=float(byteValue&127u)-float(byteValue&128u);wt[r*32u+index]=d*q;}
   {uint index=chunk*8u+3u;uint bits=0;if(colBase+r<outDim)bits=weight[offset+1u+index/4u];uint byteValue=(bits>>((index%4u)*8u))&255u;float q=float(byteValue&127u)-float(byteValue&128u);wt[r*32u+index]=d*q;}
   {uint index=chunk*8u+4u;uint bits=0;if(colBase+r<outDim)bits=weight[offset+1u+index/4u];uint byteValue=(bits>>((index%4u)*8u))&255u;float q=float(byteValue&127u)-float(byteValue&128u);wt[r*32u+index]=d*q;}
   {uint index=chunk*8u+5u;uint bits=0;if(colBase+r<outDim)bits=weight[offset+1u+index/4u];uint byteValue=(bits>>((index%4u)*8u))&255u;float q=float(byteValue&127u)-float(byteValue&128u);wt[r*32u+index]=d*q;}
   {uint index=chunk*8u+6u;uint bits=0;if(colBase+r<outDim)bits=weight[offset+1u+index/4u];uint byteValue=(bits>>((index%4u)*8u))&255u;float q=float(byteValue&127u)-float(byteValue&128u);wt[r*32u+index]=d*q;}
   {uint index=chunk*8u+7u;uint bits=0;if(colBase+r<outDim)bits=weight[offset+1u+index/4u];uint byteValue=(bits>>((index%4u)*8u))&255u;float q=float(byteValue&127u)-float(byteValue&128u);wt[r*32u+index]=d*q;}
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
  if(row<rows&&col<outDim)output_data[row*outDim+col]=sums[r*4+c]+bias[col];
 }
}
