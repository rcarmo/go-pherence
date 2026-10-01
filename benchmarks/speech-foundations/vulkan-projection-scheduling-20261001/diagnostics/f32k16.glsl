#version 450
layout(local_size_x=16,local_size_y=16) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float weight[]; };
layout(set=0,binding=2) readonly buffer B { float bias[]; };
layout(set=0,binding=3) buffer O { float output_data[]; };
layout(push_constant) uniform P { uint rows; uint inDim; uint outDim; };
// 64x64 output tile, K16, sixteen accumulators per lane. Same K order.
shared float xt[1024];
shared float wt[1024];
void main() {
 uint cx=gl_LocalInvocationID.x,ry=gl_LocalInvocationID.y,lane=ry*16+cx;
 uint rowBase=gl_WorkGroupID.y*64,colBase=gl_WorkGroupID.x*64;
 float sums[16];for(uint i=0;i<16;i++)sums[i]=0.0;
 for(uint base=0;base<inDim;base+=16) {
  for(uint i=lane;i<1024;i+=256) {
   uint r=i/16,k=base+i%16;float a=0.0,b=0.0;
   if(k<inDim){if(rowBase+r<rows)a=x[(rowBase+r)*inDim+k];if(colBase+r<outDim)b=weight[(colBase+r)*inDim+k];}
   xt[i]=a;wt[i]=b;
  }
  barrier();
  for(uint k=0;k<16;k++) {
   for(uint r=0;r<4;r++) { float a=xt[(ry+r*16)*16+k];
    for(uint c=0;c<4;c++)sums[r*4+c]+=a*wt[(cx+c*16)*16+k];
   }
  }
  barrier();
 }
 for(uint r=0;r<4;r++)for(uint c=0;c<4;c++) {
  uint row=rowBase+ry+r*16,col=colBase+cx+c*16;
  if(row<rows&&col<outDim)output_data[row*outDim+col]=sums[r*4+c]+bias[col];
 }
}
