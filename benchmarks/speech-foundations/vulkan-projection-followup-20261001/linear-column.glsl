#version 450
layout(local_size_x=16,local_size_y=16) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float weight[]; };
layout(set=0,binding=2) readonly buffer B { float bias[]; };
layout(set=0,binding=3) buffer O { float output_data[]; };
layout(push_constant) uniform P { uint rows; uint inDim; uint outDim; };
// 64x64 output tile, K32, sixteen accumulators per lane. Same K order.
shared float xt[2048];
shared float wt[2048];
void main() {
 uint cx=gl_LocalInvocationID.x,ry=gl_LocalInvocationID.y,lane=ry*16+cx;
 uint rowBase=gl_WorkGroupID.y*64,colBase=gl_WorkGroupID.x*64;
 float s0=0.0;
 float s1=0.0;
 float s2=0.0;
 float s3=0.0;
 float s4=0.0;
 float s5=0.0;
 float s6=0.0;
 float s7=0.0;
 float s8=0.0;
 float s9=0.0;
 float s10=0.0;
 float s11=0.0;
 float s12=0.0;
 float s13=0.0;
 float s14=0.0;
 float s15=0.0;
 for(uint base=0;base<inDim;base+=32) {
  for(uint i=lane;i<2048;i+=256) {
   uint r=i/32,k=base+i%32;float a=0.0,b=0.0;
   if(k<inDim){if(rowBase+r<rows)a=x[(rowBase+r)*inDim+k];if(colBase+r<outDim)b=weight[(colBase+r)*inDim+k];}
   xt[i]=a;wt[i]=b;
  }
  barrier();
  for(uint k=0;k<32;k++) {
   float a0=xt[(ry+0u)*32+k], b0=wt[(cx+0u)*32+k];
   float a1=xt[(ry+16u)*32+k], b1=wt[(cx+16u)*32+k];
   float a2=xt[(ry+32u)*32+k], b2=wt[(cx+32u)*32+k];
   float a3=xt[(ry+48u)*32+k], b3=wt[(cx+48u)*32+k];
   s0+=a0*b0;
   s4+=a1*b0;
   s8+=a2*b0;
   s12+=a3*b0;
   s1+=a0*b1;
   s5+=a1*b1;
   s9+=a2*b1;
   s13+=a3*b1;
   s2+=a0*b2;
   s6+=a1*b2;
   s10+=a2*b2;
   s14+=a3*b2;
   s3+=a0*b3;
   s7+=a1*b3;
   s11+=a2*b3;
   s15+=a3*b3;
  }
  barrier();
 }
 {uint row=rowBase+ry+0u,col=colBase+cx+0u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s0+bias[col];}
 {uint row=rowBase+ry+0u,col=colBase+cx+16u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s1+bias[col];}
 {uint row=rowBase+ry+0u,col=colBase+cx+32u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s2+bias[col];}
 {uint row=rowBase+ry+0u,col=colBase+cx+48u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s3+bias[col];}
 {uint row=rowBase+ry+16u,col=colBase+cx+0u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s4+bias[col];}
 {uint row=rowBase+ry+16u,col=colBase+cx+16u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s5+bias[col];}
 {uint row=rowBase+ry+16u,col=colBase+cx+32u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s6+bias[col];}
 {uint row=rowBase+ry+16u,col=colBase+cx+48u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s7+bias[col];}
 {uint row=rowBase+ry+32u,col=colBase+cx+0u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s8+bias[col];}
 {uint row=rowBase+ry+32u,col=colBase+cx+16u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s9+bias[col];}
 {uint row=rowBase+ry+32u,col=colBase+cx+32u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s10+bias[col];}
 {uint row=rowBase+ry+32u,col=colBase+cx+48u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s11+bias[col];}
 {uint row=rowBase+ry+48u,col=colBase+cx+0u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s12+bias[col];}
 {uint row=rowBase+ry+48u,col=colBase+cx+16u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s13+bias[col];}
 {uint row=rowBase+ry+48u,col=colBase+cx+32u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s14+bias[col];}
 {uint row=rowBase+ry+48u,col=colBase+cx+48u;if(row<rows&&col<outDim)output_data[row*outDim+col]=s15+bias[col];}
}
