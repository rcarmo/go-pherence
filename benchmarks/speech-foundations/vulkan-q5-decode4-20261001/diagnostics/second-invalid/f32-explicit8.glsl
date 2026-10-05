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
 float a0=0.0;float a1=0.0;float a2=0.0;float a3=0.0;float a4=0.0;float a5=0.0;float a6=0.0;float a7=0.0;float a8=0.0;float a9=0.0;float a10=0.0;float a11=0.0;float a12=0.0;float a13=0.0;float a14=0.0;float a15=0.0;
 for(uint base=0;base<inDim;base+=32) {
  for(uint i=lane;i<2048;i+=256) {
   uint r=i/32,k=base+i%32;float a=0.0,b=0.0;
   if(k<inDim){if(rowBase+r<rows)a=x[(rowBase+r)*inDim+k];if(colBase+r<outDim)b=weight[(colBase+r)*inDim+k];}
   xt[i]=a;wt[i]=b;
  }
  barrier();
  for(uint channel=0;channel<32;channel+=8){
  {uint k=channel+0;float x0=xt[(ry+0)*32+k];float x1=xt[(ry+16)*32+k];float x2=xt[(ry+32)*32+k];float x3=xt[(ry+48)*32+k];float w0=wt[(cx+0)*32+k];float w1=wt[(cx+16)*32+k];float w2=wt[(cx+32)*32+k];float w3=wt[(cx+48)*32+k];a0+=x0*w0;a1+=x0*w1;a2+=x0*w2;a3+=x0*w3;a4+=x1*w0;a5+=x1*w1;a6+=x1*w2;a7+=x1*w3;a8+=x2*w0;a9+=x2*w1;a10+=x2*w2;a11+=x2*w3;a12+=x3*w0;a13+=x3*w1;a14+=x3*w2;a15+=x3*w3;}
  {uint k=channel+1;float x0=xt[(ry+0)*32+k];float x1=xt[(ry+16)*32+k];float x2=xt[(ry+32)*32+k];float x3=xt[(ry+48)*32+k];float w0=wt[(cx+0)*32+k];float w1=wt[(cx+16)*32+k];float w2=wt[(cx+32)*32+k];float w3=wt[(cx+48)*32+k];a0+=x0*w0;a1+=x0*w1;a2+=x0*w2;a3+=x0*w3;a4+=x1*w0;a5+=x1*w1;a6+=x1*w2;a7+=x1*w3;a8+=x2*w0;a9+=x2*w1;a10+=x2*w2;a11+=x2*w3;a12+=x3*w0;a13+=x3*w1;a14+=x3*w2;a15+=x3*w3;}
  {uint k=channel+2;float x0=xt[(ry+0)*32+k];float x1=xt[(ry+16)*32+k];float x2=xt[(ry+32)*32+k];float x3=xt[(ry+48)*32+k];float w0=wt[(cx+0)*32+k];float w1=wt[(cx+16)*32+k];float w2=wt[(cx+32)*32+k];float w3=wt[(cx+48)*32+k];a0+=x0*w0;a1+=x0*w1;a2+=x0*w2;a3+=x0*w3;a4+=x1*w0;a5+=x1*w1;a6+=x1*w2;a7+=x1*w3;a8+=x2*w0;a9+=x2*w1;a10+=x2*w2;a11+=x2*w3;a12+=x3*w0;a13+=x3*w1;a14+=x3*w2;a15+=x3*w3;}
  {uint k=channel+3;float x0=xt[(ry+0)*32+k];float x1=xt[(ry+16)*32+k];float x2=xt[(ry+32)*32+k];float x3=xt[(ry+48)*32+k];float w0=wt[(cx+0)*32+k];float w1=wt[(cx+16)*32+k];float w2=wt[(cx+32)*32+k];float w3=wt[(cx+48)*32+k];a0+=x0*w0;a1+=x0*w1;a2+=x0*w2;a3+=x0*w3;a4+=x1*w0;a5+=x1*w1;a6+=x1*w2;a7+=x1*w3;a8+=x2*w0;a9+=x2*w1;a10+=x2*w2;a11+=x2*w3;a12+=x3*w0;a13+=x3*w1;a14+=x3*w2;a15+=x3*w3;}
  {uint k=channel+4;float x0=xt[(ry+0)*32+k];float x1=xt[(ry+16)*32+k];float x2=xt[(ry+32)*32+k];float x3=xt[(ry+48)*32+k];float w0=wt[(cx+0)*32+k];float w1=wt[(cx+16)*32+k];float w2=wt[(cx+32)*32+k];float w3=wt[(cx+48)*32+k];a0+=x0*w0;a1+=x0*w1;a2+=x0*w2;a3+=x0*w3;a4+=x1*w0;a5+=x1*w1;a6+=x1*w2;a7+=x1*w3;a8+=x2*w0;a9+=x2*w1;a10+=x2*w2;a11+=x2*w3;a12+=x3*w0;a13+=x3*w1;a14+=x3*w2;a15+=x3*w3;}
  {uint k=channel+5;float x0=xt[(ry+0)*32+k];float x1=xt[(ry+16)*32+k];float x2=xt[(ry+32)*32+k];float x3=xt[(ry+48)*32+k];float w0=wt[(cx+0)*32+k];float w1=wt[(cx+16)*32+k];float w2=wt[(cx+32)*32+k];float w3=wt[(cx+48)*32+k];a0+=x0*w0;a1+=x0*w1;a2+=x0*w2;a3+=x0*w3;a4+=x1*w0;a5+=x1*w1;a6+=x1*w2;a7+=x1*w3;a8+=x2*w0;a9+=x2*w1;a10+=x2*w2;a11+=x2*w3;a12+=x3*w0;a13+=x3*w1;a14+=x3*w2;a15+=x3*w3;}
  {uint k=channel+6;float x0=xt[(ry+0)*32+k];float x1=xt[(ry+16)*32+k];float x2=xt[(ry+32)*32+k];float x3=xt[(ry+48)*32+k];float w0=wt[(cx+0)*32+k];float w1=wt[(cx+16)*32+k];float w2=wt[(cx+32)*32+k];float w3=wt[(cx+48)*32+k];a0+=x0*w0;a1+=x0*w1;a2+=x0*w2;a3+=x0*w3;a4+=x1*w0;a5+=x1*w1;a6+=x1*w2;a7+=x1*w3;a8+=x2*w0;a9+=x2*w1;a10+=x2*w2;a11+=x2*w3;a12+=x3*w0;a13+=x3*w1;a14+=x3*w2;a15+=x3*w3;}
  {uint k=channel+7;float x0=xt[(ry+0)*32+k];float x1=xt[(ry+16)*32+k];float x2=xt[(ry+32)*32+k];float x3=xt[(ry+48)*32+k];float w0=wt[(cx+0)*32+k];float w1=wt[(cx+16)*32+k];float w2=wt[(cx+32)*32+k];float w3=wt[(cx+48)*32+k];a0+=x0*w0;a1+=x0*w1;a2+=x0*w2;a3+=x0*w3;a4+=x1*w0;a5+=x1*w1;a6+=x1*w2;a7+=x1*w3;a8+=x2*w0;a9+=x2*w1;a10+=x2*w2;a11+=x2*w3;a12+=x3*w0;a13+=x3*w1;a14+=x3*w2;a15+=x3*w3;}
  }
  barrier();
 }
 {uint row=rowBase+ry+0,col=colBase+cx+0;if(row<rows&&col<outDim)output_data[row*outDim+col]=a0+bias[col];}
 {uint row=rowBase+ry+0,col=colBase+cx+16;if(row<rows&&col<outDim)output_data[row*outDim+col]=a1+bias[col];}
 {uint row=rowBase+ry+0,col=colBase+cx+32;if(row<rows&&col<outDim)output_data[row*outDim+col]=a2+bias[col];}
 {uint row=rowBase+ry+0,col=colBase+cx+48;if(row<rows&&col<outDim)output_data[row*outDim+col]=a3+bias[col];}
 {uint row=rowBase+ry+16,col=colBase+cx+0;if(row<rows&&col<outDim)output_data[row*outDim+col]=a4+bias[col];}
 {uint row=rowBase+ry+16,col=colBase+cx+16;if(row<rows&&col<outDim)output_data[row*outDim+col]=a5+bias[col];}
 {uint row=rowBase+ry+16,col=colBase+cx+32;if(row<rows&&col<outDim)output_data[row*outDim+col]=a6+bias[col];}
 {uint row=rowBase+ry+16,col=colBase+cx+48;if(row<rows&&col<outDim)output_data[row*outDim+col]=a7+bias[col];}
 {uint row=rowBase+ry+32,col=colBase+cx+0;if(row<rows&&col<outDim)output_data[row*outDim+col]=a8+bias[col];}
 {uint row=rowBase+ry+32,col=colBase+cx+16;if(row<rows&&col<outDim)output_data[row*outDim+col]=a9+bias[col];}
 {uint row=rowBase+ry+32,col=colBase+cx+32;if(row<rows&&col<outDim)output_data[row*outDim+col]=a10+bias[col];}
 {uint row=rowBase+ry+32,col=colBase+cx+48;if(row<rows&&col<outDim)output_data[row*outDim+col]=a11+bias[col];}
 {uint row=rowBase+ry+48,col=colBase+cx+0;if(row<rows&&col<outDim)output_data[row*outDim+col]=a12+bias[col];}
 {uint row=rowBase+ry+48,col=colBase+cx+16;if(row<rows&&col<outDim)output_data[row*outDim+col]=a13+bias[col];}
 {uint row=rowBase+ry+48,col=colBase+cx+32;if(row<rows&&col<outDim)output_data[row*outDim+col]=a14+bias[col];}
 {uint row=rowBase+ry+48,col=colBase+cx+48;if(row<rows&&col<outDim)output_data[row*outDim+col]=a15+bias[col];}
}
