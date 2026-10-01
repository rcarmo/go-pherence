#version 450
#extension GL_EXT_integer_dot_product : require
layout(local_size_x=16,local_size_y=16) in;
layout(set=0,binding=0) readonly buffer X {uint x[];};
layout(set=0,binding=1) readonly buffer W {uint w[];};
layout(set=0,binding=2) readonly buffer B {float bias[];};
layout(set=0,binding=3) buffer Y {float y[];};
layout(push_constant) uniform P {uint M;uint K;uint N;};
shared uint xs[576];shared uint ws[576];
float corrected(int dot,vec2 q){precise float a=float(dot)*q.x;precise float b=a-16.0*q.y;return b;}
float blockAdd(float scale,int dot,vec2 q,float accum){precise float product=scale*corrected(dot,q);precise float sum=accum+product;return sum;}
void main(){uint lx=gl_LocalInvocationID.x,ly=gl_LocalInvocationID.y;uint row=gl_WorkGroupID.y*64+ly*4,col=gl_WorkGroupID.x*64+lx*4,tid=ly*16+lx,nb=K/32;
 float a00=0,a01=0,a02=0,a03=0,a10=0,a11=0,a12=0,a13=0,a20=0,a21=0,a22=0,a23=0,a30=0,a31=0,a32=0,a33=0;
 for(uint block=0;block<nb;block++){
  for(uint item=tid;item<64*9;item+=256){uint local=item/9,g=item%9;uint r=gl_WorkGroupID.y*64+local;xs[item]=r<M?x[(r*nb+block)*9+g]:0;uint c=gl_WorkGroupID.x*64+local;uint value=0;if(c<N){uint base=(c*nb+block)*6;if(g==0)value=w[base];else{uint high=w[base+1];for(uint j=0;j<4;j++){uint k=(g-1)*4+j;uint low=w[base+2+(k%16)/4];uint v=((low>>uint((k%4)*8+(k/16)*4))&15)|(((high>>k)&1)<<4);value|=v<<uint(j*8);}}}ws[item]=value;}
  barrier();
  int s00=0,s01=0,s02=0,s03=0,s10=0,s11=0,s12=0,s13=0,s20=0,s21=0,s22=0,s23=0,s30=0,s31=0,s32=0,s33=0;
  for(uint g=1;g<9;g++){int q0=int(xs[(ly*4)*9+g]),q1=int(xs[(ly*4+1)*9+g]),q2=int(xs[(ly*4+2)*9+g]),q3=int(xs[(ly*4+3)*9+g]);uint w0=ws[(lx*4)*9+g],w1=ws[(lx*4+1)*9+g],w2=ws[(lx*4+2)*9+g],w3=ws[(lx*4+3)*9+g];
 s00+=dotPacked4x8EXT(w0,q0);s01+=dotPacked4x8EXT(w1,q0);s02+=dotPacked4x8EXT(w2,q0);s03+=dotPacked4x8EXT(w3,q0);
 s10+=dotPacked4x8EXT(w0,q1);s11+=dotPacked4x8EXT(w1,q1);s12+=dotPacked4x8EXT(w2,q1);s13+=dotPacked4x8EXT(w3,q1);
 s20+=dotPacked4x8EXT(w0,q2);s21+=dotPacked4x8EXT(w1,q2);s22+=dotPacked4x8EXT(w2,q2);s23+=dotPacked4x8EXT(w3,q2);
 s30+=dotPacked4x8EXT(w0,q3);s31+=dotPacked4x8EXT(w1,q3);s32+=dotPacked4x8EXT(w2,q3);s33+=dotPacked4x8EXT(w3,q3);}
 vec2 q0,q1,q2,q3,d0,d1,d2,d3;q0=unpackHalf2x16(xs[(ly*4)*9]);q1=unpackHalf2x16(xs[(ly*4+1)*9]);q2=unpackHalf2x16(xs[(ly*4+2)*9]);q3=unpackHalf2x16(xs[(ly*4+3)*9]);d0=unpackHalf2x16(ws[(lx*4)*9]);d1=unpackHalf2x16(ws[(lx*4+1)*9]);d2=unpackHalf2x16(ws[(lx*4+2)*9]);d3=unpackHalf2x16(ws[(lx*4+3)*9]);
 // Explicit F32 block correction and FMA across blocks; activation scales/sums
 // have already been rounded to F16. This is not the ordered F32 oracle.
 a00=blockAdd(d0.x,s00,q0,a00);a01=blockAdd(d1.x,s01,q0,a01);a02=blockAdd(d2.x,s02,q0,a02);a03=blockAdd(d3.x,s03,q0,a03);
 a10=blockAdd(d0.x,s10,q1,a10);a11=blockAdd(d1.x,s11,q1,a11);a12=blockAdd(d2.x,s12,q1,a12);a13=blockAdd(d3.x,s13,q1,a13);
 a20=blockAdd(d0.x,s20,q2,a20);a21=blockAdd(d1.x,s21,q2,a21);a22=blockAdd(d2.x,s22,q2,a22);a23=blockAdd(d3.x,s23,q2,a23);
 a30=blockAdd(d0.x,s30,q3,a30);a31=blockAdd(d1.x,s31,q3,a31);a32=blockAdd(d2.x,s32,q3,a32);a33=blockAdd(d3.x,s33,q3,a33);barrier();}
 if(row<M){if(col<N)y[row*N+col]=a00+bias[col];if(col+1<N)y[row*N+col+1]=a01+bias[col+1];if(col+2<N)y[row*N+col+2]=a02+bias[col+2];if(col+3<N)y[row*N+col+3]=a03+bias[col+3];}
 if(row+1<M){if(col<N)y[(row+1)*N+col]=a10+bias[col];if(col+1<N)y[(row+1)*N+col+1]=a11+bias[col+1];if(col+2<N)y[(row+1)*N+col+2]=a12+bias[col+2];if(col+3<N)y[(row+1)*N+col+3]=a13+bias[col+3];}
 if(row+2<M){if(col<N)y[(row+2)*N+col]=a20+bias[col];if(col+1<N)y[(row+2)*N+col+1]=a21+bias[col+1];if(col+2<N)y[(row+2)*N+col+2]=a22+bias[col+2];if(col+3<N)y[(row+2)*N+col+3]=a23+bias[col+3];}
 if(row+3<M){if(col<N)y[(row+3)*N+col]=a30+bias[col];if(col+1<N)y[(row+3)*N+col+1]=a31+bias[col+1];if(col+2<N)y[(row+3)*N+col+2]=a32+bias[col+2];if(col+3<N)y[(row+3)*N+col+3]=a33+bias[col+3];}}
