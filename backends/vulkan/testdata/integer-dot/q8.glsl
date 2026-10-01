#version 450
layout(local_size_x=32) in;
layout(set=0,binding=0) readonly buffer X {float x[];};
layout(set=0,binding=1) buffer Q {uint q[];};
layout(push_constant) uniform P {uint blocks;};
void main(){
 uint b=gl_GlobalInvocationID.x;if(b>=blocks)return;
 float amax=0.0;for(uint j=0;j<32;j++)amax=max(amax,abs(x[b*32+j]));
 float d=amax/127.0;float inv=d<=0.0?0.0:1.0/d;int sum=0;
 for(uint g=0;g<8;g++){uint packed=0;for(uint j=0;j<4;j++){int v=int(round(x[b*32+g*4+j]*inv));sum+=v;packed|=(uint(v)&255)<<uint(j*8);}q[b*9+1+g]=packed;}
 vec2 ds;ds.x=d;ds.y=float(sum)*d;q[b*9]=packHalf2x16(ds);
}
