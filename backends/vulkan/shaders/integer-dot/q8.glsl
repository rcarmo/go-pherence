#version 450
layout(local_size_x=32) in;
layout(set=0,binding=0) readonly buffer X {float x[];};
layout(set=0,binding=1) buffer Q {uint q[];};
layout(push_constant) uniform P {uint blocks;};
void main(){
 uint b=gl_GlobalInvocationID.x;if(b>=blocks)return;
 float amax=0.0;uint bad=0;for(uint j=0;j<32;j++){float v=x[b*32+j];uint bits=floatBitsToUint(v)&0x7fffffff;if(bits>0x447a0000)bad=1;amax=max(amax,abs(v));}
 uint maximum=floatBitsToUint(amax);if(maximum>0){if(maximum<0x0da24260)bad=1;}
 if(bad!=0){q[b*9]=0x7e007e00;for(uint g=1;g<9;g++)q[b*9+g]=0;return;}
 float d=amax*(1.0/127.0);float inv=0.0;if(amax>0.0)inv=127.0/amax;int sum=0;
 for(uint g=0;g<8;g++){uint packed=0;for(uint j=0;j<4;j++){int v=int(round(x[b*32+g*4+j]*inv));sum+=v;packed|=(uint(v)&255)<<uint(j*8);}q[b*9+1+g]=packed;}
 vec2 ds;ds.x=d;ds.y=float(sum)*d;q[b*9]=packHalf2x16(ds);
}
