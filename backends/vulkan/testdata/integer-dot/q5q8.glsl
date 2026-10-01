#version 450
#extension GL_EXT_integer_dot_product : require
layout(local_size_x=32) in;
layout(set=0,binding=0) readonly buffer Q {uint q[];};
layout(set=0,binding=1) readonly buffer W {uint w[];};
layout(set=0,binding=2) buffer Y {float y[];};
layout(push_constant) uniform P {uint blocks;};
void main(){uint b=gl_GlobalInvocationID.x;if(b>=blocks)return;int isum=0;uint high=w[b*6+1];
 for(uint g=0;g<8;g++){uint packed=0;for(uint j=0;j<4;j++){uint k=g*4+j;uint low=w[b*6+2+(k%16)/4];uint v=((low>>uint((k%4)*8+(k/16)*4))&15)|(((high>>k)&1)<<4);packed|=v<<uint(j*8);}isum+=dotPacked4x8EXT(packed,int(q[b*9+1+g]));}
 vec2 wd;wd=unpackHalf2x16(w[b*6]);vec2 ds;ds=unpackHalf2x16(q[b*9]);float dw=wd.x;float dq=ds.x;float sq=ds.y;
 precise float scaled=float(isum)*dq;precise float corrected=scaled-16.0*sq;y[b]=dw*corrected;
}
