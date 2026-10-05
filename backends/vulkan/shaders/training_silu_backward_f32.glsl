#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer DY { float dy[]; };
layout(set=0,binding=2) writeonly buffer DX { float dx[]; };
layout(push_constant) uniform Args {uint count;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=a.count)return;
 // Stable sigmoid avoids exp overflow for finite negative input. Signed zero
 // is safe on either branch. Exp accuracy is device-qualified separately.
 precise float e=exp(-abs(x[i]));
 precise float s=x[i]>=0.0?1.0/(1.0+e):e/(1.0+e);
 precise float derivative=s*(1.0+x[i]*(1.0-s));
 dx[i]=dy[i]*derivative;
}
