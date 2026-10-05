#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float x[];};
layout(set=0,binding=1) buffer O {float y[];};
layout(push_constant) uniform P {uint count;uint extent;};
void main(){uint i=gl_GlobalInvocationID.x;if(i<extent)y[i]=i<count?x[i]:0.0;}
