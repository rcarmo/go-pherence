#version 450
#extension GL_EXT_integer_dot_product : require
layout(local_size_x=64) in;
layout(set=0,binding=0)readonly buffer A{uint a[];};
layout(set=0,binding=1)readonly buffer B{int b[];};
layout(set=0,binding=2)buffer C{int c[];};
layout(push_constant)uniform P{uint count;};
void main(){uint i=gl_GlobalInvocationID.x;if(i<count)c[i]=dotPacked4x8EXT(a[i],b[i]);}
