#version 450
// Independent output ownership: scaled sum or read offset with zero padding.
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer A {float av[];};
layout(set=0,binding=1) readonly buffer B {float bv[];};
layout(set=0,binding=2) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint count;uint sourceCount;uint offset;uint mode;float scale;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=a.count)return;
 if(a.mode==0u){precise float sum=av[i]+bv[i]*a.scale;result[i]=sum;}
 else if(a.mode==1u)result[i]=av[a.offset+i];
 else result[i]=(i>=a.offset&&i-a.offset<a.sourceCount)?av[i-a.offset]:0.0;
}
