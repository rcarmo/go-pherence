#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float xv[];};
layout(set=0,binding=1) readonly buffer G {float gv[];};
layout(set=0,binding=2) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint count;uint mode;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=a.count)return;
 if(a.mode==0u)result[i]=xv[i]<0.0?0.0:xv[i];
 else result[i]=xv[i]>0.0?gv[i]:0.0;
}
