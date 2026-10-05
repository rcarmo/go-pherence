#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer A {float av[];};
layout(set=0,binding=1) readonly buffer AT {float at[];};
layout(set=0,binding=2) readonly buffer B {float bv[];};
layout(set=0,binding=3) readonly buffer BT {float bt[];};
layout(set=0,binding=4) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint count;uint kind;uint mode;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=a.count)return;
 if(a.kind==0u){
  precise float scale=1.0+bv[a.count+i];
  if(a.mode==0u){precise float value=av[i]*scale+bv[i];result[i]=value;}
  else{precise float tangent=(at[i]*scale+av[i]*bt[a.count+i])+bt[i];result[i]=tangent;}
 }else{
  if(a.mode==0u){precise float value=av[i]*bv[i];result[i]=value;}
  else{precise float tangent=at[i]*bv[i]+av[i]*bt[i];result[i]=tangent;}
 }
}
