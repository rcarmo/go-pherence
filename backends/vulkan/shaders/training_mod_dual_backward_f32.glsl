#version 450
// kind0: Y=A*(1+scale)+shift; Yt=At*(1+scale)+A*scaleT+shiftT.
// kind1: Y=A*B; Yt=At*B+A*Bt (e.g. AdaLN residual gate*update).
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer A {float av[];};
layout(set=0,binding=1) readonly buffer AT {float at[];};
layout(set=0,binding=2) readonly buffer B {float bv[];};
layout(set=0,binding=3) readonly buffer BT {float bt[];};
layout(set=0,binding=4) readonly buffer G {float gv[];};
layout(set=0,binding=5) readonly buffer H {float gt[];};
layout(set=0,binding=6) writeonly buffer O {float result[];};
// mode0=dAvalue,1=dAtangent,2=dBvalue,3=dBtangent.
// Mod B/BT are planar [shift(n),scale(n)], gradients use the same planar layout.
layout(push_constant) uniform Args {uint count;uint kind;uint mode;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;
 precise float derivative;
 if(a.kind==0u){
  if(a.mode<2u){
   if(i>=a.count)return;precise float scale=1.0+bv[a.count+i];
   if(a.mode==0u)derivative=gv[i]*scale+gt[i]*bt[a.count+i];
   else derivative=gt[i]*scale;
   result[i]=derivative;
  }else{
   if(i>=2u*a.count)return;uint c=i%a.count;
   if(i<a.count)derivative=a.mode==2u?gv[c]:gt[c];
   else if(a.mode==2u)derivative=gv[c]*av[c]+gt[c]*at[c];
   else derivative=gt[c]*av[c];
   result[i]=derivative;
  }
 }else{
  if(i>=a.count)return;
  if(a.mode==0u)derivative=gv[i]*bv[i]+gt[i]*bt[i];
  else if(a.mode==1u)derivative=gt[i]*bv[i];
  else if(a.mode==2u)derivative=gv[i]*av[i]+gt[i]*at[i];
  else derivative=gt[i]*av[i];
  result[i]=derivative;
 }
}
