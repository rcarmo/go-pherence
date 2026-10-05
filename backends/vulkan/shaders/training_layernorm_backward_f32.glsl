#version 450
layout(local_size_x=256) in;
// Xhat is pre-affine normalized input, never the affine output.
layout(set=0,binding=0) readonly buffer X { float xhat[]; };
layout(set=0,binding=1) readonly buffer I { float invstd[]; };
layout(set=0,binding=2) readonly buffer W { float weight[]; };
layout(set=0,binding=3) readonly buffer G { float dy[]; };
layout(set=0,binding=4) writeonly buffer O { float result[]; };
layout(push_constant) uniform Args {uint rows;uint width;uint mode;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;
 if(a.mode==0u){
  // One invocation owns a complete dX row. No shared reduction or barriers.
  if(i>=a.rows)return;
  uint base=i*a.width;precise float sum=0.0;precise float dot=0.0;
  for(uint c=0u;c<a.width;c++){
   precise float g=dy[base+c]*weight[c];sum=sum+g;dot=dot+g*xhat[base+c];
  }
  for(uint c=0u;c<a.width;c++){
   precise float g=dy[base+c]*weight[c];
   precise float v=g-sum/float(a.width)-(xhat[base+c]*dot)/float(a.width);
   result[base+c]=invstd[i]*v;
  }
 }else{
  if(i>=a.width)return;precise float sum=0.0;
  for(uint r=0u;r<a.rows;r++){
   precise float g=dy[r*a.width+i];
   if(a.mode==1u) g=g*xhat[r*a.width+i];
   sum=sum+g;
  }
  result[i]=sum;
 }
}
