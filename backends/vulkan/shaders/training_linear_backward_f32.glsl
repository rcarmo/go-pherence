#version 450
// One invocation owns one output; separate dispatches for dX/dW/dBias.
// No float atomics and no padding keys/rows are introduced.
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float w[]; };
layout(set=0,binding=2) readonly buffer DY { float dy[]; };
layout(set=0,binding=3) writeonly buffer OUT { float result[]; };
layout(push_constant) uniform Args {uint rows;uint inputDim;uint outputDim;uint mode;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;
 precise float sum=0.0;
 if(a.mode==0u){
  if(i>=a.rows*a.inputDim)return;
  uint row=i/a.inputDim,col=i%a.inputDim;
  for(uint q=0u;q<a.outputDim;q++) sum=fma(dy[row*a.outputDim+q],w[q*a.inputDim+col],sum);
 }else if(a.mode==1u){
  if(i>=a.outputDim*a.inputDim)return;
  uint row=i/a.inputDim,col=i%a.inputDim;
  for(uint q=0u;q<a.rows;q++) sum=fma(dy[q*a.outputDim+row],x[q*a.inputDim+col],sum);
 }else{
  if(i>=a.outputDim)return;
  for(uint q=0u;q<a.rows;q++) sum=sum+dy[q*a.outputDim+i];
 }
 result[i]=sum;
}
