#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float x[];};
layout(set=0,binding=1) readonly buffer T {float xt[];};
layout(set=0,binding=2) readonly buffer W {float weight[];};
layout(set=0,binding=3) readonly buffer G {float gv[];};
layout(set=0,binding=4) readonly buffer H {float gt[];};
layout(set=0,binding=5) writeonly buffer O {float result[];};
// mode0=dXvalue,1=dXtangent,2=dWeight,3=dBias. Bias tangent is zero.
layout(push_constant) uniform Args {uint rows;uint inputDim;uint outputDim;uint mode;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;precise float sum=0.0;
 if(a.mode<2u){
  if(i>=a.rows*a.inputDim)return;uint row=i/a.inputDim,col=i%a.inputDim;
  for(uint q=0u;q<a.outputDim;q++){
   precise float g=a.mode==0u?gv[row*a.outputDim+q]:gt[row*a.outputDim+q];
   sum=fma(g,weight[q*a.inputDim+col],sum);
  }
 }else if(a.mode==2u){
  if(i>=a.outputDim*a.inputDim)return;uint row=i/a.inputDim,col=i%a.inputDim;
  // Preserve CPU reverse-over-forward order: all primal contributions first,
  // then all tangent contributions, never manufacture a tangent bias.
  for(uint q=0u;q<a.rows;q++)sum=fma(gv[q*a.outputDim+row],x[q*a.inputDim+col],sum);
  for(uint q=0u;q<a.rows;q++)sum=fma(gt[q*a.outputDim+row],xt[q*a.inputDim+col],sum);
 }else{
  if(i>=a.outputDim)return;for(uint q=0u;q<a.rows;q++)sum=sum+gv[q*a.outputDim+i];
 }
 result[i]=sum;
}
