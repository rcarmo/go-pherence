#version 450
// One invocation owns a <=256-element stable squared-norm summary.
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float xv[];};
layout(set=0,binding=1) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint count;uint nonnegative;} a;
void main(){
 uint chunk=gl_GlobalInvocationID.x;uint start=chunk*256u;if(start>=a.count)return;
 uint end=start+256u;if(end>a.count)end=a.count;float maximum=0.0;float invalid=0.0;
 for(uint i=start;i<end;i++){float v=xv[i];if(!(abs(v)<=3.402823466e38))invalid=1.0;else{if(a.nonnegative==1u){if(v<0.0)invalid=1.0;}if(abs(v)>maximum)maximum=abs(v);}}
 precise float sum=0.0;precise float correction=0.0;if(maximum>0.0){for(uint i=start;i<end;i++){// A maximum contributes exactly one. Native reciprocal lowering can make
 // x/x round below one; preserve the norm summary's exact lower bound.
 precise float scaled=abs(xv[i])==maximum?1.0:xv[i]/maximum;precise float term=scaled*scaled-correction;precise float next=sum+term;correction=(next-sum)-term;sum=next;}}
 result[chunk*3u]=maximum;result[chunk*3u+1u]=sum;result[chunk*3u+2u]=invalid;
}
