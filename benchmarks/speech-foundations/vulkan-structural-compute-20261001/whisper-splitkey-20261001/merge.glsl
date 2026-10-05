#version 450
layout(local_size_x=16,local_size_y=16)in;
layout(set=0,binding=0)readonly buffer Partials{float partials[];};
layout(set=0,binding=1)readonly buffer MaxSum{float maxsum[];};
layout(set=0,binding=2)buffer Out{float output_data[];};
layout(push_constant)uniform P{uint seqQ;uint heads;uint headDim;uint splitCount;};
void main(){
 uint d=gl_LocalInvocationID.x,row=gl_WorkGroupID.x*16+gl_LocalInvocationID.y,head=gl_WorkGroupID.y,width=heads*headDim;
 if(row<seqQ){
  float m=0.0;uint seen=0u;
  for(uint s=0;s<splitCount;s++){uint idx=((s*seqQ+row)*heads+head)*2u;if(maxsum[idx+1u]>0.0){float v=maxsum[idx];if(seen==0u)m=v;else if(v>m)m=v;seen=1u;}}
  precise float denominator=0.0;
  for(uint s=0;s<splitCount;s++){uint idx=((s*seqQ+row)*heads+head)*2u;if(maxsum[idx+1u]>0.0)denominator+=maxsum[idx+1u]*exp(maxsum[idx]-m);}
  for(uint i=0;i<4;i++){uint c=d+i*16;if(c<headDim){precise float value=0.0;for(uint s=0;s<splitCount;s++){uint idx=((s*seqQ+row)*heads+head)*2u;if(maxsum[idx+1u]>0.0)value=fma(partials[(s*seqQ+row)*width+head*headDim+c],exp(maxsum[idx]-m),value);}output_data[row*width+head*headDim+c]=value/denominator;}}
 }
}
