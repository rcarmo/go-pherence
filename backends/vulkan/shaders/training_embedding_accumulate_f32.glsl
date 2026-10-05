#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer P {float previous[];};
layout(set=0,binding=1) readonly buffer C {float compact[];};
layout(set=0,binding=2) readonly buffer I {float index[];};
layout(set=0,binding=3) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint rows;uint width;uint vocabulary;float scale;} a;
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=a.vocabulary*a.width)return;
 uint token=i/a.width,lo=0u,hi=a.rows;
 while(lo<hi){
  uint mid=lo+(hi-lo)/2u;float id=index[mid];
  if(id>=0.0){if(id<float(token)){lo=mid+1u;continue;}}
  hi=mid;
 }
 precise float contribution=0.0;
 if(lo<a.rows){if(index[lo]==float(token))contribution=compact[lo*a.width+i%a.width];}
 // Keep the legacy dense accumulation's separate multiply/add contract.
 precise float scaled=a.scale*contribution;
 result[i]=previous[i]+scaled;
}
