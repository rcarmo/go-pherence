#version 450
// Cooperative Q8_1 quantiser: 8 invocations per 32-value block, coalesced vec4
// loads. Max, invalid flags and integer sums are order-independent, and every
// per-value expression matches q8.glsl, so output words are identical.
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float x[];};
layout(set=0,binding=1) buffer Q {uint q[];};
layout(push_constant) uniform P {uint blocks;};
shared float smax[256];
shared uint sbad[256];
shared int ssum[256];
void main(){
 uint tid=gl_LocalInvocationID.x,lane=tid%8,base=tid-lane;
 uint b=gl_WorkGroupID.x*32+tid/8;bool live=b<blocks;
 float v[4];for(uint j=0;j<4;j++)v[j]=live?x[b*32+lane*4+j]:0.0;
 uint bad=0;float amax=0.0;
 for(uint j=0;j<4;j++){float e=v[j];uint bits=floatBitsToUint(e)&0x7fffffff;if(bits>0x447a0000)bad=1;amax=max(amax,abs(e));}
 smax[tid]=amax;sbad[tid]=bad;barrier();
 for(uint s=4;s>0;s>>=1){if(lane<s){smax[tid]=max(smax[tid],smax[tid+s]);sbad[tid]|=sbad[tid+s];}barrier();}
 amax=smax[base];bad=sbad[base];
 uint maximum=floatBitsToUint(amax);if(maximum>0){if(maximum<0x0da24260)bad=1;}
 float d=amax*(1.0/127.0);float inv=0.0;if(amax>0.0)inv=127.0/amax;
 int sum=0;uint packed=0;
 for(uint j=0;j<4;j++){int r=int(round(v[j]*inv));sum+=r;packed|=(uint(r)&255)<<uint(j*8);}
 ssum[tid]=sum;barrier();
 for(uint s=4;s>0;s>>=1){if(lane<s)ssum[tid]+=ssum[tid+s];barrier();}
 if(live){
  if(bad!=0){q[b*9+1+lane]=0;if(lane==0)q[b*9]=0x7e007e00;}
  else{q[b*9+1+lane]=packed;if(lane==0){vec2 ds;ds.x=d;ds.y=float(ssum[base])*d;q[b*9]=packHalf2x16(ds);}}
 }
}
