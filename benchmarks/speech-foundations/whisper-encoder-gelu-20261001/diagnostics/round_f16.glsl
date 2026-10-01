#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float x[];};
layout(set=0,binding=1) buffer O {float y[];};
layout(push_constant) uniform P {uint count;};
// Diagnostic: original whisper.cpp stores encoder K/V in an F16 kv_pad
// (F32->F16 round-to-nearest-even). Bitwise RNE to the F16 grid, kept in F32;
// subnormal F16 grid 2^-24 included. Overflow beyond F16 max rounds to infinity.
void main(){
 uint i=gl_GlobalInvocationID.x;
 if(i>=count) return;
 uint b=floatBitsToUint(x[i]);
 uint sign=b&0x80000000u;
 uint e=(b>>23)&0xffu;
 uint r=0u;
 if(e==102u){
  if((b&0x7fffffu)!=0u){ r=0x33800000u; } // (2^-25,2^-24) -> 2^-24; exact 2^-25 ties to even zero
 }
 if(e>=103u){
  uint shift=13u;
  if(e<113u){ shift=13u+(113u-e); }
  uint mag=b&0x7fffffffu;
  uint lsb=(mag>>shift)&1u;
  uint bias=(1u<<(shift-1u))-1u+lsb;
  r=(mag+bias)&(0xffffffffu<<shift);
 }
 if(r>=0x47800000u){ r=0x7f800000u; } // beyond F16 max rounds to infinity
 y[i]=uintBitsToFloat(sign|r);
}
