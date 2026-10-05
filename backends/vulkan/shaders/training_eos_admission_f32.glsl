#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X {float hidden[];};
layout(set=0,binding=1) readonly buffer W {float weight[];};
layout(set=0,binding=2) readonly buffer B {float bias[];};
layout(set=0,binding=3) writeonly buffer O {float result[];};
layout(push_constant) uniform Args {uint rows;uint width;uint budget;} a;
// Return budget+1 on rejected/nonfinite term. Integer mantissa ceilings give
// upward product bounds in 2^95 units, independent of device FTZ/float rounding.
uint units(float x,float y,uint remaining){
 uint xb=floatBitsToUint(x)&0x7fffffffu;uint yb=floatBitsToUint(y)&0x7fffffffu;
 uint xe=xb>>23u;uint ye=yb>>23u;
 if(xe==255u)return a.budget+1u;if(ye==255u)return a.budget+1u;
 uint xm=xb&0x7fffffu;uint ym=yb&0x7fffffu;
 if(xe==0u){xe=1u;}else{xm=xm|0x800000u;}
 if(ye==0u){ye=1u;}else{ym=ym|0x800000u;}
 xm=(xm+4095u)>>12u;ym=(ym+4095u)>>12u;
 uint p=xm*ym;if(p==0u)return 0u;
 uint sum=xe+ye;
 if(sum>=371u){uint shift=sum-371u;if(shift>=31u)return a.budget+1u;if(p>(remaining>>shift))return a.budget+1u;return p<<shift;}
 uint shift=371u-sum;
 if(shift>=32u){if(remaining==0u)return a.budget+1u;return 1u;}
 uint value=p>>shift;uint mask=0xffffffffu>>(32u-shift);if((p&mask)!=0u)value++;
 return value;
}
void main(){
 if(gl_GlobalInvocationID.x!=0u)return;
 for(uint row=0u;row<a.rows;row++){
  uint used=units(bias[0],1.0,a.budget);if(used>a.budget){result[0]=1.0;return;}
  uint remaining=a.budget-used;
  for(uint col=0u;col<a.width;col++){
   used=units(hidden[row*a.width+col],weight[col],remaining);if(used>remaining){result[0]=1.0;return;}remaining-=used;
  }
 }
 result[0]=0.0;
}
