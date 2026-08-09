#include <immintrin.h>
__attribute__((target("avx2")))
void go_rope_partial_avx2(float *x,const float *freq,int heads,int headDim,int rotHalf) {
 const __m256i even=_mm256_setr_epi32(0,2,4,6,0,2,4,6);
 const __m256i odd=_mm256_setr_epi32(1,3,5,7,1,3,5,7);
 for(int h=0;h<heads;h++) {
  float *x0=x+h*headDim, *x1=x0+rotHalf;
  int i=0;
  for(;i+8<=rotHalf;i+=8) {
   __m256 a=_mm256_loadu_ps(freq+2*i), b=_mm256_loadu_ps(freq+2*i+8);
   __m256 ae=_mm256_permutevar8x32_ps(a,even), be=_mm256_permutevar8x32_ps(b,even);
   __m256 ao=_mm256_permutevar8x32_ps(a,odd), bo=_mm256_permutevar8x32_ps(b,odd);
   __m256 c=_mm256_permute2f128_ps(ae,be,0x20), s=_mm256_permute2f128_ps(ao,bo,0x20);
   __m256 v0=_mm256_loadu_ps(x0+i),v1=_mm256_loadu_ps(x1+i);
   _mm256_storeu_ps(x0+i,_mm256_sub_ps(_mm256_mul_ps(v0,c),_mm256_mul_ps(v1,s)));
   _mm256_storeu_ps(x1+i,_mm256_add_ps(_mm256_mul_ps(v0,s),_mm256_mul_ps(v1,c)));
  }
  for(;i<rotHalf;i++) { float c=freq[2*i],s=freq[2*i+1],a=x0[i],b=x1[i]; x0[i]=a*c-b*s; x1[i]=a*s+b*c; }
 }
}
