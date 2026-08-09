#include <immintrin.h>
#include <stdint.h>
__attribute__((target("avx2,f16c")))
void go_gelu_fp16_mul_avx2(float *dst, const float *gate, const float *up, int n, const uint16_t *table) {
  int i=0;
  const __m256i signmask=_mm256_set1_epi32(0x80000000u), absmask=_mm256_set1_epi32(0x7fffffff);
  const __m256i mantmask=_mm256_set1_epi32(0x7fffff), hidden=_mm256_set1_epi32(0x800000);
  const __m256i roundbit=_mm256_set1_epi32(0x1000), one=_mm256_set1_epi32(1);
  const __m256i exp31=_mm256_set1_epi32(31), inf16=_mm256_set1_epi32(0x7c00);
  const __m256 lo=_mm256_set1_ps(-10.0f), hi=_mm256_set1_ps(10.0f), zero=_mm256_setzero_ps();
  for (;i+8<=n;i+=8) {
    __m256 x=_mm256_loadu_ps(gate+i); __m256i bits=_mm256_castps_si256(x);
    __m256i sign=_mm256_srli_epi32(_mm256_and_si256(bits,signmask),16);
    __m256i exp=_mm256_sub_epi32(_mm256_srli_epi32(_mm256_and_si256(bits,absmask),23),_mm256_set1_epi32(112));
    __m256i mant=_mm256_and_si256(bits,mantmask);
    __m256i normal=_mm256_cmpgt_epi32(exp,_mm256_setzero_si256());
    __m256i rounded=_mm256_add_epi32(mant,roundbit);
    __m256i carry=_mm256_srli_epi32(_mm256_and_si256(rounded,hidden),23);
    __m256i nexp=_mm256_add_epi32(exp,carry);
    __m256i nfrac=_mm256_srli_epi32(_mm256_andnot_si256(_mm256_cmpeq_epi32(carry,one),rounded),13);
    __m256i ni=_mm256_or_si256(_mm256_slli_epi32(nexp,10),nfrac);
    ni=_mm256_blendv_epi8(ni,inf16,_mm256_cmpgt_epi32(nexp,_mm256_set1_epi32(30)));
    __m256i shift=_mm256_sub_epi32(_mm256_set1_epi32(14),exp);
    __m256i tooSmall=_mm256_cmpgt_epi32(_mm256_set1_epi32(-10),exp);
    __m256i rb=_mm256_sllv_epi32(one,_mm256_sub_epi32(shift,one));
    __m256i si=_mm256_srlv_epi32(_mm256_add_epi32(_mm256_or_si256(mant,hidden),rb),shift);
    si=_mm256_andnot_si256(tooSmall,si);
    __m256i idx=_mm256_or_si256(sign,_mm256_blendv_epi8(si,ni,normal));
    __m256i g=_mm256_i32gather_epi32((const int *)table,idx,2);
    g=_mm256_and_si256(g,_mm256_set1_epi32(0xffff));
    __m256i p=_mm256_packus_epi32(g,g);
    __m128i hw=_mm_unpacklo_epi64(_mm256_castsi256_si128(p),_mm256_extracti128_si256(p,1));
    __m256 y=_mm256_cvtph_ps(hw);
    y=_mm256_blendv_ps(y,zero,_mm256_cmp_ps(x,lo,_CMP_LE_OQ));
    y=_mm256_blendv_ps(y,x,_mm256_cmp_ps(x,hi,_CMP_GE_OQ));
    _mm256_storeu_ps(dst+i,_mm256_mul_ps(y,_mm256_loadu_ps(up+i)));
  }
}
