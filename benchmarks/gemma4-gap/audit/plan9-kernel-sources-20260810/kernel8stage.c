#include <immintrin.h>
#include <stdint.h>
#define LLAMA_TARGET __attribute__((target("avx2,avxvnni,fma,f16c")))

LLAMA_TARGET void go_llama_q4_0_q8_0_8x8_stage(const uint8_t *q4, const uint8_t *q8, int blocks, float *out) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
    const __m256i mask = _mm256_set1_epi8(0x0f);
    const __m256i xor88 = _mm256_set1_epi8((char)0x88);
    const __m256i ones = _mm256_set1_epi8(1);
    const __m256i order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i correction_index[4] = {
        _mm256_set1_epi32(0), _mm256_set1_epi32(1),
        _mm256_set1_epi32(4), _mm256_set1_epi32(5),
    };

    for (int b = 0; b < blocks; b++, q4 += 144) {
        __m256i p00 = zero, p01 = zero, p10 = zero, p11 = zero;
        __m256i q00 = zero, q01 = zero, q10 = zero, q11 = zero;
        __m256i corr0 = zero, corr1 = zero;
        for (int stage = 0; stage < 4; stage++) {
            const int pair = stage & 1;
            const int high = stage >> 1;
            const __m256i r0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(q4 + 16 + pair*64)), xor88);
            const __m256i r1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(q4 + 48 + pair*64)), xor88);
            __m256i rawA = _mm256_blend_epi32(r0, _mm256_permutevar8x32_epi32(r1, order), 0xf0);
            __m256i rawB = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(r0, order), r1, 0xf0);
            if (high) {
                rawA = _mm256_srli_epi16(rawA, 4);
                rawB = _mm256_srli_epi16(rawB, 4);
            }
            rawA = _mm256_and_si256(rawA, mask);
            rawB = _mm256_and_si256(rawB, mask);
            const __m256i qa = _mm256_shuffle_epi32(rawA, 136);
            const __m256i qb = _mm256_shuffle_epi32(rawB, 136);
            const __m256i qc = _mm256_shuffle_epi32(rawA, 221);
            const __m256i qd = _mm256_shuffle_epi32(rawB, 221);

            const __m256i a0 = _mm256_loadu_si256((const __m256i *)(q8 + b*136 + 8 + stage*32));
            const __m256i a001 = _mm256_permute2x128_si256(a0, a0, 0);
            const __m256i a023 = _mm256_permute2x128_si256(a0, a0, 17);
            p00 = _mm256_dpbusd_avx_epi32(p00, qa, _mm256_shuffle_epi32(a001,160));
            p00 = _mm256_dpbusd_avx_epi32(p00, qc, _mm256_shuffle_epi32(a001,245));
            p01 = _mm256_dpbusd_avx_epi32(p01, qb, _mm256_shuffle_epi32(a001,160));
            p01 = _mm256_dpbusd_avx_epi32(p01, qd, _mm256_shuffle_epi32(a001,245));
            p10 = _mm256_dpbusd_avx_epi32(p10, qa, _mm256_shuffle_epi32(a023,160));
            p10 = _mm256_dpbusd_avx_epi32(p10, qc, _mm256_shuffle_epi32(a023,245));
            p11 = _mm256_dpbusd_avx_epi32(p11, qb, _mm256_shuffle_epi32(a023,160));
            p11 = _mm256_dpbusd_avx_epi32(p11, qd, _mm256_shuffle_epi32(a023,245));
            corr0 = _mm256_dpbusd_avx_epi32(corr0, ones, a0);

            const uint8_t *p1 = q8 + blocks*136 + b*136;
            const __m256i a1 = _mm256_loadu_si256((const __m256i *)(p1 + 8 + stage*32));
            const __m256i a101 = _mm256_permute2x128_si256(a1, a1, 0);
            const __m256i a123 = _mm256_permute2x128_si256(a1, a1, 17);
            q00 = _mm256_dpbusd_avx_epi32(q00, qa, _mm256_shuffle_epi32(a101,160));
            q00 = _mm256_dpbusd_avx_epi32(q00, qc, _mm256_shuffle_epi32(a101,245));
            q01 = _mm256_dpbusd_avx_epi32(q01, qb, _mm256_shuffle_epi32(a101,160));
            q01 = _mm256_dpbusd_avx_epi32(q01, qd, _mm256_shuffle_epi32(a101,245));
            q10 = _mm256_dpbusd_avx_epi32(q10, qa, _mm256_shuffle_epi32(a123,160));
            q10 = _mm256_dpbusd_avx_epi32(q10, qc, _mm256_shuffle_epi32(a123,245));
            q11 = _mm256_dpbusd_avx_epi32(q11, qb, _mm256_shuffle_epi32(a123,160));
            q11 = _mm256_dpbusd_avx_epi32(q11, qd, _mm256_shuffle_epi32(a123,245));
            corr1 = _mm256_dpbusd_avx_epi32(corr1, ones, a1);
        }

        const __m256 wd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q4));
#define FINISH_PANEL(PREFIX,CORR,A0,A1,A2,A3,SCALEPTR) do { \
        const __m256i d0 = _mm256_blend_epi32(PREFIX##00, _mm256_shuffle_epi32(PREFIX##01,78),204); \
        const __m256i d1 = _mm256_blend_epi32(_mm256_shuffle_epi32(PREFIX##00,78), PREFIX##01,204); \
        const __m256i d2 = _mm256_blend_epi32(PREFIX##10, _mm256_shuffle_epi32(PREFIX##11,78),204); \
        const __m256i d3 = _mm256_blend_epi32(_mm256_shuffle_epi32(PREFIX##10,78), PREFIX##11,204); \
        const __m256i cp = _mm256_hadd_epi32(CORR,CORR); \
        __m256 ad = _mm256_cvtph_ps(_mm_loadl_epi64((const __m128i *)(SCALEPTR))); \
        ad = _mm256_permute2f128_ps(ad,ad,0); \
        __m256i c0 = _mm256_permutevar8x32_epi32(cp, correction_index[0]); \
        __m256i c1 = _mm256_permutevar8x32_epi32(cp, correction_index[1]); \
        __m256i c2 = _mm256_permutevar8x32_epi32(cp, correction_index[2]); \
        __m256i c3 = _mm256_permutevar8x32_epi32(cp, correction_index[3]); \
        c0 = _mm256_sub_epi32(d0,_mm256_slli_epi32(c0,3)); \
        c1 = _mm256_sub_epi32(d1,_mm256_slli_epi32(c1,3)); \
        c2 = _mm256_sub_epi32(d2,_mm256_slli_epi32(c2,3)); \
        c3 = _mm256_sub_epi32(d3,_mm256_slli_epi32(c3,3)); \
        A0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0),_mm256_mul_ps(wd,_mm256_permute_ps(ad,0)),A0); \
        A1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1),_mm256_mul_ps(wd,_mm256_permute_ps(ad,85)),A1); \
        A2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c2),_mm256_mul_ps(wd,_mm256_permute_ps(ad,170)),A2); \
        A3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c3),_mm256_mul_ps(wd,_mm256_permute_ps(ad,255)),A3); \
    } while (0)
        FINISH_PANEL(p,corr0,acc0,acc1,acc2,acc3,q8+b*136);
        FINISH_PANEL(q,corr1,acc4,acc5,acc6,acc7,q8+blocks*136+b*136);
#undef FINISH_PANEL
    }
    _mm256_storeu_ps(out+0,acc0); _mm256_storeu_ps(out+8,acc1);
    _mm256_storeu_ps(out+16,acc2); _mm256_storeu_ps(out+24,acc3);
    _mm256_storeu_ps(out+32,acc4); _mm256_storeu_ps(out+40,acc5);
    _mm256_storeu_ps(out+48,acc6); _mm256_storeu_ps(out+56,acc7);
}
