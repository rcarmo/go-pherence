// Experimental llama.cpp b607 Q4_0x8 x Q8_0x4 kernel.
// This is intentionally quarantined behind Go's unexported experimental wrapper.
#include <immintrin.h>
#include <stdint.h>

#define LLAMA_TARGET __attribute__((target("avx2,avxvnni,fma,f16c")))

static inline LLAMA_TARGET __m256i dot_i8(__m256i acc, __m256i x, __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_dpbusd_avx_epi32(acc, ax, sy);
}

static inline LLAMA_TARGET __m256i decode_lo(__m256i x, __m256i lut, __m256i mask) {
    return _mm256_shuffle_epi8(lut, _mm256_and_si256(x, mask));
}

static inline LLAMA_TARGET __m256i decode_hi(__m256i x, __m256i lut, __m256i mask) {
    return _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x, 4), mask));
}

// q4 points to blocks consecutive at 144 bytes each; q8 points to blocks
// consecutive at 136 bytes each. out is token-major [4][8].
LLAMA_TARGET void go_llama_q4_0_q8_0_8x4(const uint8_t *q4, const uint8_t *q8, int blocks, float *out) {
    __m256 acc[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    const __m256i mask = _mm256_set1_epi8(0x0f);
    const __m128i lut128 = _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8,
                                        7,  6,  5,  4,  3,  2,  1,  0);
    const __m256i lut = _mm256_broadcastsi128_si256(lut128);
    const __m256i order = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    const __m256i zero = _mm256_setzero_si256();

    for (int b = 0; b < blocks; b++, q4 += 144, q8 += 136) {
        const __m256i r0 = _mm256_loadu_si256((const __m256i *)(q4 + 16));
        const __m256i r1 = _mm256_loadu_si256((const __m256i *)(q4 + 48));
        const __m256i r2 = _mm256_loadu_si256((const __m256i *)(q4 + 80));
        const __m256i r3 = _mm256_loadu_si256((const __m256i *)(q4 + 112));

        const __m256i raw0145_0 = _mm256_blend_epi32(r0, _mm256_permutevar8x32_epi32(r1, order), 0xf0);
        const __m256i raw2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(r0, order), r1, 0xf0);
        const __m256i raw0145_1 = _mm256_blend_epi32(r2, _mm256_permutevar8x32_epi32(r3, order), 0xf0);
        const __m256i raw2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(r2, order), r3, 0xf0);

        const __m256i q0145_0 = decode_lo(raw0145_0, lut, mask);
        const __m256i q2367_0 = decode_lo(raw2367_0, lut, mask);
        const __m256i q0145_1 = decode_lo(raw0145_1, lut, mask);
        const __m256i q2367_1 = decode_lo(raw2367_1, lut, mask);
        const __m256i q0145_2 = decode_hi(raw0145_0, lut, mask);
        const __m256i q2367_2 = decode_hi(raw2367_0, lut, mask);
        const __m256i q0145_3 = decode_hi(raw0145_1, lut, mask);
        const __m256i q2367_3 = decode_hi(raw2367_1, lut, mask);

        const __m256i q0145_0a = _mm256_shuffle_epi32(q0145_0, 136);
        const __m256i q0145_1a = _mm256_shuffle_epi32(q0145_1, 136);
        const __m256i q0145_2a = _mm256_shuffle_epi32(q0145_2, 136);
        const __m256i q0145_3a = _mm256_shuffle_epi32(q0145_3, 136);
        const __m256i q2367_0a = _mm256_shuffle_epi32(q2367_0, 136);
        const __m256i q2367_1a = _mm256_shuffle_epi32(q2367_1, 136);
        const __m256i q2367_2a = _mm256_shuffle_epi32(q2367_2, 136);
        const __m256i q2367_3a = _mm256_shuffle_epi32(q2367_3, 136);
        const __m256i q0145_0b = _mm256_shuffle_epi32(q0145_0, 221);
        const __m256i q0145_1b = _mm256_shuffle_epi32(q0145_1, 221);
        const __m256i q0145_2b = _mm256_shuffle_epi32(q0145_2, 221);
        const __m256i q0145_3b = _mm256_shuffle_epi32(q0145_3, 221);
        const __m256i q2367_0b = _mm256_shuffle_epi32(q2367_0, 221);
        const __m256i q2367_1b = _mm256_shuffle_epi32(q2367_1, 221);
        const __m256i q2367_2b = _mm256_shuffle_epi32(q2367_2, 221);
        const __m256i q2367_3b = _mm256_shuffle_epi32(q2367_3, 221);

        const __m256i a0 = _mm256_loadu_si256((const __m256i *)(q8 + 8));
        const __m256i a1 = _mm256_loadu_si256((const __m256i *)(q8 + 40));
        const __m256i a2 = _mm256_loadu_si256((const __m256i *)(q8 + 72));
        const __m256i a3 = _mm256_loadu_si256((const __m256i *)(q8 + 104));
        const __m256i a01_0 = _mm256_permute2x128_si256(a0, a0, 0);
        const __m256i a23_0 = _mm256_permute2x128_si256(a0, a0, 17);
        const __m256i a01_1 = _mm256_permute2x128_si256(a1, a1, 0);
        const __m256i a23_1 = _mm256_permute2x128_si256(a1, a1, 17);
        const __m256i a01_2 = _mm256_permute2x128_si256(a2, a2, 0);
        const __m256i a23_2 = _mm256_permute2x128_si256(a2, a2, 17);
        const __m256i a01_3 = _mm256_permute2x128_si256(a3, a3, 0);
        const __m256i a23_3 = _mm256_permute2x128_si256(a3, a3, 17);

#define SHUF_A(v, imm) _mm256_shuffle_epi32((v), (imm))
#define DOT4(a_0,a_1,a_2,a_3,q_0,q_1,q_2,q_3) \
        dot_i8(dot_i8(dot_i8(dot_i8(zero, (a_3), (q_3)), (a_2), (q_2)), (a_1), (q_1)), (a_0), (q_0))
        __m256i i00a = DOT4(SHUF_A(a01_0,160), SHUF_A(a01_1,160), SHUF_A(a01_2,160), SHUF_A(a01_3,160), q0145_0a,q0145_1a,q0145_2a,q0145_3a);
        __m256i i01a = DOT4(SHUF_A(a01_0,160), SHUF_A(a01_1,160), SHUF_A(a01_2,160), SHUF_A(a01_3,160), q2367_0a,q2367_1a,q2367_2a,q2367_3a);
        __m256i i10a = DOT4(SHUF_A(a23_0,160), SHUF_A(a23_1,160), SHUF_A(a23_2,160), SHUF_A(a23_3,160), q0145_0a,q0145_1a,q0145_2a,q0145_3a);
        __m256i i11a = DOT4(SHUF_A(a23_0,160), SHUF_A(a23_1,160), SHUF_A(a23_2,160), SHUF_A(a23_3,160), q2367_0a,q2367_1a,q2367_2a,q2367_3a);
        __m256i i00b = DOT4(SHUF_A(a01_0,245), SHUF_A(a01_1,245), SHUF_A(a01_2,245), SHUF_A(a01_3,245), q0145_0b,q0145_1b,q0145_2b,q0145_3b);
        __m256i i01b = DOT4(SHUF_A(a01_0,245), SHUF_A(a01_1,245), SHUF_A(a01_2,245), SHUF_A(a01_3,245), q2367_0b,q2367_1b,q2367_2b,q2367_3b);
        __m256i i10b = DOT4(SHUF_A(a23_0,245), SHUF_A(a23_1,245), SHUF_A(a23_2,245), SHUF_A(a23_3,245), q0145_0b,q0145_1b,q0145_2b,q0145_3b);
        __m256i i11b = DOT4(SHUF_A(a23_0,245), SHUF_A(a23_1,245), SHUF_A(a23_2,245), SHUF_A(a23_3,245), q2367_0b,q2367_1b,q2367_2b,q2367_3b);
#undef DOT4
#undef SHUF_A

        const __m256i i00 = _mm256_add_epi32(i00a, i00b);
        const __m256i i01 = _mm256_add_epi32(i01a, i01b);
        const __m256i i10 = _mm256_add_epi32(i10a, i10b);
        const __m256i i11 = _mm256_add_epi32(i11a, i11b);
        const __m256i dots[4] = {
            _mm256_blend_epi32(i00, _mm256_shuffle_epi32(i01, 78), 204),
            _mm256_blend_epi32(_mm256_shuffle_epi32(i00, 78), i01, 204),
            _mm256_blend_epi32(i10, _mm256_shuffle_epi32(i11, 78), 204),
            _mm256_blend_epi32(_mm256_shuffle_epi32(i10, 78), i11, 204),
        };

        const __m256 wd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q4));
        for (int token = 0; token < 4; token++) {
            const float ad = _cvtsh_ss(*(const uint16_t *)(q8 + token * 2));
            const __m256 scale = _mm256_mul_ps(wd, _mm256_set1_ps(ad));
            acc[token] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dots[token]), scale, acc[token]);
        }
    }
    for (int token = 0; token < 4; token++) {
        _mm256_storeu_ps(out + token * 8, acc[token]);
    }
}

// q4 points to blocks consecutive at 144 bytes each. q8 points to four
// block-consecutive 136-byte panels. out is token-major [16][8]. Q4 is loaded
// and decoded once per block, then reused across all four Q8 panels.
LLAMA_TARGET void go_llama_q4_0_q8_0_8x16(const uint8_t *q4, const uint8_t *q8, int blocks, float *out) {
    __m256 acc[16];
    for (int token = 0; token < 16; token++) {
        acc[token] = _mm256_setzero_ps();
    }
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
        const __m256i r0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(q4 + 16)), xor88);
        const __m256i r1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(q4 + 48)), xor88);
        const __m256i r2 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(q4 + 80)), xor88);
        const __m256i r3 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(q4 + 112)), xor88);

        const __m256i raw0145_0 = _mm256_blend_epi32(r0, _mm256_permutevar8x32_epi32(r1, order), 0xf0);
        const __m256i raw2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(r0, order), r1, 0xf0);
        const __m256i raw0145_1 = _mm256_blend_epi32(r2, _mm256_permutevar8x32_epi32(r3, order), 0xf0);
        const __m256i raw2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(r2, order), r3, 0xf0);

        const __m256i q0145_0 = _mm256_and_si256(raw0145_0, mask);
        const __m256i q2367_0 = _mm256_and_si256(raw2367_0, mask);
        const __m256i q0145_1 = _mm256_and_si256(raw0145_1, mask);
        const __m256i q2367_1 = _mm256_and_si256(raw2367_1, mask);
        const __m256i q0145_2 = _mm256_and_si256(_mm256_srli_epi16(raw0145_0, 4), mask);
        const __m256i q2367_2 = _mm256_and_si256(_mm256_srli_epi16(raw2367_0, 4), mask);
        const __m256i q0145_3 = _mm256_and_si256(_mm256_srli_epi16(raw0145_1, 4), mask);
        const __m256i q2367_3 = _mm256_and_si256(_mm256_srli_epi16(raw2367_1, 4), mask);

        const __m256i q0145_0a = _mm256_shuffle_epi32(q0145_0, 136);
        const __m256i q0145_1a = _mm256_shuffle_epi32(q0145_1, 136);
        const __m256i q0145_2a = _mm256_shuffle_epi32(q0145_2, 136);
        const __m256i q0145_3a = _mm256_shuffle_epi32(q0145_3, 136);
        const __m256i q2367_0a = _mm256_shuffle_epi32(q2367_0, 136);
        const __m256i q2367_1a = _mm256_shuffle_epi32(q2367_1, 136);
        const __m256i q2367_2a = _mm256_shuffle_epi32(q2367_2, 136);
        const __m256i q2367_3a = _mm256_shuffle_epi32(q2367_3, 136);
        const __m256i q0145_0b = _mm256_shuffle_epi32(q0145_0, 221);
        const __m256i q0145_1b = _mm256_shuffle_epi32(q0145_1, 221);
        const __m256i q0145_2b = _mm256_shuffle_epi32(q0145_2, 221);
        const __m256i q0145_3b = _mm256_shuffle_epi32(q0145_3, 221);
        const __m256i q2367_0b = _mm256_shuffle_epi32(q2367_0, 221);
        const __m256i q2367_1b = _mm256_shuffle_epi32(q2367_1, 221);
        const __m256i q2367_2b = _mm256_shuffle_epi32(q2367_2, 221);
        const __m256i q2367_3b = _mm256_shuffle_epi32(q2367_3, 221);

        const __m256 wd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q4));
        for (int panel = 0; panel < 4; panel++) {
            const uint8_t *a = q8 + (int64_t)panel * blocks * 136 + (int64_t)b * 136;
            const __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + 8));
            const __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + 40));
            const __m256i a2 = _mm256_loadu_si256((const __m256i *)(a + 72));
            const __m256i a3 = _mm256_loadu_si256((const __m256i *)(a + 104));
            const __m256i a01_0 = _mm256_permute2x128_si256(a0, a0, 0);
            const __m256i a23_0 = _mm256_permute2x128_si256(a0, a0, 17);
            const __m256i a01_1 = _mm256_permute2x128_si256(a1, a1, 0);
            const __m256i a23_1 = _mm256_permute2x128_si256(a1, a1, 17);
            const __m256i a01_2 = _mm256_permute2x128_si256(a2, a2, 0);
            const __m256i a23_2 = _mm256_permute2x128_si256(a2, a2, 17);
            const __m256i a01_3 = _mm256_permute2x128_si256(a3, a3, 0);
            const __m256i a23_3 = _mm256_permute2x128_si256(a3, a3, 17);

#define SHUF_A16(v, imm) _mm256_shuffle_epi32((v), (imm))
#define DOT4_16(a_0,a_1,a_2,a_3,q_0,q_1,q_2,q_3) \
            _mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(_mm256_dpbusd_avx_epi32(zero, (q_3), (a_3)), (q_2), (a_2)), (q_1), (a_1)), (q_0), (a_0))
            __m256i i00a = DOT4_16(SHUF_A16(a01_0,160), SHUF_A16(a01_1,160), SHUF_A16(a01_2,160), SHUF_A16(a01_3,160), q0145_0a,q0145_1a,q0145_2a,q0145_3a);
            __m256i i01a = DOT4_16(SHUF_A16(a01_0,160), SHUF_A16(a01_1,160), SHUF_A16(a01_2,160), SHUF_A16(a01_3,160), q2367_0a,q2367_1a,q2367_2a,q2367_3a);
            __m256i i10a = DOT4_16(SHUF_A16(a23_0,160), SHUF_A16(a23_1,160), SHUF_A16(a23_2,160), SHUF_A16(a23_3,160), q0145_0a,q0145_1a,q0145_2a,q0145_3a);
            __m256i i11a = DOT4_16(SHUF_A16(a23_0,160), SHUF_A16(a23_1,160), SHUF_A16(a23_2,160), SHUF_A16(a23_3,160), q2367_0a,q2367_1a,q2367_2a,q2367_3a);
            __m256i i00b = DOT4_16(SHUF_A16(a01_0,245), SHUF_A16(a01_1,245), SHUF_A16(a01_2,245), SHUF_A16(a01_3,245), q0145_0b,q0145_1b,q0145_2b,q0145_3b);
            __m256i i01b = DOT4_16(SHUF_A16(a01_0,245), SHUF_A16(a01_1,245), SHUF_A16(a01_2,245), SHUF_A16(a01_3,245), q2367_0b,q2367_1b,q2367_2b,q2367_3b);
            __m256i i10b = DOT4_16(SHUF_A16(a23_0,245), SHUF_A16(a23_1,245), SHUF_A16(a23_2,245), SHUF_A16(a23_3,245), q0145_0b,q0145_1b,q0145_2b,q0145_3b);
            __m256i i11b = DOT4_16(SHUF_A16(a23_0,245), SHUF_A16(a23_1,245), SHUF_A16(a23_2,245), SHUF_A16(a23_3,245), q2367_0b,q2367_1b,q2367_2b,q2367_3b);
#undef DOT4_16
#undef SHUF_A16

            const __m256i i00 = _mm256_add_epi32(i00a, i00b);
            const __m256i i01 = _mm256_add_epi32(i01a, i01b);
            const __m256i i10 = _mm256_add_epi32(i10a, i10b);
            const __m256i i11 = _mm256_add_epi32(i11a, i11b);
            const __m256i dots[4] = {
                _mm256_blend_epi32(i00, _mm256_shuffle_epi32(i01, 78), 204),
                _mm256_blend_epi32(_mm256_shuffle_epi32(i00, 78), i01, 204),
                _mm256_blend_epi32(i10, _mm256_shuffle_epi32(i11, 78), 204),
                _mm256_blend_epi32(_mm256_shuffle_epi32(i10, 78), i11, 204),
            };
            __m256i activation_sums = _mm256_dpbusd_avx_epi32(zero, ones, a0);
            activation_sums = _mm256_dpbusd_avx_epi32(activation_sums, ones, a1);
            activation_sums = _mm256_dpbusd_avx_epi32(activation_sums, ones, a2);
            activation_sums = _mm256_dpbusd_avx_epi32(activation_sums, ones, a3);
            const __m256i correction_pairs = _mm256_hadd_epi32(activation_sums, activation_sums);
            for (int token = 0; token < 4; token++) {
                const __m256i sum = _mm256_permutevar8x32_epi32(correction_pairs, correction_index[token]);
                const __m256i corrected = _mm256_sub_epi32(dots[token], _mm256_slli_epi32(sum, 3));
                const float ad = _cvtsh_ss(*(const uint16_t *)(a + token * 2));
                const __m256 scale = _mm256_mul_ps(wd, _mm256_set1_ps(ad));
                const int output = panel * 4 + token;
                acc[output] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(corrected), scale, acc[output]);
            }
        }
    }
    for (int token = 0; token < 16; token++) {
        _mm256_storeu_ps(out + token * 8, acc[token]);
    }
}

// Row-range orchestration lets Go schedule aligned groups without one language
// crossing per tile. q4 starts at row_base and contains row_groups panels.
LLAMA_TARGET void go_llama_q4_0_q8_0_projection_rows(const uint8_t *q4, const uint8_t *q8,
        int row_base, int row_groups, int rows, int tokens, int blocks, float *out) {
    const int token_groups = (tokens + 3) / 4;
    for (int super = 0; super < token_groups; super += 4) {
        const int panels = token_groups - super;
        if (panels >= 4) {
            const uint8_t *a = q8 + (int64_t)super * blocks * 136;
            for (int rg = 0; rg < row_groups; rg++) {
                const int output_row = row_base + rg * 8;
                const uint8_t *w = q4 + (int64_t)rg * blocks * 144;
                float tile[128];
                go_llama_q4_0_q8_0_8x16(w, a, blocks, tile);
                for (int token = 0; token < 16 && super * 4 + token < tokens; token++) {
                    for (int row = 0; row < 8 && output_row + row < rows; row++) {
                        out[(int64_t)(super * 4 + token) * rows + output_row + row] = tile[token * 8 + row];
                    }
                }
            }
            continue;
        }
        for (int tg = super; tg < token_groups; tg++) {
            const uint8_t *a = q8 + (int64_t)tg * blocks * 136;
            for (int rg = 0; rg < row_groups; rg++) {
                const int output_row = row_base + rg * 8;
                const uint8_t *w = q4 + (int64_t)rg * blocks * 144;
                float tile[32];
                go_llama_q4_0_q8_0_8x4(w, a, blocks, tile);
                for (int token = 0; token < 4 && tg * 4 + token < tokens; token++) {
                    for (int row = 0; row < 8 && output_row + row < rows; row++) {
                        out[(int64_t)(tg * 4 + token) * rows + output_row + row] = tile[token * 8 + row];
                    }
                }
            }
        }
    }
}

// Whole-projection orchestration amortises the language boundary. Full groups
// use the fused 8x16 kernel; one to three remaining Q8 panels use 8x4 calls.
// Padded row/token tails are not copied to the logical output.
LLAMA_TARGET void go_llama_q4_0_q8_0_projection(const uint8_t *q4, const uint8_t *q8,
        int rows, int tokens, int blocks, float *out) {
    go_llama_q4_0_q8_0_projection_rows(q4, q8, 0, (rows + 7) / 8,
            rows, tokens, blocks, out);
}
