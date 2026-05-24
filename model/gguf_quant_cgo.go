//go:build cgo && riscv64

package model

/*
#cgo CFLAGS: -O3 -march=rv64gcv_zba_zbb_zbs -mabi=lp64d
#include <stdint.h>
#include <stddef.h>

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15);
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1F);
    uint32_t mant = (uint32_t)(h & 0x3FF);
    uint32_t bits;
    if (exp == 0x1F) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            exp++;
            mant &= 0x3FFu;
            bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
        }
    } else {
        bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } v = { bits };
    return v.f;
}
static inline uint16_t le16(const uint8_t *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
static inline uint32_t le32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

float gguf_dot_q2k_row(const uint8_t *raw, const float *x, int in_dim) {
    const int block_elems = 256, block_size = 84;
    int blocks = in_dim / block_elems;
    float total = 0.0f;
    for (int b = 0; b < blocks; b++) {
        const uint8_t *blk = raw + b*block_size;
        const uint8_t *scales = blk;
        const uint8_t *q = blk + 16;
        float d = f16_to_f32(le16(blk + 80));
        float minv = f16_to_f32(le16(blk + 82));
        const float *xb = x + b*block_elems;
        int is = 0, qoff = 0;
        float sum = 0.0f;
        for (int n = 0; n < block_elems; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (float)(sc & 0x0F);
                float ml = minv * (float)(sc >> 4);
                int base = n + j*32;
                for (int l = 0; l < 16; l++) {
                    float w = dl * (float)((q[qoff+l] >> shift) & 3) - ml;
                    sum += w * xb[base+l];
                }
                sc = scales[is++];
                dl = d * (float)(sc & 0x0F);
                ml = minv * (float)(sc >> 4);
                for (int l = 0; l < 16; l++) {
                    float w = dl * (float)((q[qoff+l+16] >> shift) & 3) - ml;
                    sum += w * xb[base+16+l];
                }
                shift += 2;
            }
            qoff += 32;
        }
        total += sum;
    }
    return total;
}

float gguf_dot_q3k_row(const uint8_t *raw, const float *x, int in_dim) {
    const int block_elems = 256, block_size = 110;
    int blocks = in_dim / block_elems;
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    float total = 0.0f;
    for (int b = 0; b < blocks; b++) {
        const uint8_t *blk = raw + b*block_size;
        const uint8_t *hm = blk;
        const uint8_t *q = blk + 32;
        const uint8_t *s = blk + 96;
        float d_all = f16_to_f32(le16(blk + 108));
        uint32_t aux[4];
        aux[0] = le32(s+0); aux[1] = le32(s+4); aux[2] = le32(s+8); aux[3] = 0;
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int8_t scales[16];
        for (int i = 0; i < 4; i++) {
            uint32_t u = aux[i];
            scales[4*i+0] = (int8_t)(uint8_t)(u >> 0);
            scales[4*i+1] = (int8_t)(uint8_t)(u >> 8);
            scales[4*i+2] = (int8_t)(uint8_t)(u >> 16);
            scales[4*i+3] = (int8_t)(uint8_t)(u >> 24);
        }
        const float *xb = x + b*block_elems;
        int is = 0, qoff = 0;
        uint8_t m = 1;
        float sum = 0.0f;
        for (int n = 0; n < block_elems; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (float)(scales[is++] - 32);
                int base = n + j*32;
                for (int l = 0; l < 16; l++) {
                    int8_t lo = (int8_t)((q[qoff+l] >> shift) & 3);
                    if ((hm[l] & m) == 0) lo -= 4;
                    sum += dl * (float)lo * xb[base+l];
                }
                dl = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int8_t lo = (int8_t)((q[qoff+l+16] >> shift) & 3);
                    if ((hm[l+16] & m) == 0) lo -= 4;
                    sum += dl * (float)lo * xb[base+16+l];
                }
                shift += 2;
                m <<= 1;
            }
            qoff += 32;
        }
        total += sum;
    }
    return total;
}

float gguf_dot_q6k_row(const uint8_t *raw, const float *x, int in_dim) {
    const int block_elems = 256, block_size = 210;
    int blocks = in_dim / block_elems;
    float total = 0.0f;
    for (int b = 0; b < blocks; b++) {
        const uint8_t *blk = raw + b*block_size;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        float d = f16_to_f32(le16(blk + 208));
        const float *xb = x + b*block_elems;
        int qlOff = 0, qhOff = 0, scOff = 0;
        float sum = 0.0f;
        for (int n = 0; n < block_elems; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[qlOff+l] & 0x0F) | (((qh[qhOff+l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[qlOff+l+32] & 0x0F) | (((qh[qhOff+l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[qlOff+l] >> 4) | (((qh[qhOff+l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[qlOff+l+32] >> 4) | (((qh[qhOff+l] >> 6) & 3) << 4)) - 32;
                int base = n + l;
                sum += d * (float)sc[scOff+is+0] * (float)q1 * xb[base+0];
                sum += d * (float)sc[scOff+is+2] * (float)q2 * xb[base+32];
                sum += d * (float)sc[scOff+is+4] * (float)q3 * xb[base+64];
                sum += d * (float)sc[scOff+is+6] * (float)q4 * xb[base+96];
            }
            qlOff += 64; qhOff += 32; scOff += 8;
        }
        total += sum;
    }
    return total;
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func QuantGemvCgoFused(out, x []float32, w *gguf.QuantMatrix) error {
	if len(out) < w.OutDim || len(x) < w.InDim || len(w.Raw) == 0 {
		return fmt.Errorf("QuantGemvCgoFused %s bad sizes", w.Name)
	}
	rowBytes, err := w.RowBytes()
	if err != nil {
		return err
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > w.OutDim {
		workers = w.OutDim
	}
	chunk := (w.OutDim + workers - 1) / workers
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	for start := 0; start < w.OutDim; start += chunk {
		end := start + chunk
		if end > w.OutDim {
			end = w.OutDim
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for row := start; row < end; row++ {
				rawPtr := (*C.uint8_t)(unsafe.Pointer(&w.Raw[row*rowBytes]))
				xPtr := (*C.float)(unsafe.Pointer(&x[0]))
				switch w.QType {
				case gguf.QuantQ2_K:
					out[row] = float32(C.gguf_dot_q2k_row(rawPtr, xPtr, C.int(w.InDim)))
				case gguf.QuantQ3_K:
					out[row] = float32(C.gguf_dot_q3k_row(rawPtr, xPtr, C.int(w.InDim)))
				case gguf.QuantQ6_K:
					out[row] = float32(C.gguf_dot_q6k_row(rawPtr, xPtr, C.int(w.InDim)))
				default:
					errCh <- fmt.Errorf("unsupported cgo fused quant type %d", w.QType)
					return
				}
			}
		}(start, end)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return err
		}
	}
	return nil
}
