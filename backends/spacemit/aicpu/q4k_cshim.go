//go:build cgo && q4kcshim

package aicpu

/*
#cgo CFLAGS: -march=rv64gcv_xsmtvdotii
#cgo LDFLAGS: -lggml-cpu -lggml-base -lggml -lm -lstdc++
#include <stdint.h>
#include <stddef.h>
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <stdio.h>
#include <sched.h>

void k3_i8i4_m1(const uint8_t *a, const uint8_t *b, float *c, size_t k_blks, size_t nblks);

static inline void ensure_ai_thread(void) {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd >= 0) {
        char buf[32];
        int n = snprintf(buf, sizeof(buf), "%ld", (long)syscall(SYS_gettid));
        write(fd, buf, n);
        close(fd);
    }
}

static inline void call_k3_i8i4_m1_ai(const uint8_t *a, const uint8_t *b, float *c, size_t k_blks, size_t nblks) {
    ensure_ai_thread();
    k3_i8i4_m1(a, b, c, k_blks, nblks);
}

extern size_t ime2_gemm_i8i4(size_t blk_len,
    const uint8_t * quant_a_ptr,
    const uint8_t * quant_b_data,
    const uint8_t * quant_b_zp,
    float * c_ptr,
    size_t count_m,
    size_t count_n,
    size_t k_blks,
    size_t ldc) asm("_ZN16spacemit_kernels4ime216gemm_kernel_i8i4EmPKhS2_S2_Pfmmmm");

static inline size_t call_ime2_gemm_i8i4(size_t blk_len,
    const uint8_t * quant_a_ptr,
    const uint8_t * quant_b_data,
    const uint8_t * quant_b_zp,
    float * c_ptr,
    size_t count_m,
    size_t count_n,
    size_t k_blks,
    size_t ldc) {
    return ime2_gemm_i8i4(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
}
*/
import "C"

import "unsafe"

func callLocalK3I8I4M1(quantA, quantB []byte, out []float32, countN, kBlks int) {
	if len(quantA) == 0 || len(quantB) == 0 || len(out) == 0 {
		return
	}
	C.call_k3_i8i4_m1_ai(
		(*C.uint8_t)(unsafe.Pointer(&quantA[0])),
		(*C.uint8_t)(unsafe.Pointer(&quantB[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.size_t(kBlks),
		C.size_t(countN),
	)
}

func callIME2GemmI8I4(quantA, quantB []byte, out []float32, countN, kBlks int) {
	if len(quantA) == 0 || len(quantB) == 0 || len(out) == 0 {
		return
	}
	pa := C.aligned_alloc(128, C.size_t((len(quantA)+127)&^127))
	pb := C.aligned_alloc(128, C.size_t((len(quantB)+127)&^127))
	pc := C.aligned_alloc(128, C.size_t(((len(out)*4)+127)&^127))
	if pa == nil || pb == nil || pc == nil {
		panic("aligned_alloc failed")
	}
	defer C.free(pa)
	defer C.free(pb)
	defer C.free(pc)
	C.memset(pa, 0, C.size_t((len(quantA)+127)&^127))
	C.memset(pb, 0, C.size_t((len(quantB)+127)&^127))
	C.memset(pc, 0, C.size_t(((len(out)*4)+127)&^127))
	C.memcpy(pa, unsafe.Pointer(&quantA[0]), C.size_t(len(quantA)))
	C.memcpy(pb, unsafe.Pointer(&quantB[0]), C.size_t(len(quantB)))
	var zpDummy [1]byte
	C.call_ime2_gemm_i8i4(
		C.size_t(32),
		(*C.uint8_t)(pa),
		(*C.uint8_t)(pb),
		(*C.uint8_t)(unsafe.Pointer(&zpDummy[0])),
		(*C.float)(pc),
		C.size_t(1),
		C.size_t(countN),
		C.size_t(kBlks),
		C.size_t(countN),
	)
	C.memcpy(unsafe.Pointer(&out[0]), pc, C.size_t(len(out)*4))
}
