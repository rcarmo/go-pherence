package q4kcshim

/*
#cgo CFLAGS: -O3 -march=rv64gcv_xsmtvdotii -I/home/me/src/llama.cpp-spacemit-pr22863/ggml/include -I/home/me/src/llama.cpp-spacemit-pr22863/ggml/src -I/home/me/src/llama.cpp-spacemit-pr22863/ggml/src/ggml-cpu
#cgo LDFLAGS: -L/usr/lib -lggml-cpu -lggml-base -lggml -lm -lstdc++
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>

extern size_t ime2_gemm_i8i4(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bz, float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc) __asm__("_ZN16spacemit_kernels4ime216gemm_kernel_i8i4EmPKhS2_S2_Pfmmmm");
extern size_t ime2_gemm_i8i8(size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * bz, float * c, size_t count_m, size_t count_n, size_t k_blks, size_t ldc) __asm__("_ZN16spacemit_kernels4ime216gemm_kernel_i8i8EmPKhS2_S2_Pfmmmm");

static void ensure_ai_thread() {
    static __thread int done = 0;
    if (!done) {
        char buf[32];
        int n = snprintf(buf, sizeof(buf), "%ld", (long)syscall(SYS_gettid));
        int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd >= 0) { ssize_t wr = write(fd, buf, n); (void)wr; close(fd); }
        done = 1;
    }
}

static void call_ime2_gemm_i8i4(const uint8_t *a, const uint8_t *b, float *c, int count_n, int k_blks) {
    ensure_ai_thread();
    ime2_gemm_i8i4(32, a, b, b, c, 1, (size_t)count_n, (size_t)k_blks, (size_t)count_n);
}
static void call_ime2_gemm_i8i8(const uint8_t *a, const uint8_t *b, float *c, int count_n, int k_blks) {
    ensure_ai_thread();
    ime2_gemm_i8i8(32, a, b, NULL, c, 1, (size_t)count_n, (size_t)k_blks, (size_t)count_n);
}
*/
import "C"
import "unsafe"

func CallIME2GemmI8I4(a []byte, b []byte, out []float32, countN, kBlks int) {
	if len(a) == 0 || len(b) == 0 || len(out) == 0 {
		return
	}
	C.call_ime2_gemm_i8i4((*C.uint8_t)(unsafe.Pointer(&a[0])), (*C.uint8_t)(unsafe.Pointer(&b[0])), (*C.float)(unsafe.Pointer(&out[0])), C.int(countN), C.int(kBlks))
}

func CallIME2GemmI8I8(a []byte, b []byte, out []float32, countN, kBlks int) {
	if len(a) == 0 || len(b) == 0 || len(out) == 0 {
		return
	}
	C.call_ime2_gemm_i8i8((*C.uint8_t)(unsafe.Pointer(&a[0])), (*C.uint8_t)(unsafe.Pointer(&b[0])), (*C.float)(unsafe.Pointer(&out[0])), C.int(countN), C.int(kBlks))
}
