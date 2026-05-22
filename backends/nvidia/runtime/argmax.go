package nvidia

import (
	"fmt"
	"math"
	"unsafe"
)

var fnArgmaxF32 CUfunction

const argmaxF32PTX = `.version 7.0
.target sm_70
.address_size 64

.visible .entry argmax_f32(
    .param .u64 logits,
    .param .u64 outVal,
    .param .u64 outIdx,
    .param .u32 n
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [logits];
    ld.param.u64 %rd2, [outVal];
    ld.param.u64 %rd3, [outIdx];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ntid.x;

    mov.f32 %f1, 0fFF800000; // -inf
    mov.u32 %r4, 0;
    mov.u32 %r5, %r2;

LOOP:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;
    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    setp.gt.f32 %p2, %f2, %f1;
    setp.eq.f32 %p3, %f2, %f1;
    setp.lt.u32 %p4, %r5, %r4;
    and.pred %p5, %p3, %p4;
    or.pred %p2, %p2, %p5;
    @!%p2 bra SKIP;
    mov.f32 %f1, %f2;
    mov.u32 %r4, %r5;
SKIP:
    add.u32 %r5, %r5, %r3;
    bra LOOP;

DONE:
    // shared val[256], idx[256]
    .shared .align 4 .b8 s_val[1024];
    .shared .align 4 .b8 s_idx[1024];
    cvta.shared.u64 %rd6, s_val;
    cvta.shared.u64 %rd7, s_idx;
    mul.wide.u32 %rd8, %r2, 4;
    add.u64 %rd9, %rd6, %rd8;
    st.shared.f32 [%rd9], %f1;
    add.u64 %rd10, %rd7, %rd8;
    st.shared.u32 [%rd10], %r4;
    bar.sync 0;

    mov.u32 %r6, 128;
REDUCE:
    setp.ge.u32 %p1, %r2, %r6;
    @%p1 bra RED_NEXT;
    add.u32 %r7, %r2, %r6;
    mul.wide.u32 %rd8, %r7, 4;
    add.u64 %rd9, %rd6, %rd8;
    ld.shared.f32 %f2, [%rd9];
    add.u64 %rd10, %rd7, %rd8;
    ld.shared.u32 %r8, [%rd10];
    mul.wide.u32 %rd8, %r2, 4;
    add.u64 %rd9, %rd6, %rd8;
    ld.shared.f32 %f3, [%rd9];
    add.u64 %rd10, %rd7, %rd8;
    ld.shared.u32 %r9, [%rd10];
    setp.gt.f32 %p2, %f2, %f3;
    setp.eq.f32 %p3, %f2, %f3;
    setp.lt.u32 %p4, %r8, %r9;
    and.pred %p5, %p3, %p4;
    or.pred %p2, %p2, %p5;
    @!%p2 bra RED_NEXT;
    st.shared.f32 [%rd9], %f2;
    st.shared.u32 [%rd10], %r8;
RED_NEXT:
    bar.sync 0;
    shr.u32 %r6, %r6, 1;
    setp.gt.u32 %p1, %r6, 0;
    @%p1 bra REDUCE;

    setp.ne.u32 %p1, %r2, 0;
    @%p1 bra EXIT;
    ld.shared.f32 %f4, [%rd6];
    ld.shared.u32 %r10, [%rd7];
    st.global.f32 [%rd2], %f4;
    st.global.u32 [%rd3], %r10;
EXIT:
    ret;
}`

func ensureArgmaxF32() error {
	if fnArgmaxF32 != 0 {
		return nil
	}
	if !SgemmReady() {
		return fmt.Errorf("GPU not available")
	}
	fn, err := LoadPTX(argmaxF32PTX, "argmax_f32")
	if err != nil {
		return err
	}
	fnArgmaxF32 = fn
	return nil
}

// ArgmaxF32 returns the max value/index for a GPU-resident float32 buffer.
func ArgmaxF32(buf *Buffer, n int) (int, float32, error) {
	if buf == nil {
		return 0, 0, fmt.Errorf("nil GPU buffer")
	}
	if n <= 0 {
		return 0, 0, fmt.Errorf("invalid argmax length %d", n)
	}
	if n > buf.Size/4 {
		return 0, 0, fmt.Errorf("argmax length %d exceeds buffer floats %d", n, buf.Size/4)
	}
	if err := ensureArgmaxF32(); err != nil {
		return 0, 0, err
	}
	outVal, err := Malloc(1)
	if err != nil {
		return 0, 0, err
	}
	defer outVal.Free()
	outIdx, err := Malloc(1)
	if err != nil {
		return 0, 0, err
	}
	defer outIdx.Free()
	nn := uint32(n)
	if err := LaunchKernel(fnArgmaxF32, 1, 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&buf.Ptr), unsafe.Pointer(&outVal.Ptr), unsafe.Pointer(&outIdx.Ptr), unsafe.Pointer(&nn)); err != nil {
		return 0, 0, err
	}
	val := []float32{float32(math.Inf(-1))}
	if err := outVal.Download(val); err != nil {
		return 0, 0, err
	}
	idxBits := []uint32{0}
	EnsureContext()
	if r := cuMemcpyDtoH(unsafe.Pointer(&idxBits[0]), outIdx.Ptr, 4); r != CUDA_SUCCESS {
		return 0, 0, fmt.Errorf("cuMemcpyDtoH argmax idx: error %d", r)
	}
	return int(idxBits[0]), val[0], nil
}
