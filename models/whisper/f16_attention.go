package whisper

import (
	"os"
	"time"

	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
	"github.com/rcarmo/go-pherence/half"
)

// attnF16 routes encoder attention GEMMs through the K3-native RVV/Zvfh FP16
// path. It is opt-in while the kernel and transcript quality are being tuned.
var attnF16 = os.Getenv("WHISPER_FP16_ATTN") != ""

var (
	f16PackNs int64
	f16GemmNs int64
)

func f32ToF16RowsPadded(src []float32, rows, cols, colsPad int, dst []uint16) {
	if cols == colsPad {
		rvv.F32ToF16RVV(dst[:rows*cols], src[:rows*cols])
		return
	}
	for r := 0; r < rows; r++ {
		so := r * cols
		do := r * colsPad
		rvv.F32ToF16RVV(dst[do:do+cols], src[so:so+cols])
		for c := cols; c < colsPad; c++ {
			dst[do+c] = 0
		}
	}
}

func transposeF32ToF16RowsPadded(src []float32, rows, cols, rowsPad int, dst []uint16) {
	// src is [rows, cols]; dst is [cols, rowsPad], i.e. transposed-B layout for rvv.GemmF16Outer32.
	for c := 0; c < cols; c++ {
		do := c * rowsPad
		for r := 0; r < rows; r++ {
			dst[do+r] = half.F32ToF16(src[r*cols+c])
		}
		for r := rows; r < rowsPad; r++ {
			dst[do+r] = 0
		}
	}
}

func attnF16Head(scores, outh, qh, kh, vh []float32, seqQ, seqKV, headDim int, scale float32,
	kpad int, qf16, kf16, sf16, vtf16 []uint16, cqk []float32, kp, vtp []uint16) {

	tp := nowNs()
	f32ToF16RowsPadded(qh, seqQ, headDim, headDim, qf16)
	f32ToF16RowsPadded(kh, seqKV, headDim, headDim, kf16)
	kp = rvv.PackBF16N32(kf16, kpad, headDim)
	f16PackNs += nowNs() - tp

	tg := nowNs()
	rvv.GemmF16Outer32(qf16, kp, cqk, seqQ, kpad, headDim, linearWorkers)
	f16GemmNs += nowNs() - tg
	for i := 0; i < seqQ; i++ {
		crow := i * kpad
		srow := i * seqKV
		for j := 0; j < seqKV; j++ {
			scores[srow+j] = cqk[crow+j] * scale
		}
	}

	for tq := 0; tq < seqQ; tq++ {
		softmax(scores[tq*seqKV : (tq+1)*seqKV])
	}

	tp = nowNs()
	f32ToF16RowsPadded(scores, seqQ, seqKV, kpad, sf16)
	transposeF32ToF16RowsPadded(vh, seqKV, headDim, kpad, vtf16)
	vtp = rvv.PackBF16N32(vtf16, headDim, kpad)
	f16PackNs += nowNs() - tp

	tg = nowNs()
	rvv.GemmF16Outer32(sf16, vtp, outh, seqQ, headDim, kpad, linearWorkers)
	f16GemmNs += nowNs() - tg
}

func resetF16Timers() { f16PackNs, f16GemmNs = 0, 0 }

func f16TimingLine() string {
	return "[fp16] pack=" + time.Duration(f16PackNs).String() + " gemm=" + time.Duration(f16GemmNs).String()
}
