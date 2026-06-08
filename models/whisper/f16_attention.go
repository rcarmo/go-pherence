package whisper

import (
	"math"
	"os"
	"strconv"
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

func f16HeadBatchSize(numHeads int) int {
	if v := os.Getenv("WHISPER_FP16_HEAD_BATCH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			if n > numHeads {
				n = numHeads
			}
			return n
		}
	}
	n := linearWorkers
	if n > numHeads {
		n = numHeads
	}
	if n < 1 {
		n = 1
	}
	return n
}

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
	kpad, gemmWorkers int, qf16, kf16, sf16, vtf16 []uint16, cqk []float32, kp, vtp []uint16) {

	tp := nowNs()
	f32ToF16RowsPadded(qh, seqQ, headDim, headDim, qf16)
	f32ToF16RowsPadded(kh, seqKV, headDim, headDim, kf16)
	kp = rvv.PackBF16N32(kf16, kpad, headDim)
	f16PackNs += nowNs() - tp

	tg := nowNs()
	rvv.GemmF16Outer32(qf16, kp, cqk, seqQ, kpad, headDim, gemmWorkers)
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
	rvv.GemmF16Outer32(sf16, vtp, outh, seqQ, headDim, kpad, gemmWorkers)
	f16GemmNs += nowNs() - tg
}

func resetF16Timers() { f16PackNs, f16GemmNs = 0, 0 }

func f16TimingLine() string {
	return "[fp16] pack=" + time.Duration(f16PackNs).String() + " gemm=" + time.Duration(f16GemmNs).String()
}

func fullAttentionF16(q, k, v []float32, seqQ, seqKV, numHeads, headDim int) []float32 {
	dModel := numHeads * headDim
	out := make([]float32, seqQ*dModel)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	kpad := (seqKV + 31) &^ 31
	batchSize := f16HeadBatchSize(numHeads)

	type hs struct {
		h            int
		qh, kh, vh   []float32
		scores, outh []float32
		qf16, kf16   []uint16
		sf16, vtf16  []uint16
		cqk          []float32
		kp, vtp      []uint16
	}

	for h0 := 0; h0 < numHeads; h0 += batchSize {
		h1 := h0 + batchSize
		if h1 > numHeads {
			h1 = numHeads
		}
		heads := make([]hs, h1-h0)
		qkSpecs := make([]rvv.GemmF16Outer32Spec, 0, len(heads))

		tp := nowNs()
		for bi, h := range makeRange(h0, h1) {
			hOff := h * headDim
			sx := &heads[bi]
			sx.h = h
			sx.qh = make([]float32, seqQ*headDim)
			sx.kh = make([]float32, seqKV*headDim)
			sx.vh = make([]float32, seqKV*headDim)
			sx.scores = make([]float32, seqQ*seqKV)
			sx.outh = make([]float32, seqQ*headDim)
			for t := 0; t < seqQ; t++ {
				copy(sx.qh[t*headDim:(t+1)*headDim], q[t*dModel+hOff:t*dModel+hOff+headDim])
			}
			for t := 0; t < seqKV; t++ {
				copy(sx.kh[t*headDim:(t+1)*headDim], k[t*dModel+hOff:t*dModel+hOff+headDim])
				copy(sx.vh[t*headDim:(t+1)*headDim], v[t*dModel+hOff:t*dModel+hOff+headDim])
			}
			sx.qf16 = make([]uint16, seqQ*headDim)
			sx.kf16 = make([]uint16, kpad*headDim)
			sx.cqk = make([]float32, seqQ*kpad)
			f32ToF16RowsPadded(sx.qh, seqQ, headDim, headDim, sx.qf16)
			f32ToF16RowsPadded(sx.kh, seqKV, headDim, headDim, sx.kf16[:seqKV*headDim])
			// Zero padded K rows, if any.
			for i := seqKV * headDim; i < len(sx.kf16); i++ {
				sx.kf16[i] = 0
			}
			sx.kp = rvv.PackBF16N32(sx.kf16, kpad, headDim)
			qkSpecs = append(qkSpecs, rvv.GemmF16Outer32Spec{A: sx.qf16, Bp: sx.kp, C: sx.cqk, M: seqQ, N: kpad, K: headDim})
		}
		f16PackNs += nowNs() - tp

		tg := nowNs()
		rvv.GemmF16Outer32Batch(linearWorkers, qkSpecs...)
		f16GemmNs += nowNs() - tg

		for bi := range heads {
			sx := &heads[bi]
			for i := 0; i < seqQ; i++ {
				crow := i * kpad
				srow := i * seqKV
				for j := 0; j < seqKV; j++ {
					sx.scores[srow+j] = sx.cqk[crow+j] * scale
				}
				softmax(sx.scores[srow : srow+seqKV])
			}
		}

		svSpecs := make([]rvv.GemmF16Outer32Spec, 0, len(heads))
		tp = nowNs()
		for bi := range heads {
			sx := &heads[bi]
			sx.sf16 = make([]uint16, seqQ*kpad)
			sx.vtf16 = make([]uint16, headDim*kpad)
			f32ToF16RowsPadded(sx.scores, seqQ, seqKV, kpad, sx.sf16)
			transposeF32ToF16RowsPadded(sx.vh, seqKV, headDim, kpad, sx.vtf16)
			sx.vtp = rvv.PackBF16N32(sx.vtf16, headDim, kpad)
			svSpecs = append(svSpecs, rvv.GemmF16Outer32Spec{A: sx.sf16, Bp: sx.vtp, C: sx.outh, M: seqQ, N: headDim, K: kpad})
		}
		f16PackNs += nowNs() - tp

		tg = nowNs()
		rvv.GemmF16Outer32Batch(linearWorkers, svSpecs...)
		f16GemmNs += nowNs() - tg

		for bi := range heads {
			sx := &heads[bi]
			hOff := sx.h * headDim
			for t := 0; t < seqQ; t++ {
				copy(out[t*dModel+hOff:t*dModel+hOff+headDim], sx.outh[t*headDim:(t+1)*headDim])
			}
		}
	}
	return out
}

func makeRange(a, b int) []int {
	out := make([]int, b-a)
	for i := range out {
		out[i] = a + i
	}
	return out
}
