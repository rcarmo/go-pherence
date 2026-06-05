//go:build riscv64

package whisper

import (
	"os"
	"sync"
	"time"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

func nowNs() int64 { return time.Now().UnixNano() }

// useInt8 enables the int8 IME (vmadot) path for qualifying encoder GEMMs.
var useInt8 = os.Getenv("WHISPER_INT8") != ""

// attnInt8 routes the two per-head attention GEMMs (Qh@Kh^T and scores@Vh)
// through the int8 IME path. Enabled by WHISPER_INT8 (validated byte-identical
// on jfk); set WHISPER_INT8_NOATTN=1 to keep attention in F32 for debugging.
var attnInt8 = (useInt8 || os.Getenv("WHISPER_INT8_ATTN") != "") && os.Getenv("WHISPER_INT8_NOATTN") == ""

// attnInt8Head computes one attention head with int8 IME GEMMs, writing the
// row-softmaxed scores into `scores` ([seqQ,seqKV]) and the head output into
// `outh` ([seqQ,headDim]). All other slices are caller-provided scratch reused
// across heads in the same goroutine. Padded rows/cols of the int8 scratch must
// be pre-zeroed (fresh make) and are never written, so they stay zero.
func attnInt8Head(scores, outh, qh, kh, vh []float32, seqQ, seqKV, headDim int, scale float32,
	mq, nk, kpad int, qi8 []int8, qs []float32, ki8 []int8, ks []float32, cqk []int32, qp, kp []int8,
	vhT []float32, vti8 []int8, vts []float32, sPad []float32, si8 []int8, ss []float32, cout []int32, sp, vtp []int8) {

	// GEMM1: cqk = Qh @ Kh^T (int8); scores[i,j] = cqk*qs[i]*ks[j]*scale.
	quantizeRowsInto(qh, seqQ, headDim, qi8, qs)
	quantizeRowsInto(kh, seqKV, headDim, ki8, ks)
	ime2.PackTilesInto(qi8, mq, headDim, qp)
	ime2.PackTilesInto(ki8, nk, headDim, kp)
	ime2.GemmINT8Packed(mq, nk, headDim, qp, kp, cqk)
	for i := 0; i < seqQ; i++ {
		qsi := qs[i] * scale
		crow := i * nk
		srow := i * seqKV
		for j := 0; j < seqKV; j++ {
			scores[srow+j] = float32(cqk[crow+j]) * qsi * ks[j]
		}
	}

	// Row softmax.
	for tq := 0; tq < seqQ; tq++ {
		softmax(scores[tq*seqKV : (tq+1)*seqKV])
	}

	// GEMM2: outh = scores @ Vh = scores @ (Vh^T)^T; Vh^T is [headDim, seqKV].
	for d := 0; d < headDim; d++ {
		drow := d * kpad
		for t := 0; t < seqKV; t++ {
			vhT[drow+t] = vh[t*headDim+d]
		}
	}
	quantizeRowsInto(vhT, headDim, kpad, vti8, vts)
	for i := 0; i < seqQ; i++ {
		copy(sPad[i*kpad:i*kpad+seqKV], scores[i*seqKV:(i+1)*seqKV])
	}
	quantizeRowsInto(sPad, seqQ, kpad, si8, ss)
	ime2.PackTilesInto(si8, mq, kpad, sp)
	ime2.PackTilesInto(vti8, headDim, kpad, vtp)
	ime2.GemmINT8Packed(mq, headDim, kpad, sp, vtp, cout)
	for i := 0; i < seqQ; i++ {
		ssi := ss[i]
		crow := i * headDim
		for d := 0; d < headDim; d++ {
			outh[crow+d] = float32(cout[crow+d]) * ssi * vts[d]
		}
	}
}

// int8 phase timers (ns), summed across calls; printed under WHISPER_DEBUG.
var (
	i8QuantNs int64
	i8PackNs  int64
	i8GemmNs  int64
	i8DeqNs   int64
)

// int8Eligible reports whether a linear of these dims can use the packed int8
// IME GEMM. PackTiles needs rows%4==0 and K%8==0; M is padded to a multiple of
// 4 internally, so only N and K are constrained here.
func int8Eligible(inDim, outDim int) bool {
	return useInt8 && outDim%4 == 0 && inDim%8 == 0 && inDim >= 8
}

type packedWeight struct {
	wp []int8   // PackTiles(quant(weight)) [outDim, inDim]
	ws []float32 // per-row (output channel) scale
}

var i8WeightCache sync.Map // key: uintptr(&weight[0]) -> *packedWeight

func getPackedWeight(weight []float32, outDim, inDim int) *packedWeight {
	key := uintptr(unsafe.Pointer(&weight[0]))
	if v, ok := i8WeightCache.Load(key); ok {
		return v.(*packedWeight)
	}
	wi8 := make([]int8, outDim*inDim)
	ws := make([]float32, outDim)
	quantizeRowsInto(weight, outDim, inDim, wi8, ws)
	pw := &packedWeight{wp: ime2.PackTiles(wi8, outDim, inDim), ws: ws}
	i8WeightCache.Store(key, pw)
	return pw
}

// quantizeRowsInto fills q (rows*K int8) and sc (rows scales) with a per-row
// symmetric int8 quantization of x, using the RVV max-abs + quantize kernels.
// Dequant scale is sc[r]=maxAbs/127; the RVV kernel multiplies by 127/maxAbs.
func quantizeRowsInto(x []float32, rows, K int, q []int8, sc []float32) {
	for r := 0; r < rows; r++ {
		base := r * K
		row := x[base : base+K]
		m := ime2.FindMaxAbsRVV(row)
		if m == 0 {
			sc[r] = 0
			for k := 0; k < K; k++ {
				q[base+k] = 0
			}
			continue
		}
		sc[r] = m / 127.0
		ime2.QuantizeF32ToI8RVV(row, 127.0/m, q[base:base+K])
	}
}

// linearForwardInt8 computes out[seqLen,outDim] = x @ weight^T + bias using the
// int8 IME GEMM. Weights are quantized+packed once and cached; activations are
// quantized per call. M is zero-padded to a multiple of 4 for PackTiles.
func linearForwardInt8(x, weight, bias []float32, seqLen, inDim, outDim int) []float32 {
	K, N := inDim, outDim
	Mp := (seqLen + 3) &^ 3

	tq := nowNs()
	xi8 := make([]int8, Mp*K) // padded rows stay zero
	xs := make([]float32, Mp)
	quantizeRowsInto(x, seqLen, K, xi8, xs)
	i8QuantNs += nowNs() - tq

	tp := nowNs()
	xp := ime2.PackTiles(xi8, Mp, K)
	i8PackNs += nowNs() - tp

	pw := getPackedWeight(weight, N, K)

	C := make([]int32, Mp*N)
	tg := nowNs()
	if Mp <= 8 {
		// Decode (M padded 1->4): the GEMM is one tile-row, so threading is pure
		// goroutine-spawn overhead (~256 linear calls per decoded token).
		ime2.GemmINT8Packed(Mp, N, K, xp, pw.wp, C)
	} else {
		ime2.GemmINT8PackedParallel(Mp, N, K, xp, pw.wp, C, linearWorkers)
	}
	i8GemmNs += nowNs() - tg

	td := nowNs()
	out := make([]float32, seqLen*outDim)
	ws := pw.ws
	for i := 0; i < seqLen; i++ {
		xsi := xs[i]
		cRow := i * N
		oRow := i * outDim
		if xsi == 0 {
			if bias != nil {
				for j := 0; j < N && j < len(bias); j++ {
					out[oRow+j] = bias[j]
				}
			}
			continue
		}
		for j := 0; j < N; j++ {
			v := float32(C[cRow+j]) * xsi * ws[j]
			if bias != nil && j < len(bias) {
				v += bias[j]
			}
			out[oRow+j] = v
		}
	}
	i8DeqNs += nowNs() - td
	return out
}
