package whisper

import (
	"os"
	"runtime"
	"strconv"
	"sync"

	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// linearWorkers controls how many goroutines split the seqLen (row) dimension
// of the batched encoder GEMM. Defaults to min(GOMAXPROCS, 4): the encoder is
// memory-bandwidth bound so ~2-4 cores already saturate the K1's memory system,
// and driving all 8 cores at full RVV load was observed to brown-out/reboot the
// board. Override with WHISPER_THREADS for experiments.
var linearWorkers = func() int {
	if v := os.Getenv("WHISPER_THREADS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	n := runtime.GOMAXPROCS(0)
	if n > 4 {
		n = 4
	}
	if n < 1 {
		n = 1
	}
	return n
}()

// blockM is the frame-tile height for linearRowBlock. blockM*inDim*4 bytes of
// activations must fit in L2 so they stay resident across the outDim sweep,
// while each weight row stays in L1 across the blockM inner iterations. 32 keeps
// 32*1280*4 = 160 KiB resident. Override with WHISPER_BLOCKM.
var blockM = func() int {
	if v := os.Getenv("WHISPER_BLOCKM"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return 32
}()

// linearForwardOpt computes out = x @ W^T + bias.
// x: [seqLen, inDim], W: [outDim, inDim], bias: [outDim]. Returns [seqLen, outDim].
//
// The seqLen=1 path (decoder, single token) uses the RVV dot product directly.
// The batched path (encoder, seqLen up to 1500) is the dominant cost in the
// pipeline; rows are independent, so it is split across linearWorkers goroutines
// and each block runs the RVV SGEMM. On the 8-core K1 this is the largest lever,
// since the per-frame GEMM is bound by per-cell orchestration and weight-stream
// bandwidth rather than raw dot FLOPs.
func linearForwardOpt(x, weight, bias []float32, seqLen, inDim, outDim int) []float32 {
	out := make([]float32, seqLen*outDim)

	// Single-token (decoder): RVV dot product per output cell, split across
	// workers for wide projections (FFN, LM head) since per-token decode is
	// otherwise fully serial.
	if seqLen == 1 {
		compute := func(oStart, oEnd int) {
			for o := oStart; o < oEnd; o++ {
				wOff := o * inDim
				sum := simdrt.Sdot(x[:inDim], weight[wOff:wOff+inDim])
				if bias != nil && o < len(bias) {
					sum += bias[o]
				}
				out[o] = sum
			}
		}
		nw := linearWorkers
		if nw > 1 && outDim >= 512 {
			chunk := (outDim + nw - 1) / nw
			var wg sync.WaitGroup
			for os := 0; os < outDim; os += chunk {
				oe := os + chunk
				if oe > outDim {
					oe = outDim
				}
				wg.Add(1)
				go func(a, b int) {
					defer wg.Done()
					compute(a, b)
				}(os, oe)
			}
			wg.Wait()
		} else {
			compute(0, outDim)
		}
		return out
	}

	nw := linearWorkers
	if nw > seqLen {
		nw = seqLen
	}
	// int8 IME path for qualifying batched GEMMs (encoder).
	if int8Eligible(inDim, outDim) {
		return linearForwardInt8(x, weight, bias, seqLen, inDim, outDim)
	}
	// Small batches: not worth the goroutine fan-out.
	if nw <= 1 || seqLen < 4 {
		linearRowBlock(out, x, weight, bias, 0, seqLen, inDim, outDim)
		return out
	}

	chunk := (seqLen + nw - 1) / nw
	var wg sync.WaitGroup
	for s := 0; s < seqLen; s += chunk {
		e := s + chunk
		if e > seqLen {
			e = seqLen
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			linearRowBlock(out, x, weight, bias, s, e, inDim, outDim)
		}(s, e)
	}
	wg.Wait()
	return out
}

// linearRowBlock computes out[s:e] = x[s:e] @ W^T + bias for one row block.
//
// Frame-tiled: the frame (row) range is swept in tiles of blockM so that for
// each weight row the inner loop processes blockM frames while that weight row
// stays hot in L1, and the blockM x-rows stay resident in L2 across the whole
// outDim sweep. This converts the weight matrix from "re-streamed from DRAM once
// per frame" to "streamed once per blockM frames" — a ~blockM x cut in weight
// memory traffic, which is the encoder's actual bottleneck. The inner dot uses
// the RVV m4 kernel via simdrt.Sdot.
func linearRowBlock(out, x, weight, bias []float32, s, e, inDim, outDim int) {
	bm := blockM
	if bm < 1 {
		bm = 1
	}
	for ib := s; ib < e; ib += bm {
		ie := ib + bm
		if ie > e {
			ie = e
		}
		for j := 0; j < outDim; j++ {
			wRow := weight[j*inDim : j*inDim+inDim]
			var bj float32
			if bias != nil && j < len(bias) {
				bj = bias[j]
			}
			for i := ib; i < ie; i++ {
				xRow := x[i*inDim : i*inDim+inDim]
				out[i*outDim+j] = simdrt.Sdot(xRow, wRow) + bj
			}
		}
	}
}

// scalarLinearBlock is the portable fallback when the RVV SGEMM is unavailable.
func scalarLinearBlock(out, x, weight, bias []float32, m, inDim, outDim int) {
	const tile = 4
	for t := 0; t < m; t++ {
		xOff := t * inDim
		oOff := t * outDim
		o := 0
		for ; o+tile-1 < outDim; o += tile {
			var s0, s1, s2, s3 float32
			w0 := (o + 0) * inDim
			w1 := (o + 1) * inDim
			w2 := (o + 2) * inDim
			w3 := (o + 3) * inDim
			for d := 0; d < inDim; d++ {
				xd := x[xOff+d]
				s0 += xd * weight[w0+d]
				s1 += xd * weight[w1+d]
				s2 += xd * weight[w2+d]
				s3 += xd * weight[w3+d]
			}
			if bias != nil {
				s0 += bias[o+0]
				s1 += bias[o+1]
				s2 += bias[o+2]
				s3 += bias[o+3]
			}
			out[oOff+o+0] = s0
			out[oOff+o+1] = s1
			out[oOff+o+2] = s2
			out[oOff+o+3] = s3
		}
		for ; o < outDim; o++ {
			var sum float32
			wOff := o * inDim
			for d := 0; d < inDim; d++ {
				sum += x[xOff+d] * weight[wOff+d]
			}
			if bias != nil && o < len(bias) {
				sum += bias[o]
			}
			out[oOff+o] = sum
		}
	}
}
