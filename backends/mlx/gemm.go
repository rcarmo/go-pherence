package mlx

import (
	"runtime"
	"sync"

	"github.com/rcarmo/go-pherence/internal/checked"
)

const (
	gemm4BatchTile   = 4
	gemm4StackGroups = 256
)

// Gemm performs a batched MLX quantized matrix multiply.
//
// x is [batch, inDim], out is [batch, outDim], both row-major. Batch=1 keeps
// the GEMV fast path. For 4-bit affine weights with 8-value packed groups, GEMM
// processes up to four activation rows together so each packed weight is
// unpacked once per output row/group and reused across the batch tile.
func Gemm(out, x []float32, batch int, qw *QuantWeight) bool {
	if batch <= 0 || ValidateQuantWeight(qw) != nil {
		return false
	}
	xLen, okX := checked.MulInt(batch, qw.InDim)
	outLen, okOut := checked.MulInt(batch, qw.OutDim)
	if !okX || !okOut || len(x) < xLen || len(out) < outLen {
		return false
	}
	if batch == 1 {
		return GemvTo(out[:qw.OutDim], x[:qw.InDim], qw)
	}
	if qw.Bits == 4 && qw.GroupSize%8 == 0 {
		return gemm4Batched(out[:outLen], x[:xLen], batch, qw)
	}
	return gemmRepeated(out[:outLen], x[:xLen], batch, qw)
}

func gemmRepeated(out, x []float32, batch int, qw *QuantWeight) bool {
	if batch < 2 || qw.OutDim*qw.InDim < 4096 {
		for b := 0; b < batch; b++ {
			if !GemvTo(out[b*qw.OutDim:(b+1)*qw.OutDim], x[b*qw.InDim:(b+1)*qw.InDim], qw) {
				return false
			}
		}
		return true
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > batch {
		workers = batch
	}
	chunk := (batch + workers - 1) / workers
	ok := true
	var mu sync.Mutex
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > batch {
			end = batch
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for b := start; b < end; b++ {
				if !GemvTo(out[b*qw.OutDim:(b+1)*qw.OutDim], x[b*qw.InDim:(b+1)*qw.InDim], qw) {
					mu.Lock()
					ok = false
					mu.Unlock()
					return
				}
			}
		}(start, end)
	}
	wg.Wait()
	return ok
}

func gemm4Batched(out, x []float32, batch int, qw *QuantWeight) bool {
	outDim := qw.OutDim
	inDim := qw.InDim
	for b := 0; b < batch; {
		batchTile := batch - b
		if batchTile > gemm4BatchTile {
			batchTile = gemm4BatchTile
		}
		if !gemm4Tile(out[b*outDim:(b+batchTile)*outDim], x[b*inDim:(b+batchTile)*inDim], batchTile, qw) {
			return false
		}
		b += batchTile
	}
	return true
}

func gemm4Tile(out, x []float32, batchTile int, qw *QuantWeight) bool {
	if batchTile == 1 {
		return GemvTo(out[:qw.OutDim], x[:qw.InDim], qw)
	}

	need := batchTile * qw.Groups
	var groupXSumsStack [gemm4BatchTile * gemm4StackGroups]float32
	groupXSums := groupXSumsStack[:]
	if qw.Groups > gemm4StackGroups {
		groupXSums = make([]float32, need)
	} else {
		groupXSums = groupXSums[:need]
	}
	gemm4BatchXSums(x, batchTile, qw, groupXSums)
	gemm4Rows(out, x, batchTile, qw, groupXSums)
	return true
}

// gemm4BatchXSums precomputes the per-batch, per-group sums of x used by the
// affine bias term. For the common case of groups<=256 and batch tiles<=4,
// callers can keep dst on the stack; larger group counts require batch*groups
// scratch floats.
func gemm4BatchXSums(x []float32, batch int, qw *QuantWeight, dst []float32) {
	for b := 0; b < batch; b++ {
		gemv4XSums(x[b*qw.InDim:(b+1)*qw.InDim], qw, dst[b*qw.Groups:(b+1)*qw.Groups])
	}
}

func gemm4Rows(out, x []float32, batchTile int, qw *QuantWeight, groupXSums []float32) {
	switch batchTile {
	case 2:
		gemm4Rows2(out, x, qw, groupXSums)
	case 3:
		gemm4Rows3(out, x, qw, groupXSums)
	case 4:
		gemm4Rows4(out, x, qw, groupXSums)
	default:
		gemv4Rows(out[:qw.OutDim], x[:qw.InDim], qw, groupXSums[:qw.Groups], 0, qw.OutDim)
	}
}

func gemm4Rows2(out, x []float32, qw *QuantWeight, groupXSums []float32) {
	inDim := qw.InDim
	outDim := qw.OutDim
	groups := qw.Groups
	packedPerRow := inDim / 8
	packsPerGroup := qw.GroupSize / 8

	x0 := x[:inDim]
	x1 := x[inDim : 2*inDim]
	out0 := out[:outDim]
	out1 := out[outDim : 2*outDim]
	xsum0 := groupXSums[:groups]
	xsum1 := groupXSums[groups : 2*groups]

	for row := 0; row < outDim; row++ {
		packedOff := row * packedPerRow
		scaleOff := row * groups
		s0 := float32(0)
		s1 := float32(0)
		for g := 0; g < groups; g++ {
			scale := qw.Scales[scaleOff+g]
			bias := qw.Biases[scaleOff+g]
			packBase := packedOff + g*packsPerGroup
			xBase := g * qw.GroupSize
			g0 := float32(0)
			g1 := float32(0)
			for p := 0; p < packsPerGroup; p++ {
				packed := qw.Weight[packBase+p]
				xi := xBase + p*8
				w0 := float32(packed & 0xF)
				w1 := float32((packed >> 4) & 0xF)
				w2 := float32((packed >> 8) & 0xF)
				w3 := float32((packed >> 12) & 0xF)
				w4 := float32((packed >> 16) & 0xF)
				w5 := float32((packed >> 20) & 0xF)
				w6 := float32((packed >> 24) & 0xF)
				w7 := float32((packed >> 28) & 0xF)
				g0 += w0*x0[xi] + w1*x0[xi+1] + w2*x0[xi+2] + w3*x0[xi+3] + w4*x0[xi+4] + w5*x0[xi+5] + w6*x0[xi+6] + w7*x0[xi+7]
				g1 += w0*x1[xi] + w1*x1[xi+1] + w2*x1[xi+2] + w3*x1[xi+3] + w4*x1[xi+4] + w5*x1[xi+5] + w6*x1[xi+6] + w7*x1[xi+7]
			}
			s0 += g0*scale + xsum0[g]*bias
			s1 += g1*scale + xsum1[g]*bias
		}
		out0[row] = s0
		out1[row] = s1
	}
}

func gemm4Rows3(out, x []float32, qw *QuantWeight, groupXSums []float32) {
	inDim := qw.InDim
	outDim := qw.OutDim
	groups := qw.Groups
	packedPerRow := inDim / 8
	packsPerGroup := qw.GroupSize / 8

	x0 := x[:inDim]
	x1 := x[inDim : 2*inDim]
	x2 := x[2*inDim : 3*inDim]
	out0 := out[:outDim]
	out1 := out[outDim : 2*outDim]
	out2 := out[2*outDim : 3*outDim]
	xsum0 := groupXSums[:groups]
	xsum1 := groupXSums[groups : 2*groups]
	xsum2 := groupXSums[2*groups : 3*groups]

	for row := 0; row < outDim; row++ {
		packedOff := row * packedPerRow
		scaleOff := row * groups
		s0 := float32(0)
		s1 := float32(0)
		s2 := float32(0)
		for g := 0; g < groups; g++ {
			scale := qw.Scales[scaleOff+g]
			bias := qw.Biases[scaleOff+g]
			packBase := packedOff + g*packsPerGroup
			xBase := g * qw.GroupSize
			g0 := float32(0)
			g1 := float32(0)
			g2 := float32(0)
			for p := 0; p < packsPerGroup; p++ {
				packed := qw.Weight[packBase+p]
				xi := xBase + p*8
				w0 := float32(packed & 0xF)
				w1 := float32((packed >> 4) & 0xF)
				w2 := float32((packed >> 8) & 0xF)
				w3 := float32((packed >> 12) & 0xF)
				w4 := float32((packed >> 16) & 0xF)
				w5 := float32((packed >> 20) & 0xF)
				w6 := float32((packed >> 24) & 0xF)
				w7 := float32((packed >> 28) & 0xF)
				g0 += w0*x0[xi] + w1*x0[xi+1] + w2*x0[xi+2] + w3*x0[xi+3] + w4*x0[xi+4] + w5*x0[xi+5] + w6*x0[xi+6] + w7*x0[xi+7]
				g1 += w0*x1[xi] + w1*x1[xi+1] + w2*x1[xi+2] + w3*x1[xi+3] + w4*x1[xi+4] + w5*x1[xi+5] + w6*x1[xi+6] + w7*x1[xi+7]
				g2 += w0*x2[xi] + w1*x2[xi+1] + w2*x2[xi+2] + w3*x2[xi+3] + w4*x2[xi+4] + w5*x2[xi+5] + w6*x2[xi+6] + w7*x2[xi+7]
			}
			s0 += g0*scale + xsum0[g]*bias
			s1 += g1*scale + xsum1[g]*bias
			s2 += g2*scale + xsum2[g]*bias
		}
		out0[row] = s0
		out1[row] = s1
		out2[row] = s2
	}
}

func gemm4Rows4(out, x []float32, qw *QuantWeight, groupXSums []float32) {
	inDim := qw.InDim
	outDim := qw.OutDim
	groups := qw.Groups
	packedPerRow := inDim / 8
	packsPerGroup := qw.GroupSize / 8

	x0 := x[:inDim]
	x1 := x[inDim : 2*inDim]
	x2 := x[2*inDim : 3*inDim]
	x3 := x[3*inDim : 4*inDim]
	out0 := out[:outDim]
	out1 := out[outDim : 2*outDim]
	out2 := out[2*outDim : 3*outDim]
	out3 := out[3*outDim : 4*outDim]
	xsum0 := groupXSums[:groups]
	xsum1 := groupXSums[groups : 2*groups]
	xsum2 := groupXSums[2*groups : 3*groups]
	xsum3 := groupXSums[3*groups : 4*groups]

	for row := 0; row < outDim; row++ {
		packedOff := row * packedPerRow
		scaleOff := row * groups
		s0 := float32(0)
		s1 := float32(0)
		s2 := float32(0)
		s3 := float32(0)
		for g := 0; g < groups; g++ {
			scale := qw.Scales[scaleOff+g]
			bias := qw.Biases[scaleOff+g]
			packBase := packedOff + g*packsPerGroup
			xBase := g * qw.GroupSize
			g0 := float32(0)
			g1 := float32(0)
			g2 := float32(0)
			g3 := float32(0)
			for p := 0; p < packsPerGroup; p++ {
				packed := qw.Weight[packBase+p]
				xi := xBase + p*8
				w0 := float32(packed & 0xF)
				w1 := float32((packed >> 4) & 0xF)
				w2 := float32((packed >> 8) & 0xF)
				w3 := float32((packed >> 12) & 0xF)
				w4 := float32((packed >> 16) & 0xF)
				w5 := float32((packed >> 20) & 0xF)
				w6 := float32((packed >> 24) & 0xF)
				w7 := float32((packed >> 28) & 0xF)
				g0 += w0*x0[xi] + w1*x0[xi+1] + w2*x0[xi+2] + w3*x0[xi+3] + w4*x0[xi+4] + w5*x0[xi+5] + w6*x0[xi+6] + w7*x0[xi+7]
				g1 += w0*x1[xi] + w1*x1[xi+1] + w2*x1[xi+2] + w3*x1[xi+3] + w4*x1[xi+4] + w5*x1[xi+5] + w6*x1[xi+6] + w7*x1[xi+7]
				g2 += w0*x2[xi] + w1*x2[xi+1] + w2*x2[xi+2] + w3*x2[xi+3] + w4*x2[xi+4] + w5*x2[xi+5] + w6*x2[xi+6] + w7*x2[xi+7]
				g3 += w0*x3[xi] + w1*x3[xi+1] + w2*x3[xi+2] + w3*x3[xi+3] + w4*x3[xi+4] + w5*x3[xi+5] + w6*x3[xi+6] + w7*x3[xi+7]
			}
			s0 += g0*scale + xsum0[g]*bias
			s1 += g1*scale + xsum1[g]*bias
			s2 += g2*scale + xsum2[g]*bias
			s3 += g3*scale + xsum3[g]*bias
		}
		out0[row] = s0
		out1[row] = s1
		out2[row] = s2
		out3[row] = s3
	}
}
