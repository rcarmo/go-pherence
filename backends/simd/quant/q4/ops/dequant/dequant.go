package dequant

import (
	"runtime"
	"sync"
)

// DequantGPTQ dequantizes a GPTQ INT4 weight tensor to float32.
//
// Parameters:
//
//	qweight: [inFeatures/8, outFeatures] packed int32 (8 x 4-bit per int32)
//	scales:  [numGroups, outFeatures] float32 (already converted from F16)
//	qzeros:  [numGroups, outFeatures/8] packed int32
//	gIdx:    [inFeatures] int32 group index per input
//	inFeatures, outFeatures: weight dimensions
//	sym: true if symmetric quantization (zero point = 8 for 4-bit)
//
// Returns: [outFeatures, inFeatures] float32 weight matrix (row-major, row=output)
func Dequant(qweight, qzeros, gIdx []int32, scales []float32,
	inFeatures, outFeatures int, sym bool) []float32 {
	if err := Validate(qweight, qzeros, gIdx, scales, inFeatures, outFeatures, sym); err != nil {
		return nil
	}

	outLen, ok := checkedMulInt(outFeatures, inFeatures)
	if !ok {
		return nil
	}
	out := make([]float32, outLen)
	dequantTo(out, qweight, qzeros, gIdx, scales, inFeatures, outFeatures, sym)
	return out
}

// DequantTo dequantizes into a caller-owned output buffer. The output layout is
// [outFeatures, inFeatures] row-major. It returns false on malformed inputs.
func DequantTo(out []float32, qweight, qzeros, gIdx []int32, scales []float32,
	inFeatures, outFeatures int, sym bool) bool {
	if err := Validate(qweight, qzeros, gIdx, scales, inFeatures, outFeatures, sym); err != nil {
		return false
	}
	outLen, ok := checkedMulInt(outFeatures, inFeatures)
	if !ok || len(out) < outLen {
		return false
	}
	dequantTo(out[:outLen], qweight, qzeros, gIdx, scales, inFeatures, outFeatures, sym)
	return true
}

func dequantTo(out []float32, qweight, qzeros, gIdx []int32, scales []float32,
	inFeatures, outFeatures int, sym bool) {
	if sym {
		dequantSymTo(out, qweight, gIdx, scales, inFeatures, outFeatures)
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if outFeatures < 1024 || workers <= 1 {
		dequantRows(out, qweight, qzeros, gIdx, scales, inFeatures, outFeatures, 0, outFeatures)
		return
	}
	if workers > outFeatures {
		workers = outFeatures
	}
	chunkSize := (outFeatures + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		jStart := w * chunkSize
		jEnd := jStart + chunkSize
		if jEnd > outFeatures {
			jEnd = outFeatures
		}
		if jStart >= jEnd {
			continue
		}
		wg.Add(1)
		go func(jStart, jEnd int) {
			defer wg.Done()
			dequantRows(out, qweight, qzeros, gIdx, scales, inFeatures, outFeatures, jStart, jEnd)
		}(jStart, jEnd)
	}
	wg.Wait()
}

func dequantRows(out []float32, qweight, qzeros, gIdx []int32, scales []float32,
	inFeatures, outFeatures, jStart, jEnd int) {
	for j := jStart; j < jEnd; j++ {
		outRow := out[j*inFeatures : (j+1)*inFeatures]
		zPackIdx := j / 8
		zBitIdx := uint(j%8) * 4
		for packIdx := 0; packIdx < inFeatures/8; packIdx++ {
			qwRow := qweight[packIdx*outFeatures : (packIdx+1)*outFeatures]
			for bit := 0; bit < 8; bit++ {
				i := packIdx*8 + bit
				g := int(gIdx[i])
				qw := (qwRow[j] >> (uint(bit) * 4)) & 0xF
				qz := (qzeros[g*(outFeatures/8)+zPackIdx] >> zBitIdx) & 0xF
				outRow[i] = scales[g*outFeatures+j] * float32(qw-qz)
			}
		}
	}
}

// DequantGPTQSym is an optimized parallel symmetric dequantization (zero point = 8).
func DequantSym(qweight, gIdx []int32, scales []float32,
	inFeatures, outFeatures int) []float32 {
	if err := ValidateSym(qweight, gIdx, scales, inFeatures, outFeatures); err != nil {
		return nil
	}

	outLen, ok := checkedMulInt(outFeatures, inFeatures)
	if !ok {
		return nil
	}
	out := make([]float32, outLen)
	if !DequantSymTo(out, qweight, gIdx, scales, inFeatures, outFeatures) {
		return nil
	}
	return out
}

// DequantSymTo dequantizes a symmetric GPTQ tensor into caller-owned storage.
// The output layout is [outFeatures, inFeatures] row-major. It returns false on
// malformed inputs or undersized output.
func DequantSymTo(out []float32, qweight, gIdx []int32, scales []float32,
	inFeatures, outFeatures int) bool {
	if err := ValidateSym(qweight, gIdx, scales, inFeatures, outFeatures); err != nil {
		return false
	}
	outLen, ok := checkedMulInt(outFeatures, inFeatures)
	if !ok || len(out) < outLen {
		return false
	}
	dequantSymTo(out[:outLen], qweight, gIdx, scales, inFeatures, outFeatures)
	return true
}

func dequantSymTo(out []float32, qweight, gIdx []int32, scales []float32,
	inFeatures, outFeatures int) {
	// Parallelize across output rows only when row count is large enough to
	// amortize goroutine overhead. Respect GOMAXPROCS so constrained runtimes do
	// not oversubscribe logical CPUs.
	nWorkers := runtime.GOMAXPROCS(0)
	if outFeatures < 1024 || nWorkers <= 1 {
		dequantSymRows(out, qweight, gIdx, scales, inFeatures, outFeatures, 0, outFeatures)
		return
	}
	if nWorkers > outFeatures {
		nWorkers = outFeatures
	}
	var wg sync.WaitGroup
	chunkSize := (outFeatures + nWorkers - 1) / nWorkers

	for w := 0; w < nWorkers; w++ {
		jStart := w * chunkSize
		jEnd := jStart + chunkSize
		if jEnd > outFeatures {
			jEnd = outFeatures
		}
		if jStart >= jEnd {
			continue
		}
		wg.Add(1)
		go func(jStart, jEnd int) {
			defer wg.Done()
			dequantSymRows(out, qweight, gIdx, scales, inFeatures, outFeatures, jStart, jEnd)
		}(jStart, jEnd)
	}
	wg.Wait()
}

func dequantSymRows(out []float32, qweight, gIdx []int32, scales []float32, inFeatures, outFeatures, jStart, jEnd int) {
	nPacks := inFeatures / 8
	for packIdx := 0; packIdx < nPacks; packIdx++ {
		qwRow := qweight[packIdx*outFeatures : (packIdx+1)*outFeatures]
		for bit := 0; bit < 8; bit++ {
			i := packIdx*8 + bit
			g := int(gIdx[i])
			bitIdx := uint(bit) * 4
			scaleRow := scales[g*outFeatures : (g+1)*outFeatures]

			for j := jStart; j < jEnd; j++ {
				qw := (qwRow[j] >> bitIdx) & 0xF
				out[j*inFeatures+i] = scaleRow[j] * float32(qw-8)
			}
		}
	}
}
