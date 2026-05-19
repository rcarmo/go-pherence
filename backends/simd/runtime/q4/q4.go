package q4

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

func GemvSym(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	if err := ValidateGemvSym(out, x, qweight, gIdx, scales, inDim, outDim); err != nil {
		return
	}
	for j := 0; j < outDim; j++ {
		var sum float32
		for packIdx := 0; packIdx < inDim/8; packIdx++ {
			packed := qweight[packIdx*outDim+j]
			for bit := 0; bit < 8; bit++ {
				i := packIdx*8 + bit
				qw := (packed >> (uint(bit) * 4)) & 0xF
				g := int(gIdx[i])
				scale := scales[g*outDim+j]
				sum += x[i] * scale * float32(qw-8)
			}
		}
		out[j] = sum
	}
}

func ValidateGemvSym(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) error {
	if err := ValidateSym(qweight, gIdx, scales, inDim, outDim); err != nil {
		return fmt.Errorf("GemvQ4Sym GPTQ validation: %w", err)
	}
	if len(out) < outDim {
		return fmt.Errorf("GemvQ4Sym out length=%d, expected at least %d", len(out), outDim)
	}
	if len(x) < inDim {
		return fmt.Errorf("GemvQ4Sym x length=%d, expected at least %d", len(x), inDim)
	}
	return nil
}

func ValidateSym(qweight, gIdx []int32, scales []float32, inFeatures, outFeatures int) error {
	return Validate(qweight, nil, gIdx, scales, inFeatures, outFeatures, true)
}

func Validate(qweight, qzeros, gIdx []int32, scales []float32, inFeatures, outFeatures int, sym bool) error {
	if inFeatures <= 0 || outFeatures <= 0 {
		return fmt.Errorf("invalid GPTQ dims in=%d out=%d", inFeatures, outFeatures)
	}
	if inFeatures%8 != 0 {
		return fmt.Errorf("GPTQ inFeatures=%d is not divisible by 8", inFeatures)
	}
	if !sym && outFeatures%8 != 0 {
		return fmt.Errorf("GPTQ outFeatures=%d is not divisible by 8 for qzeros", outFeatures)
	}
	if _, ok := checkedMulInt(inFeatures, outFeatures); !ok {
		return fmt.Errorf("GPTQ output size overflows for in=%d out=%d", inFeatures, outFeatures)
	}
	wantQWeight, ok := checkedMulInt(inFeatures/8, outFeatures)
	if !ok {
		return fmt.Errorf("GPTQ qweight size overflows for in=%d out=%d", inFeatures, outFeatures)
	}
	if len(qweight) < wantQWeight {
		return fmt.Errorf("GPTQ qweight length=%d, expected at least %d", len(qweight), wantQWeight)
	}
	if len(gIdx) < inFeatures {
		return fmt.Errorf("GPTQ g_idx length=%d, expected at least %d", len(gIdx), inFeatures)
	}

	maxGroup := -1
	for i := 0; i < inFeatures; i++ {
		g := int(gIdx[i])
		if g < 0 {
			return fmt.Errorf("GPTQ g_idx[%d]=%d is negative", i, g)
		}
		if g > maxGroup {
			maxGroup = g
		}
	}
	if maxGroup < 0 {
		return fmt.Errorf("GPTQ g_idx has no groups")
	}

	wantScales, ok := checkedMulInt(maxGroup+1, outFeatures)
	if !ok {
		return fmt.Errorf("GPTQ scales size overflows for %d groups and out=%d", maxGroup+1, outFeatures)
	}
	if len(scales) < wantScales {
		return fmt.Errorf("GPTQ scales length=%d, expected at least %d for %d groups", len(scales), wantScales, maxGroup+1)
	}
	if !sym {
		wantQZeros, ok := checkedMulInt(maxGroup+1, outFeatures/8)
		if !ok {
			return fmt.Errorf("GPTQ qzeros size overflows for %d groups and out=%d", maxGroup+1, outFeatures)
		}
		if len(qzeros) < wantQZeros {
			return fmt.Errorf("GPTQ qzeros length=%d, expected at least %d for %d groups", len(qzeros), wantQZeros, maxGroup+1)
		}
	}
	return nil
}

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

	for i := 0; i < inFeatures; i++ {
		g := int(gIdx[i]) // group for this input row

		// Extract zero point for each output from qzeros
		// qzeros is [numGroups, outFeatures/8] with 8 x 4-bit per int32
		for j := 0; j < outFeatures; j++ {
			// Extract 4-bit quantized weight
			packIdx := i / 8        // which int32 in qweight
			bitIdx := uint(i%8) * 4 // bit offset within int32
			qw := (qweight[packIdx*outFeatures+j] >> bitIdx) & 0xF

			// Extract 4-bit zero point
			var qz int32
			if sym {
				qz = 8 // symmetric: zero point is always 2^(bits-1)
			} else {
				zPackIdx := j / 8
				zBitIdx := uint(j%8) * 4
				qz = (qzeros[g*(outFeatures/8)+zPackIdx] >> zBitIdx) & 0xF
			}

			// Dequantize: w = scale * (qw - qz)
			scale := scales[g*outFeatures+j]
			out[j*inFeatures+i] = scale * float32(qw-qz)
		}
	}

	return out
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
	nPacks := inFeatures / 8

	// Parallelize across output rows
	nWorkers := runtime.NumCPU()
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
		wg.Add(1)
		go func(jStart, jEnd int) {
			defer wg.Done()
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
		}(jStart, jEnd)
	}
	wg.Wait()

	return out
}

// Float16ToFloat32 converts a uint16 IEEE 754 half-precision to float32.
func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h) & 0x3FF

	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		frac &= 0x3FF
		exp++
		exp += 127 - 15
	} else if exp == 0x1F {
		exp = 0xFF
	} else {
		exp += 127 - 15
	}

	return math.Float32frombits((sign << 31) | (exp << 23) | (frac << 13))
}

func checkedMulInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if b != 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
}
