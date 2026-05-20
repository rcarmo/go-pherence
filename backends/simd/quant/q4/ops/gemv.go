package ops

// Gemv computes out = x @ W^T where W is stored as GPTQ INT4. It supports
// both symmetric and asymmetric qzeros layouts and is scalar/reference-owned;
// optimized kernels can dispatch through the symmetric GemvSym path when they
// land.
func Gemv(out, x []float32, qweight, qzeros, gIdx []int32, scales []float32, inDim, outDim int, sym bool) {
	_ = GemvTo(out, x, qweight, qzeros, gIdx, scales, inDim, outDim, sym)
}

// GemvTo computes GPTQ INT4 GEMV into caller-owned output and reports malformed
// inputs. It preserves Gemv's output layout and no-allocation behavior.
func GemvTo(out, x []float32, qweight, qzeros, gIdx []int32, scales []float32, inDim, outDim int, sym bool) bool {
	if err := ValidateGemv(out, x, qweight, qzeros, gIdx, scales, inDim, outDim, sym); err != nil {
		return false
	}
	if sym {
		gemvSymScalar(out, x, qweight, gIdx, scales, inDim, outDim)
		return true
	}
	gemvScalar(out, x, qweight, qzeros, gIdx, scales, inDim, outDim)
	return true
}

func GemvSym(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	_ = GemvSymTo(out, x, qweight, gIdx, scales, inDim, outDim)
}

// GemvSymTo computes symmetric GPTQ INT4 GEMV into caller-owned output and
// reports malformed inputs.
func GemvSymTo(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) bool {
	if err := ValidateGemvSym(out, x, qweight, gIdx, scales, inDim, outDim); err != nil {
		return false
	}
	// Dispatch hook kept explicit so AVX2/NEON kernels can be wired without
	// changing callers. Scalar remains active until hasGemvSymAsm flips true.
	gemvSymScalar(out, x, qweight, gIdx, scales, inDim, outDim)
	return true
}

func gemvScalar(out, x []float32, qweight, qzeros, gIdx []int32, scales []float32, inDim, outDim int) {
	for j := 0; j < outDim; j++ {
		out[j] = 0
	}
	for packIdx := 0; packIdx < inDim/8; packIdx++ {
		qrow := qweight[packIdx*outDim : (packIdx+1)*outDim]
		for bit := 0; bit < 8; bit++ {
			i := packIdx*8 + bit
			g := int(gIdx[i])
			scaleRow := scales[g*outDim : (g+1)*outDim]
			qzeroRow := qzeros[g*(outDim/8) : (g+1)*(outDim/8)]
			xi := x[i]
			shift := uint(bit) * 4
			for j := 0; j < outDim; j++ {
				qw := (qrow[j] >> shift) & 0xF
				zPackIdx := j / 8
				zShift := uint(j%8) * 4
				qz := (qzeroRow[zPackIdx] >> zShift) & 0xF
				out[j] += xi * scaleRow[j] * float32(qw-qz)
			}
		}
	}
}

func gemvSymScalar(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	for j := 0; j < outDim; j++ {
		out[j] = 0
	}
	for packIdx := 0; packIdx < inDim/8; packIdx++ {
		qrow := qweight[packIdx*outDim : (packIdx+1)*outDim]
		for bit := 0; bit < 8; bit++ {
			i := packIdx*8 + bit
			g := int(gIdx[i])
			scaleRow := scales[g*outDim : (g+1)*outDim]
			xi := x[i]
			shift := uint(bit) * 4
			for j := 0; j < outDim; j++ {
				qw := (qrow[j] >> shift) & 0xF
				out[j] += xi * scaleRow[j] * float32(qw-8)
			}
		}
	}
}
