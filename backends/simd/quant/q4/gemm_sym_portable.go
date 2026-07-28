package q4

// gemmSymBatchedPortable computes symmetric GPTQ INT4 GEMM by sharing a
// single packed-weight/group traversal across up to four activation rows at a
// time. It preserves exact scalar accumulation order per output row while
// reducing repeated qweight/scales walks versus per-row GEMV.
func gemmSymBatchedPortable(out, x []float32, batch int, qweight, gIdx []int32, scales []float32, inDim, outDim int) bool {
	out = out[:batch*outDim]
	clear(out)
	for b := 0; b < batch; b += 4 {
		remaining := batch - b
		if remaining > 4 {
			remaining = 4
		}
		x0 := x[b*inDim : (b+1)*inDim]
		out0 := out[b*outDim : (b+1)*outDim]
		switch remaining {
		case 1:
			gemmSymAccumulate1(out0, x0, qweight, gIdx, scales, inDim, outDim)
		case 2:
			x1 := x[(b+1)*inDim : (b+2)*inDim]
			out1 := out[(b+1)*outDim : (b+2)*outDim]
			gemmSymAccumulate2(out0, out1, x0, x1, qweight, gIdx, scales, inDim, outDim)
		case 3:
			x1 := x[(b+1)*inDim : (b+2)*inDim]
			x2 := x[(b+2)*inDim : (b+3)*inDim]
			out1 := out[(b+1)*outDim : (b+2)*outDim]
			out2 := out[(b+2)*outDim : (b+3)*outDim]
			gemmSymAccumulate3(out0, out1, out2, x0, x1, x2, qweight, gIdx, scales, inDim, outDim)
		default:
			x1 := x[(b+1)*inDim : (b+2)*inDim]
			x2 := x[(b+2)*inDim : (b+3)*inDim]
			x3 := x[(b+3)*inDim : (b+4)*inDim]
			out1 := out[(b+1)*outDim : (b+2)*outDim]
			out2 := out[(b+2)*outDim : (b+3)*outDim]
			out3 := out[(b+3)*outDim : (b+4)*outDim]
			gemmSymAccumulate4(out0, out1, out2, out3, x0, x1, x2, x3, qweight, gIdx, scales, inDim, outDim)
		}
	}
	return true
}

func gemmSymAccumulate1(out0, x0 []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	nPacks := inDim / 8
	for packIdx := 0; packIdx < nPacks; packIdx++ {
		qrow := qweight[packIdx*outDim : (packIdx+1)*outDim]
		base := packIdx * 8
		for bit := 0; bit < 8; bit++ {
			i := base + bit
			scaleBase := int(gIdx[i]) * outDim
			scaleRow := scales[scaleBase : scaleBase+outDim]
			shift := uint(bit) * 4
			x0i := x0[i]
			for j := 0; j < outDim; j++ {
				out0[j] += x0i * scaleRow[j] * float32(((qrow[j]>>shift)&0xF)-8)
			}
		}
	}
}

func gemmSymAccumulate2(out0, out1, x0, x1 []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	nPacks := inDim / 8
	for packIdx := 0; packIdx < nPacks; packIdx++ {
		qrow := qweight[packIdx*outDim : (packIdx+1)*outDim]
		base := packIdx * 8
		for bit := 0; bit < 8; bit++ {
			i := base + bit
			scaleBase := int(gIdx[i]) * outDim
			scaleRow := scales[scaleBase : scaleBase+outDim]
			shift := uint(bit) * 4
			x0i, x1i := x0[i], x1[i]
			for j := 0; j < outDim; j++ {
				wv := scaleRow[j] * float32(((qrow[j]>>shift)&0xF)-8)
				out0[j] += x0i * wv
				out1[j] += x1i * wv
			}
		}
	}
}

func gemmSymAccumulate3(out0, out1, out2, x0, x1, x2 []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	nPacks := inDim / 8
	for packIdx := 0; packIdx < nPacks; packIdx++ {
		qrow := qweight[packIdx*outDim : (packIdx+1)*outDim]
		base := packIdx * 8
		for bit := 0; bit < 8; bit++ {
			i := base + bit
			scaleBase := int(gIdx[i]) * outDim
			scaleRow := scales[scaleBase : scaleBase+outDim]
			shift := uint(bit) * 4
			x0i, x1i, x2i := x0[i], x1[i], x2[i]
			for j := 0; j < outDim; j++ {
				wv := scaleRow[j] * float32(((qrow[j]>>shift)&0xF)-8)
				out0[j] += x0i * wv
				out1[j] += x1i * wv
				out2[j] += x2i * wv
			}
		}
	}
}

func gemmSymAccumulate4(out0, out1, out2, out3, x0, x1, x2, x3 []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	nPacks := inDim / 8
	for packIdx := 0; packIdx < nPacks; packIdx++ {
		qrow := qweight[packIdx*outDim : (packIdx+1)*outDim]
		base := packIdx * 8
		for bit := 0; bit < 8; bit++ {
			i := base + bit
			scaleBase := int(gIdx[i]) * outDim
			scaleRow := scales[scaleBase : scaleBase+outDim]
			shift := uint(bit) * 4
			x0i, x1i, x2i, x3i := x0[i], x1[i], x2[i], x3[i]
			for j := 0; j < outDim; j++ {
				wv := scaleRow[j] * float32(((qrow[j]>>shift)&0xF)-8)
				out0[j] += x0i * wv
				out1[j] += x1i * wv
				out2[j] += x2i * wv
				out3[j] += x3i * wv
			}
		}
	}
}
