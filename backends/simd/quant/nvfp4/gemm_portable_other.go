//go:build !riscv64

package nvfp4

import (
	"runtime"
	"sync"
)

func gemmNVFP4Portable(out, x []float32, batch int, qw *NVFP4Weight) bool {
	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || qw.OutDim < 128 || qw.InDim < 64 {
		gemmNVFP4RowsBatchedPortable(out, x, batch, qw, 0, qw.OutDim)
		return true
	}
	if workers > qw.OutDim {
		workers = qw.OutDim
	}
	rowsPer := (qw.OutDim + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < qw.OutDim; start += rowsPer {
		end := start + rowsPer
		if end > qw.OutDim {
			end = qw.OutDim
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			gemmNVFP4RowsBatchedPortable(out, x, batch, qw, start, end)
		}(start, end)
	}
	wg.Wait()
	return true
}

func gemmNVFP4RowsBatchedPortable(out, x []float32, batch int, qw *NVFP4Weight, start, end int) {
	outDim := qw.OutDim
	inDim := qw.InDim
	for b := 0; b < batch; b += 4 {
		remaining := batch - b
		if remaining > 4 {
			remaining = 4
		}
		x0 := x[b*inDim : (b+1)*inDim]
		out0 := out[b*outDim : (b+1)*outDim]
		switch remaining {
		case 1:
			gemmNVFP4Rows1(out0, x0, qw, start, end)
		case 2:
			x1 := x[(b+1)*inDim : (b+2)*inDim]
			out1 := out[(b+1)*outDim : (b+2)*outDim]
			gemmNVFP4Rows2(out0, out1, x0, x1, qw, start, end)
		case 3:
			x1 := x[(b+1)*inDim : (b+2)*inDim]
			x2 := x[(b+2)*inDim : (b+3)*inDim]
			out1 := out[(b+1)*outDim : (b+2)*outDim]
			out2 := out[(b+2)*outDim : (b+3)*outDim]
			gemmNVFP4Rows3(out0, out1, out2, x0, x1, x2, qw, start, end)
		default:
			x1 := x[(b+1)*inDim : (b+2)*inDim]
			x2 := x[(b+2)*inDim : (b+3)*inDim]
			x3 := x[(b+3)*inDim : (b+4)*inDim]
			out1 := out[(b+1)*outDim : (b+2)*outDim]
			out2 := out[(b+2)*outDim : (b+3)*outDim]
			out3 := out[(b+3)*outDim : (b+4)*outDim]
			gemmNVFP4Rows4(out0, out1, out2, out3, x0, x1, x2, x3, qw, start, end)
		}
	}
}

func gemmNVFP4Rows1(out0, x0 []float32, qw *NVFP4Weight, start, end int) {
	packedPerRow := qw.InDim / 2
	groups := qw.Groups
	groupSize := qw.GroupSize
	scale2 := qw.WeightScale2
	for row := start; row < end; row++ {
		packed := qw.Weight[row*packedPerRow : (row+1)*packedPerRow]
		scales := qw.WeightScale[row*groups : (row+1)*groups]
		sum0 := float32(0)
		for group := 0; group < groups; group++ {
			scale := DecodeF8E4M3(scales[group]) * scale2
			groupStart := group * groupSize
			groupEnd := groupStart + groupSize
			for col := groupStart; col < groupEnd; col++ {
				code := packed[col/2] & 0x0f
				if col&1 != 0 {
					code = packed[col/2] >> 4
				}
				sum0 += DecodeFP4E2M1(code) * scale * x0[col]
			}
		}
		out0[row] = sum0
	}
}

func gemmNVFP4Rows2(out0, out1, x0, x1 []float32, qw *NVFP4Weight, start, end int) {
	packedPerRow := qw.InDim / 2
	groups := qw.Groups
	groupSize := qw.GroupSize
	scale2 := qw.WeightScale2
	for row := start; row < end; row++ {
		packed := qw.Weight[row*packedPerRow : (row+1)*packedPerRow]
		scales := qw.WeightScale[row*groups : (row+1)*groups]
		sum0, sum1 := float32(0), float32(0)
		for group := 0; group < groups; group++ {
			scale := DecodeF8E4M3(scales[group]) * scale2
			groupStart := group * groupSize
			groupEnd := groupStart + groupSize
			for col := groupStart; col < groupEnd; col++ {
				code := packed[col/2] & 0x0f
				if col&1 != 0 {
					code = packed[col/2] >> 4
				}
				wv := DecodeFP4E2M1(code) * scale
				sum0 += wv * x0[col]
				sum1 += wv * x1[col]
			}
		}
		out0[row] = sum0
		out1[row] = sum1
	}
}

func gemmNVFP4Rows3(out0, out1, out2, x0, x1, x2 []float32, qw *NVFP4Weight, start, end int) {
	packedPerRow := qw.InDim / 2
	groups := qw.Groups
	groupSize := qw.GroupSize
	scale2 := qw.WeightScale2
	for row := start; row < end; row++ {
		packed := qw.Weight[row*packedPerRow : (row+1)*packedPerRow]
		scales := qw.WeightScale[row*groups : (row+1)*groups]
		sum0, sum1, sum2 := float32(0), float32(0), float32(0)
		for group := 0; group < groups; group++ {
			scale := DecodeF8E4M3(scales[group]) * scale2
			groupStart := group * groupSize
			groupEnd := groupStart + groupSize
			for col := groupStart; col < groupEnd; col++ {
				code := packed[col/2] & 0x0f
				if col&1 != 0 {
					code = packed[col/2] >> 4
				}
				wv := DecodeFP4E2M1(code) * scale
				sum0 += wv * x0[col]
				sum1 += wv * x1[col]
				sum2 += wv * x2[col]
			}
		}
		out0[row] = sum0
		out1[row] = sum1
		out2[row] = sum2
	}
}

func gemmNVFP4Rows4(out0, out1, out2, out3, x0, x1, x2, x3 []float32, qw *NVFP4Weight, start, end int) {
	packedPerRow := qw.InDim / 2
	groups := qw.Groups
	groupSize := qw.GroupSize
	scale2 := qw.WeightScale2
	for row := start; row < end; row++ {
		packed := qw.Weight[row*packedPerRow : (row+1)*packedPerRow]
		scales := qw.WeightScale[row*groups : (row+1)*groups]
		sum0, sum1, sum2, sum3 := float32(0), float32(0), float32(0), float32(0)
		for group := 0; group < groups; group++ {
			scale := DecodeF8E4M3(scales[group]) * scale2
			groupStart := group * groupSize
			groupEnd := groupStart + groupSize
			for col := groupStart; col < groupEnd; col++ {
				code := packed[col/2] & 0x0f
				if col&1 != 0 {
					code = packed[col/2] >> 4
				}
				wv := DecodeFP4E2M1(code) * scale
				sum0 += wv * x0[col]
				sum1 += wv * x1[col]
				sum2 += wv * x2[col]
				sum3 += wv * x3[col]
			}
		}
		out0[row] = sum0
		out1[row] = sum1
		out2[row] = sum2
		out3[row] = sum3
	}
}
