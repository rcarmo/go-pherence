package community1

import (
	"context"
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func validWeSpeakerBlockMode(mode WeSpeakerBlockMode) bool {
	return mode == WeSpeakerBlockScalar || mode == WeSpeakerBlockSIMD || mode == WeSpeakerBlockGEMM
}

// Bounded spatial im2col: pack at most64 positions, reduce all input channels
// in one GEMM and scatter channel-major output. Row-major weights are unchanged.
// Packing crosses frequency rows explicitly and zero-fills edge coordinates.
// Packed scratch is <=576KiB at256channels/kernel3, plus<=64KiB projection
// and<=4.5KiB offsets on64-bit. No full-window im2col or workers.
func weSpeakerBlockConvTiled(ctx context.Context, x []float32, in, out CHWShape, w []float32, kernel, stride, padding int) ([]float32, error) {
	const tile = 64
	k := in.Channels * kernel * kernel
	positions := out.Frequency * out.Frames
	packed := make([]float32, k*tile)
	projected := make([]float32, out.Channels*tile)
	result := make([]float32, out.Channels*positions)
	for start := 0; start < positions; start += tile {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n := min(tile, positions-start)
		b, c := packed[:k*n], projected[:out.Channels*n]
		clear(b)
		// Spatial offsets are shared by every input channel. Pack contiguous
		// reduction rows, avoiding strided writes and repeated division.
		var offsets [9][tile]int
		for kf := 0; kf < kernel; kf++ {
			for kt := 0; kt < kernel; kt++ {
				row := &offsets[kf*kernel+kt]
				for j := 0; j < n; j++ {
					f, t := (start+j)/out.Frames, (start+j)%out.Frames
					sf, st := f*stride+kf-padding, t*stride+kt-padding
					row[j] = -1
					if sf >= 0 && sf < in.Frequency && st >= 0 && st < in.Frames {
						row[j] = sf*in.Frames + st
					}
				}
			}
		}
		for channel := 0; channel < in.Channels; channel++ {
			if channel%32 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			source := x[channel*in.Frequency*in.Frames : (channel+1)*in.Frequency*in.Frames]
			for tap := 0; tap < kernel*kernel; tap++ {
				row := b[(channel*kernel*kernel+tap)*n : (channel*kernel*kernel+tap+1)*n]
				for j, offset := range offsets[tap][:n] {
					if offset >= 0 {
						row[j] = source[offset]
					}
				}
			}
		}
		if !simd.FMAMatrixF32Checked(c, w, b, out.Channels, n, k) {
			return nil, fmt.Errorf("checked WeSpeaker FMA GEMM rejected tile/FP state")
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for channel := 0; channel < out.Channels; channel++ {
			copy(result[channel*positions+start:channel*positions+start+n], c[channel*n:(channel+1)*n])
		}
	}
	if err := finiteBlock(ctx, result); err != nil {
		return nil, err
	}
	return result, nil
}
