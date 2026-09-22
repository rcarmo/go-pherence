package pockettts

import (
	"fmt"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// FlowNet predicts a latent velocity for one LSD interval. Buffers are owned by
// the caller so production implementations can reuse request-local scratch.
type FlowNet func(dst []float32, start, target float32, current []float32) error

// LSDDecodeSIMD mirrors upstream lsd_decode. It uses the repository's SIMD
// scale/add primitive for every latent update.
func LSDDecodeSIMD(dst, initial []float32, steps int, velocity []float32, network FlowNet) error {
	if len(dst) == 0 || len(initial) != len(dst) || len(velocity) != len(dst) || steps <= 0 || network == nil {
		return fmt.Errorf("invalid Pocket TTS LSD buffers/steps")
	}
	copy(dst, initial)
	for i := 0; i < steps; i++ {
		start := float32(i) / float32(steps)
		target := float32(i+1) / float32(steps)
		if err := network(velocity, start, target, dst); err != nil {
			return fmt.Errorf("Pocket TTS LSD step %d: %w", i, err)
		}
		if !finiteF32(velocity) {
			return fmt.Errorf("Pocket TTS LSD step %d produced non-finite velocity", i)
		}
		if !simd.VecScaleAddTo(dst, dst, velocity, 1/float32(steps)) {
			return fmt.Errorf("Pocket TTS LSD SIMD update failed")
		}
	}
	if !finiteF32(dst) {
		return fmt.Errorf("Pocket TTS LSD produced non-finite latent")
	}
	return nil
}

func lsdDecodeScalar(dst, initial []float32, steps int, velocity []float32, network FlowNet) error {
	if len(dst) == 0 || len(initial) != len(dst) || len(velocity) != len(dst) || steps <= 0 || network == nil {
		return fmt.Errorf("invalid Pocket TTS LSD buffers/steps")
	}
	copy(dst, initial)
	for i := 0; i < steps; i++ {
		start, target := float32(i)/float32(steps), float32(i+1)/float32(steps)
		if err := network(velocity, start, target, dst); err != nil {
			return err
		}
		for j := range dst {
			dst[j] += velocity[j] / float32(steps)
		}
	}
	return nil
}

// AffineSIMD applies y = Wx+b for a row-major [out,in] F32 matrix.
func AffineSIMD(dst, input, weight, bias []float32, in, out int) error {
	if len(dst) != out || len(input) != in || len(weight) != in*out || (bias != nil && len(bias) != out) || in <= 0 || out <= 0 {
		return fmt.Errorf("invalid Pocket TTS affine shape in=%d out=%d", in, out)
	}
	if !simd.GemvRows(dst, input, weight, out, in) {
		return fmt.Errorf("Pocket TTS SIMD affine failed")
	}
	if bias != nil {
		if !simd.VecAddTo(dst, dst, bias) {
			return fmt.Errorf("Pocket TTS SIMD bias failed")
		}
	}
	return nil
}

func affineScalar(dst, input, weight, bias []float32, in, out int) {
	for row := 0; row < out; row++ {
		sum := float32(0)
		if bias != nil {
			sum = bias[row]
		}
		for col := 0; col < in; col++ {
			sum += input[col] * weight[row*in+col]
		}
		dst[row] = sum
	}
}
