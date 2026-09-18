package community1

import (
	"context"
	"fmt"
	"testing"
)

// Fixed work, no training/checkpoint/PCM workload. Source copy is included in
// both arms; normalisation includes unchanged statistics + affine + finite scan.
func BenchmarkSincNetNormalization(b *testing.B) {
	for _, shape := range [][2]int{{80, 43}, {1, 4096}, {1, 160000}} {
		channels, frames := shape[0], shape[1]
		source := make([]float32, channels*frames)
		scratch := make([]float32, len(source))
		norm := SincNetNorm{Weight: make([]float32, channels), Bias: make([]float32, channels)}
		for i := range source {
			source[i] = float32(i%137-68) / 137
		}
		for i := range norm.Weight {
			norm.Weight[i] = .83
			norm.Bias[i] = .017
		}
		for _, mode := range []SincNetMode{SincNetScalar, SincNetSIMD} {
			b.Run(fmt.Sprintf("%dx%d/mode%d", channels, frames, mode), func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(len(source) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(scratch, source)
					if err := sincNetNormMode(context.Background(), scratch, channels, frames, norm, mode); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
