package qwen

import (
	"fmt"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/tensor"
)

func BenchmarkSIMDBranchProjectionPadding(b *testing.B) {
	for _, rows := range []int{6, 12, 18, 41, 42, 48, 60, 126, 128, 256, 512} {
		b.Run(fmt.Sprintf("rows%d", rows), func(b *testing.B) {
			const in, out = 1024, 3584
			w := tensor.FromOwnedFloat32(make([]float32, in*out), []int{out, in})
			for i := range w.Data() {
				w.Data()[i] = float32(i%19-9) / 64
			}
			packed, err := simd.PackSgemmNTWeights(w.Data(), out, in, in)
			if err != nil {
				b.Fatal(err)
			}
			// Old-12 capacity admits both padding policies without moving an
			// allocation into the timed region. Model construction is separate.
			padded := (rows + 11) / 12 * 12
			s := &Qwen35SIMDBranch{packedOnly: true, packed: map[*tensor.Tensor][]float32{w: packed}, scratch: map[string][]float32{"padIn": make([]float32, padded*in), "padOut": make([]float32, padded*out)}}
			stop := s.startProjectionWorkers()
			defer stop()
			x, dst := make([]float32, rows*in), make([]float32, rows*out)
			for i := range x {
				x[i] = float32(i%23-11) / 32
			}
			if err := s.project(dst, x, w, rows, in, out); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err := s.project(dst, x, w, rows, in, out); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
