package mlx

import (
	"math/rand"
	"testing"
)

func makeQuantWeight(bits, inDim, outDim, groupSize int, seed int64) *QuantWeight {
	r := rand.New(rand.NewSource(seed))
	packFactor := 32 / bits
	groups := inDim / groupSize
	qw := &QuantWeight{
		Bits:      bits,
		InDim:     inDim,
		OutDim:    outDim,
		GroupSize: groupSize,
		Groups:    groups,
		Weight:    make([]uint32, outDim*(inDim/packFactor)),
		Scales:    make([]float32, outDim*groups),
		Biases:    make([]float32, outDim*groups),
	}
	for i := range qw.Weight {
		qw.Weight[i] = r.Uint32()
	}
	for i := range qw.Scales {
		qw.Scales[i] = r.Float32()*0.1 + 0.01
		qw.Biases[i] = r.Float32() * 0.05
	}
	return qw
}

// TestGemvParallelMatchesSerial verifies the row-parallel MLX GEMV is
// bit-identical to the serial GemvTo for both the 4-bit and generic paths.
func TestGemvParallelMatchesSerial(t *testing.T) {
	cases := []struct {
		name              string
		bits, group, outD int
	}{
		{"4bit", 4, 64, 1024},
		{"8bit", 8, 64, 1024},
	}
	inDim := 256
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			qw := makeQuantWeight(tc.bits, inDim, tc.outD, tc.group, 7)
			x := make([]float32, inDim)
			r := rand.New(rand.NewSource(11))
			for i := range x {
				x[i] = r.Float32()*2 - 1
			}
			want := make([]float32, tc.outD)
			got := make([]float32, tc.outD)
			if !GemvTo(want, x, qw) {
				t.Fatal("GemvTo failed")
			}
			if !GemvParallel(got, x, qw) {
				t.Fatal("GemvParallel failed")
			}
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("row %d: serial=%v parallel=%v", i, want[i], got[i])
				}
			}
		})
	}
}

func TestGemvParallelMalformed(t *testing.T) {
	if GemvParallel(nil, nil, nil) {
		t.Fatal("GemvParallel accepted nil weight")
	}
}
