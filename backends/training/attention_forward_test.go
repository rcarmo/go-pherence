package training

import (
	"math"
	"testing"
)

func TestTrainingAttentionForwardAdmission(t *testing.T) {
	for mode := uint32(0); mode < 3; mode++ {
		if err := AttentionForwardShape(13, 3, 14, 2, mode, .25); err != nil {
			t.Fatal(err)
		}
	}
	for _, v := range []struct {
		rows, heads, dim, context int
		mode                      uint32
		scale                     float32
	}{{0, 1, 2, 0, 0, 1}, {1025, 1, 2, 0, 0, 1}, {1, 33, 2, 0, 0, 1}, {1, 1, 257, 0, 0, 1}, {1, 1, 2, -1, 0, 1}, {1, 1, 2, 2, 0, 1}, {1, 1, 2, 0, 3, 1}, {1, 1, 2, 0, 0, 0}, {1, 1, 2, 0, 0, float32(math.NaN())}} {
		if err := AttentionForwardShape(v.rows, v.heads, v.dim, v.context, v.mode, v.scale); err == nil {
			t.Fatal("invalid attention forward")
		}
	}
	tape := attentionForwardTape()
	if err := tape.Validate(1 << 20); err != nil {
		t.Fatal(err)
	}
	// Backward tape admission uses the same exact dense and tensor shapes.
	backward := func() Tape {
		t := attentionForwardTape()
		t.Tensors = append(t.Tensors, TensorSpec{Shape: []int{3, 2, 2}, Values: make([]float32, 12)}, TensorSpec{Shape: []int{2, 3, 3}, Values: make([]float32, 18)}, TensorSpec{Shape: []int{3, 2, 2}})
		t.Commands = append(t.Commands, Command{Op: "attention-backward", Inputs: []int{0, 1, 2, 5, 7, 8}, Output: 9, Mode: 2, Scale: .5})
		return t
	}
	if err := backward().Validate(1 << 20); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Tape){func(t *Tape) { t.Commands[3].Inputs = nil }, func(t *Tape) { t.Commands[3].Mode = 5 }, func(t *Tape) { t.Tensors[7].Shape = []int{3, 1, 4} }, func(t *Tape) { t.Tensors[8].Shape = []int{1, 3, 6} }, func(t *Tape) { t.Tensors[9].Shape = []int{2, 3, 3} }} {
		bad := backward()
		mutate(&bad)
		if err := bad.Validate(1 << 20); err == nil {
			t.Fatal("invalid backward tape")
		}
	}
	for _, mutate := range []func(*Tape){func(t *Tape) { t.Commands[0].Inputs = nil }, func(t *Tape) { t.Tensors[0].Shape = []int{3, 4} }, func(t *Tape) { t.Tensors[1].Shape = []int{3, 1, 2} }, func(t *Tape) { t.Tensors[3].Shape = []int{1, 3, 3} }, func(t *Tape) { t.Commands[1].Mode = 3 }, func(t *Tape) { t.Tensors[6].Shape = []int{2, 3, 3} }, func(t *Tape) { t.Commands[0].Offset = 4 }} {
		bad := attentionForwardTape()
		mutate(&bad)
		if err := bad.Validate(1 << 20); err == nil {
			t.Fatal("invalid attention tape")
		}
	}
}
func attentionForwardTape() Tape {
	tensors := []TensorSpec{}
	for i := 0; i < 3; i++ {
		tensors = append(tensors, TensorSpec{Shape: []int{3, 2, 2}, Values: make([]float32, 12)})
	}
	tensors = append(tensors, TensorSpec{Shape: []int{2, 3, 3}, Values: make([]float32, 18)}, TensorSpec{Shape: []int{2, 3, 3}}, TensorSpec{Shape: []int{2, 3, 3}}, TensorSpec{Shape: []int{3, 2, 2}})
	return Tape{Tensors: tensors, Commands: []Command{{Op: "attention-forward", Inputs: []int{0, 1, 2, 3}, Output: 4, Scale: .5}, {Op: "attention-forward", Inputs: []int{0, 1, 2, 4}, Output: 5, Mode: 1, Scale: .5}, {Op: "attention-forward", Inputs: []int{0, 1, 2, 5}, Output: 6, Mode: 2, Scale: .5}}, Outputs: []int{5, 6}}
}
func TestTrainingAttentionForwardHostNumerics(t *testing.T) {
	maximum, square, count := float64(0), float64(0), 0
	for _, s := range []struct {
		rows, heads, dim int
		amplitude        float64
	}{{1, 1, 2, 1}, {3, 2, 5, 1}, {13, 3, 14, 1}, {3, 2, 5, 12}} {
		n := s.rows * s.heads * s.dim
		q, k, v := make([]float32, n), make([]float32, n), make([]float32, n)
		for i := range q {
			q[i] = float32(math.Sin(float64(i)*.37) * s.amplitude)
			k[i] = float32(math.Cos(float64(i)*.29) * s.amplitude)
			v[i] = float32(math.Sin(float64(i)*.11) * 2)
		}
		for _, ctx := range []int{0, 1, 2, s.rows} {
			if ctx > s.rows {
				continue
			}
			scale := float32(1 / math.Sqrt(float64(s.dim)))
			scores, prob, out := make([]float32, s.heads*s.rows*s.rows), make([]float32, s.heads*s.rows*s.rows), make([]float32, n)
			refProb, refOut := make([]float64, len(prob)), make([]float64, n)
			for h := 0; h < s.heads; h++ {
				for query := 0; query < s.rows; query++ {
					start := 0
					if ctx > 0 {
						start = max(0, query-ctx+1)
					}
					base := (h*s.rows + query) * s.rows
					mx := float32(-math.MaxFloat32)
					rmx := math.Inf(-1)
					for key := start; key <= query; key++ {
						var sum float32
						var rs float64
						for d := 0; d < s.dim; d++ {
							a, b := q[(query*s.heads+h)*s.dim+d], k[(key*s.heads+h)*s.dim+d]
							sum += a * b
							rs += float64(a) * float64(b)
						}
						scores[base+key] = sum * scale
						refProb[base+key] = rs * float64(scale)
						mx = max(mx, scores[base+key])
						rmx = math.Max(rmx, refProb[base+key])
					}
					var sum float32
					var rsum float64
					for key := start; key <= query; key++ {
						prob[base+key] = float32(math.Exp(float64(scores[base+key] - mx)))
						sum += prob[base+key]
						refProb[base+key] = math.Exp(refProb[base+key] - rmx)
						rsum += refProb[base+key]
					}
					for key := start; key <= query; key++ {
						prob[base+key] /= sum
						refProb[base+key] /= rsum
						for d := 0; d < s.dim; d++ {
							idx := (query*s.heads+h)*s.dim + d
							out[idx] += prob[base+key] * v[(key*s.heads+h)*s.dim+d]
							refOut[idx] += refProb[base+key] * float64(v[(key*s.heads+h)*s.dim+d])
						}
					}
					var ps float64
					for key := 0; key < s.rows; key++ {
						allowed := key >= start && key <= query
						if !allowed && prob[base+key] != 0 {
							t.Fatal("masked key nonzero")
						}
						ps += float64(prob[base+key])
					}
					if math.Abs(ps-1) > 2e-7 {
						t.Fatal("row probability mass", ps)
					}
				}
			}
			for j, got := range [][]float32{prob, out} {
				ref := refProb
				if j == 1 {
					ref = refOut
				}
				for i, x := range got {
					err := math.Abs(float64(x) - ref[i])
					if err > 2e-6 {
						t.Fatalf("host attention drift rows%d amp%g context%d array%d i%d=%g", s.rows, s.amplitude, ctx, j, i, err)
					}
					maximum = math.Max(maximum, err)
					square += err * err
					count++
				}
			}
		}
	}
	t.Logf("host attention forward values=%d max_abs=%g rms=%g", count, maximum, math.Sqrt(square/float64(count)))
}
