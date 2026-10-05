package pockettts

import (
	"math"
	"testing"
)

func TestTrainingAttentionTapeForwardBackward(t *testing.T) {
	for _, contextRows := range []int{0, 1, 2, 4} {
		rows, heads, dim, origin := 4, 2, 6, 7
		n := rows * heads * dim
		q, k, v, dy := make([]float32, n), make([]float32, n), make([]float32, n), make([]float32, n)
		for i := range q {
			q[i] = float32(math.Sin(float64(i) * .3))
			k[i] = float32(math.Cos(float64(i) * .4))
			v[i] = float32(math.Sin(float64(i) * .7))
			dy[i] = float32(math.Cos(float64(i)*.9)) * .3
		}
		tape, err := BuildAttentionTrainingTape(q, k, v, dy, rows, heads, dim, contextRows, origin, 10000, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		got, err := tape.Decode(interpretFlowTape(t, tape.Tape))
		if err != nil {
			t.Fatal(err)
		}
		evaluate := func(which, index int, delta float64) (float64, []float64) {
			input := [][]float32{q, k, v}
			rotated := make([][]float64, 3)
			for a := range rotated {
				rotated[a] = make([]float64, n)
				for i := 0; i < n; i += 2 {
					re, im := float64(input[a][i]), float64(input[a][i+1])
					if which == a && index == i {
						re += delta
					}
					if which == a && index == i+1 {
						im += delta
					}
					if a < 2 {
						row, pair := i/(heads*dim), (i%dim)/2
						angle := float64(origin+row) * math.Exp(-math.Log(10000)*float64(pair)/float64(dim/2))
						z := complex(re, im) * complex(math.Cos(angle), math.Sin(angle))
						re, im = real(z), imag(z)
					}
					rotated[a][i], rotated[a][i+1] = re, im
				}
			}
			out := make([]float64, n)
			for h := 0; h < heads; h++ {
				for query := 0; query < rows; query++ {
					start := 0
					if contextRows > 0 {
						start = max(0, query-contextRows+1)
					}
					scores := make([]float64, rows)
					maximum := math.Inf(-1)
					for key := start; key <= query; key++ {
						for d := 0; d < dim; d++ {
							scores[key] += rotated[0][(query*heads+h)*dim+d] * rotated[1][(key*heads+h)*dim+d]
						}
						scores[key] /= math.Sqrt(float64(dim))
						maximum = math.Max(maximum, scores[key])
					}
					var sum float64
					for key := start; key <= query; key++ {
						scores[key] = math.Exp(scores[key] - maximum)
						sum += scores[key]
					}
					for key := start; key <= query; key++ {
						p := scores[key] / sum
						for d := 0; d < dim; d++ {
							out[(query*heads+h)*dim+d] += p * rotated[2][(key*heads+h)*dim+d]
						}
					}
				}
			}
			var objective float64
			for i, x := range out {
				objective += x * float64(dy[i])
			}
			return objective, out
		}
		_, reference := evaluate(-1, 0, 0)
		for i, x := range got.Output {
			if math.Abs(float64(x)-reference[i]) > 2e-6 {
				t.Fatal("attention tape independent output", contextRows, i, x, reference[i])
			}
		}
		for which, gradient := range [][]float32{got.DQ, got.DK, got.DV} {
			for i, x := range gradient {
				hi, _ := evaluate(which, i, 1e-4)
				lo, _ := evaluate(which, i, -1e-4)
				ref := (hi - lo) / 2e-4
				if math.Abs(float64(x)-ref) > 2e-6 {
					t.Fatal("attention tape independent VJP", contextRows, which, i, x, ref)
				}
			}
		}
		// Own immutable graph inputs and decoded outputs even after caller reuse.
		q[0] = 42
		if tape.Tape.Tensors[0].Values[0] == 42 {
			t.Fatal("borrowed graph input")
		}
		decoded := interpretFlowTape(t, tape.Tape)
		stable, err := tape.Decode(decoded)
		if err != nil {
			t.Fatal(err)
		}
		decoded[0][0] = 42
		if stable.Output[0] == 42 {
			t.Fatal("borrowed decoded output")
		}
	}
}
func TestTrainingAttentionTapeAdmission(t *testing.T) {
	x := make([]float32, 12)
	for _, v := range []struct {
		rows, heads, dim, context, origin int
		period                            float64
		bytes                             int64
	}{{1, 2, 6, 0, 0, 10000, 1}, {1, 2, 5, 0, 0, 10000, 1 << 20}, {1, 2, 6, 2, 0, 10000, 1 << 20}, {1, 2, 6, 0, -1, 10000, 1 << 20}, {1, 2, 6, 0, 0, 0, 1 << 20}} {
		if _, err := BuildAttentionTrainingTape(x, x, x, x, v.rows, v.heads, v.dim, v.context, v.origin, v.period, v.bytes); err == nil {
			t.Fatal("invalid attention tape")
		}
	}
	if _, err := BuildAttentionTrainingTape(nil, x, x, x, 1, 2, 6, 0, 0, 10000, 1<<20); err == nil {
		t.Fatal("short input")
	}
	x[0] = float32(math.NaN())
	if _, err := BuildAttentionTrainingTape(x, x, x, x, 1, 2, 6, 0, 0, 10000, 1<<20); err == nil {
		t.Fatal("nonfinite input")
	}
	x[0] = 0
	tape, err := BuildAttentionTrainingTape(x, x, x, x, 1, 2, 6, 0, 0, 10000, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	for _, values := range [][][]float32{nil, {x, x, x, x, x}, {make([]float32, 12), make([]float32, 2), x, x, nil}} {
		if _, err := tape.Decode(values); err == nil {
			t.Fatal("bad output")
		}
	}
	values := interpretFlowTape(t, tape.Tape)
	values[0][0] = float32(math.NaN())
	if _, err := tape.Decode(values); err == nil {
		t.Fatal("nonfinite output")
	}
}
