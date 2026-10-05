package training

import (
	"math"
	"testing"
)

func TestTrainingRoPEAdmission(t *testing.T) {
	for _, v := range []struct {
		rows, heads, dim int
		mode             uint32
	}{{1, 1, 2, 0}, {1024, 32, 256, 1}, {13, 3, 14, 0}} {
		if err := RoPEShape(v.rows, v.heads, v.dim, v.mode); err != nil {
			t.Fatal(err)
		}
	}
	for _, v := range []struct {
		rows, heads, dim int
		mode             uint32
	}{{0, 1, 2, 0}, {1025, 1, 2, 0}, {1, 0, 2, 0}, {1, 33, 2, 0}, {1, 1, 1, 0}, {1, 1, 3, 0}, {1, 1, 258, 0}, {1, 1, 2, 2}} {
		if err := RoPEShape(v.rows, v.heads, v.dim, v.mode); err == nil {
			t.Fatal("invalid shape")
		}
	}
	for _, v := range []struct {
		rows, dim, origin int
		period            float64
	}{{0, 2, 0, 10000}, {1, 3, 0, 10000}, {2, 4, -1, 10000}, {2, 4, 1 << 24, 10000}, {2, 4, 0, 0}, {2, 4, 0, math.NaN()}, {2, 4, 0, math.Inf(1)}, {2, 4, 0, -1}, {2, 256, 0, math.SmallestNonzeroFloat64}} {
		dst := make([]float32, 512)
		for i := range dst {
			dst[i] = 42
		}
		n := max(0, min(len(dst), v.rows*v.dim))
		err := RoPETableInto(dst[:n], v.rows, v.dim, v.origin, v.period)
		if err == nil {
			t.Fatal("invalid table geometry admitted", v)
		}
		for _, x := range dst {
			if x != 42 {
				t.Fatal("failed admission wrote table")
			}
		}
	}
	if err := RoPETableInto(make([]float32, 3), 2, 2, 0, 10000); err == nil {
		t.Fatal("short table")
	}
	tape := ropeTapeFixture(t)
	if err := tape.Validate(1 << 20); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Tape){func(t *Tape) { t.Commands[0].Inputs = nil }, func(t *Tape) { t.Tensors[0].Shape = []int{4, 6} }, func(t *Tape) { t.Tensors[1].Shape = []int{4, 2, 3} }, func(t *Tape) { t.Tensors[2].Shape = []int{4, 2, 3} }, func(t *Tape) { t.Commands[0].Mode = 2 }, func(t *Tape) { t.Commands[0].Output = 0 }} {
		bad := ropeTapeFixture(t)
		mutate(&bad)
		if err := bad.Validate(1 << 20); err == nil {
			t.Fatal("bad rope tape")
		}
	}
}
func ropeTapeFixture(t *testing.T) Tape {
	t.Helper()
	table := make([]float32, 24)
	if err := RoPETableInto(table, 4, 6, 7, 10000); err != nil {
		t.Fatal(err)
	}
	return Tape{Tensors: []TensorSpec{{Shape: []int{4, 1, 6}, Values: make([]float32, 24)}, {Shape: []int{4, 3, 2}, Values: table}, {Shape: []int{4, 1, 6}}}, Commands: []Command{{Op: "rope", Inputs: []int{0, 1}, Output: 2}}, Outputs: []int{2}}
}

// This host expression models the shader's two rounded multiplies and add/sub.
// Independent oracle uses complex multiplication and F64 trig; the adjoint
// reference is an F64 scalar-objective central difference, not this expression.
func ropeSchedule(x, table []float32, rows, heads, dim int, inverse bool) []float32 {
	out := make([]float32, len(x))
	for i := 0; i < len(x); i += 2 {
		row, col := i/(heads*dim), i%dim
		c, s := table[row*dim+col], table[row*dim+col+1]
		if inverse {
			s = -s
		}
		out[i] = x[i]*c - x[i+1]*s
		out[i+1] = x[i+1]*c + x[i]*s
	}
	return out
}
func TestTrainingRoPEHostNumerics(t *testing.T) {
	maxError, square, count := float64(0), float64(0), 0
	for _, v := range []struct {
		rows, heads, dim, origin int
		period                   float64
	}{{1, 1, 2, 0, 10000}, {3, 2, 6, 7, 10000}, {13, 3, 14, 137, 4096}, {2, 1, 256, (1 << 24) - 2, 10000}, {3, 1, 4, 0, .5}} {
		table := make([]float32, v.rows*v.dim)
		if err := RoPETableInto(table, v.rows, v.dim, v.origin, v.period); err != nil {
			t.Fatal(err)
		}
		x, g := make([]float32, v.rows*v.heads*v.dim), make([]float32, v.rows*v.heads*v.dim)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*.7) * 3)
			g[i] = float32(math.Cos(float64(i)*.9) * .4)
		}
		want := func(input []float32) []float64 {
			out := make([]float64, len(input))
			for i := 0; i < len(input); i += 2 {
				row, pair := i/(v.heads*v.dim), (i%v.dim)/2
				theta := float64(v.origin+row) * math.Exp(-math.Log(v.period)*float64(pair)/float64(v.dim/2))
				z := complex(float64(input[i]), float64(input[i+1])) * complex(math.Cos(theta), math.Sin(theta))
				out[i], out[i+1] = real(z), imag(z)
			}
			return out
		}
		got, reference := ropeSchedule(x, table, v.rows, v.heads, v.dim, false), want(x)
		for i, y := range got {
			d := math.Abs(float64(y) - reference[i])
			if d > 1e-6 {
				t.Fatalf("forward[%d] drift=%g", i, d)
			}
			maxError = math.Max(maxError, d)
			square += d * d
			count++
		}
		// Round-trip is supplementary: both directions could share a wiring bug.
		back := ropeSchedule(got, table, v.rows, v.heads, v.dim, true)
		for i, y := range back {
			if math.Abs(float64(y-x[i])) > 1.5e-6 {
				t.Fatal("roundtrip", i, y, x[i])
			}
		}
		analytic := ropeSchedule(g, table, v.rows, v.heads, v.dim, true)
		objective := func(index int, delta float64) float64 {
			var sum float64
			for i := 0; i < len(x); i += 2 {
				row, pair := i/(v.heads*v.dim), (i%v.dim)/2
				theta := float64(v.origin+row) * math.Exp(-math.Log(v.period)*float64(pair)/float64(v.dim/2))
				re, im := float64(x[i]), float64(x[i+1])
				if index == i {
					re += delta
				}
				if index == i+1 {
					im += delta
				}
				z := complex(re, im) * complex(math.Cos(theta), math.Sin(theta))
				sum += real(z)*float64(g[i]) + imag(z)*float64(g[i+1])
			}
			return sum
		}
		for i, y := range analytic {
			ref := (objective(i, 1e-4) - objective(i, -1e-4)) / 2e-4
			d := math.Abs(float64(y) - ref)
			if d > 2e-7 {
				t.Fatalf("adjoint[%d] drift=%g", i, d)
			}
			maxError = math.Max(maxError, d)
			square += d * d
			count++
		}
	}
	t.Logf("host RoPE forward/adjoint values=%d max_abs=%g rms=%g", count, maxError, math.Sqrt(square/float64(count)))
	dst := make([]float32, 24)
	if allocs := testing.AllocsPerRun(10, func() {
		if err := RoPETableInto(dst, 4, 6, 7, 10000); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatal("table builder allocated", allocs)
	}
}
