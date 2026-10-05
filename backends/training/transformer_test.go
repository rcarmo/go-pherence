package training

import (
	"math"
	"testing"
)

func TestTrainingTransformerPrimitiveAdmission(t *testing.T) {
	for _, n := range []int{1, 257, 1 << 26} {
		for mode := uint32(0); mode < 2; mode++ {
			if GELUShape(n, mode) != nil {
				t.Fatal("valid GELU")
			}
		}
	}
	for _, n := range []int{0, -1, 1<<26 + 1} {
		if GELUShape(n, 0) == nil {
			t.Fatal("invalid GELU")
		}
	}
	if GELUShape(1, 2) == nil {
		t.Fatal("invalid GELU mode")
	}
	for mode := uint32(0); mode < 4; mode++ {
		parts, slot := 1, 0
		if mode < 2 {
			parts, slot = 3, 2
		}
		a, b, n, err := RowShape(3, 5, parts, slot, mode)
		if err != nil || a < 1 || b < 1 || n < 1 {
			t.Fatal(a, b, n, err)
		}
	}
	for _, s := range []struct {
		r, w, p, s int
		m          uint32
	}{{0, 5, 3, 0, 0}, {1025, 5, 3, 0, 0}, {1, 0, 3, 0, 0}, {1, 16385, 3, 0, 0}, {1, 5, 0, 0, 0}, {1, 5, 5, 0, 0}, {1, 5, 3, -1, 0}, {1, 5, 3, 3, 0}, {1, 5, 3, 0, 2}, {1, 5, 3, 0, 3}, {1, 5, 1, 0, 4}} {
		if _, _, _, err := RowShape(s.r, s.w, s.p, s.s, s.m); err == nil {
			t.Fatal("invalid row")
		}
	}
	gelu := Tape{Tensors: []TensorSpec{{Shape: []int{3}, Values: []float32{1, 2, 3}}, {Shape: []int{3}, Values: []float32{1, 2, 3}}, {Shape: []int{3}}}, Commands: []Command{{Op: "gelu", Inputs: []int{0, 1}, Output: 2}}, Outputs: []int{2}}
	if gelu.Validate(1024) != nil {
		t.Fatal("valid gelu tape")
	}
	gelu.Commands[0].Inputs = nil
	if gelu.Validate(1024) == nil {
		t.Fatal("invalid gelu input")
	}
	for mode := uint32(0); mode < 4; mode++ {
		parts, slot := 1, 0
		if mode < 2 {
			parts, slot = 3, 2
		}
		an, bn, n, _ := RowShape(3, 5, parts, slot, mode)
		tape := Tape{Tensors: []TensorSpec{{Shape: []int{an}, Values: make([]float32, an)}, {Shape: []int{bn}, Values: make([]float32, bn)}, {Shape: []int{n}}}, Commands: []Command{{Op: "row", Inputs: []int{0, 1}, Output: 2, Rows: 3, Width: 5, Parts: parts, Slot: slot, Mode: mode}}, Outputs: []int{2}}
		if err := tape.Validate(1024); err != nil {
			t.Fatal(err)
		}
		tape.Commands[0].Inputs = nil
		if tape.Validate(1024) == nil {
			t.Fatal("invalid row input")
		}
		tape.Commands[0].Inputs = []int{0, 1}
		tape.Tensors[2].Shape = []int{n + 1}
		if tape.Validate(1024) == nil {
			t.Fatal("invalid row output")
		}
		tape.Tensors[2].Shape = []int{n}
		tape.Commands[0].Parts = 0
		if tape.Validate(1024) == nil {
			t.Fatal("invalid row mode")
		}
	}
}
func TestTrainingGELUHostNumerics(t *testing.T) {
	maxError, square, count := float64(0), float64(0), 0
	for i := -160; i <= 160; i++ {
		x := float32(i) * .1
		u := float32(.7978845608028654) * (x + ((float32(.044715)*x)*x)*x)
		e := float32(math.Exp(float64(-2 * float32(math.Abs(float64(u))))))
		halfOnePlus := float32(1) / (1 + e)
		if u < 0 {
			halfOnePlus = e / (1 + e)
		}
		sech2 := (4 * e) / ((1 + e) * (1 + e))
		value := x * halfOnePlus
		du := float32(.7978845608028654) * (1 + (float32(.134145)*x)*x)
		derivative := halfOnePlus + ((.5*x)*sech2)*du
		ref := func(v float64) float64 { return .5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+.044715*v*v*v))) }
		for j, v := range []float32{value, derivative} {
			want := ref(float64(x))
			if j == 1 {
				want = (ref(float64(x)+1e-4) - ref(float64(x)-1e-4)) / 2e-4
			}
			err := math.Abs(float64(v) - want)
			if err > 6e-7 {
				t.Fatalf("GELU host x%g mode%d drift=%g", x, j, err)
			}
			maxError = math.Max(maxError, err)
			square += err * err
			count++
		}
	}
	t.Logf("GELU host values=%d max_abs=%g rms=%g", count, maxError, math.Sqrt(square/float64(count)))
}
func TestTrainingRowIndependentReference(t *testing.T) {
	// Distinct integer labels identify every row, part and column. Expected
	// gather/scatter uses nested loops, separate from the device flat schedule.
	rows, width, parts := 3, 5, 3
	packed := make([]float32, rows*parts*width)
	for r := 0; r < rows; r++ {
		for p := 0; p < parts; p++ {
			for c := 0; c < width; c++ {
				packed[(r*parts+p)*width+c] = float32(100*r + 10*p + c)
			}
		}
	}
	for slot := 0; slot < parts; slot++ {
		out := make([]float32, rows*width)
		for i := range out {
			out[i] = packed[(i/width*parts+slot)*width+i%width]
		}
		for r := 0; r < rows; r++ {
			for c := 0; c < width; c++ {
				if out[r*width+c] != float32(100*r+10*slot+c) {
					t.Fatal("gather")
				}
			}
		}
		scatter := make([]float32, len(packed))
		for i := range scatter {
			p, r, c := (i/width)%parts, i/(parts*width), i%width
			if p == slot {
				scatter[i] = out[r*width+c]
			}
		}
		for r := 0; r < rows; r++ {
			for p := 0; p < parts; p++ {
				for c := 0; c < width; c++ {
					want := float32(0)
					if p == slot {
						want = out[r*width+c]
					}
					if scatter[(r*parts+p)*width+c] != want {
						t.Fatal("scatter")
					}
				}
			}
		}
	}
	for c := 0; c < width; c++ {
		var reduced float32
		var want float64
		for r := 0; r < rows; r++ {
			x, y := float32(r+c+1), float32(2*r-c)
			reduced += x * y
			want += float64(x) * float64(y)
		}
		if float64(reduced) != want {
			t.Fatal("row dot reduction")
		}
	}
}
