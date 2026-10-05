package training

import (
	"math"
	"testing"
)

func TestTrainingEOSAdmissionSoundness(t *testing.T) {
	state := uint32(71)
	next := func() float32 { state = state*1664525 + 1013904223; return math.Float32frombits(state & 0x7f7fffff) }
	accepted, rejected, hostReject := 0, 0, 0
	for _, rows := range []int{1, 2, 7, 1024} {
		for _, width := range []int{1, 3, 64, 16384} {
			for trial := 0; trial < 8; trial++ {
				x, w := make([]float32, rows*width), make([]float32, width)
				bias := next()
				// Half cases are ordinary small inputs, half span all F32 magnitudes.
				if trial < 4 {
					bias = .1
				}
				for i := range w {
					if trial < 4 {
						w[i] = float32(i%11) * .01
					} else {
						w[i] = next()
					}
				}
				for i := range x {
					if trial < 4 {
						x[i] = float32(i%17) * .01
					} else {
						x[i] = next()
					}
				}
				flag, err := EOSAdmissionFlag(x, w, bias, rows, width)
				if err != nil {
					t.Fatal(err)
				}
				limit := float64(math.MaxFloat32) * .125 / float64(rows)
				safe := true
				for row := 0; row < rows; row++ {
					bound := math.Abs(float64(bias))
					for col := range w {
						bound += math.Abs(float64(x[row*width+col]) * float64(w[col]))
					}
					if bound > limit {
						safe = false
					}
				}
				if !safe {
					hostReject++
				}
				if flag == 0 {
					accepted++
					if !safe {
						t.Fatal("integer admission accepted unsafe F64 absolute dot")
					}
				} else {
					rejected++
				}
			}
		}
	}
	t.Logf("cases %d accepted %d rejected %d host_rejected %d false_accept 0", accepted+rejected, accepted, rejected, hostReject)
}
func TestTrainingEOSAdmissionBoundary(t *testing.T) {
	for _, rows := range []int{1, 7, 1024} {
		limit := float64(math.MaxFloat32) * .125 / float64(rows)
		safe := float32(limit * .999)
		x := make([]float32, rows)
		for i := range x {
			x[i] = safe
		}
		flag, err := EOSAdmissionFlag(x, []float32{1}, 0, rows, 1)
		if err != nil || flag != 0 {
			t.Fatal("margin safe rejected", err, flag)
		}
		x[rows-1] = float32(limit * 1.001) // Padding/all rows must be checked.
		if flag, _ = EOSAdmissionFlag(x, []float32{1}, 0, rows, 1); flag != 1 {
			t.Fatal("unsafe last row")
		}
	}
	for _, v := range []float32{float32(math.Inf(1)), float32(math.NaN())} {
		for _, slot := range []int{0, 1, 2} {
			x, w, b := []float32{0}, []float32{0}, float32(0)
			switch slot {
			case 0:
				x[0] = v
			case 1:
				w[0] = v
			case 2:
				b = v
			}
			if flag, _ := EOSAdmissionFlag(x, w, b, 1, 1); flag != 1 {
				t.Fatal("nonfinite even times zero")
			}
		}
	}
	if flag, err := EOSAdmissionFlag([]float32{math.SmallestNonzeroFloat32}, []float32{math.MaxFloat32}, 0, 1, 1); err != nil || flag != 0 {
		t.Fatal("subnormal bitwise admission")
	}
}
func TestTrainingEOSAdmissionShape(t *testing.T) {
	for _, pair := range [][2]int{{0, 1}, {1025, 1}, {1, 0}, {1, 16385}} {
		if _, _, err := EOSAdmissionShape(pair[0], pair[1]); err == nil {
			t.Fatal("shape")
		}
	}
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{.1, .2}}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}}}, Commands: []Command{{Op: "eos-admission", Inputs: []int{0, 1, 2}, Output: 3, Rows: 1, Width: 2}}, Outputs: []int{3}}
	if err := tape.Validate(1024); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"inputs", "rows", "shape", "dead"} {
		p := copyHandoffTape(tape)
		switch kind {
		case "inputs":
			p.Commands[0].Inputs = p.Commands[0].Inputs[:2]
		case "rows":
			p.Commands[0].Rows = 0
		case "shape":
			p.Tensors[3].Shape = []int{2}
		case "dead":
			p.Tensors = append(p.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
		}
		if p.Validate(1024) == nil {
			t.Fatal("source", kind)
		}
	}
	if _, err := EOSAdmissionFlag(nil, nil, 0, 1, 1); err == nil {
		t.Fatal("extent")
	}
}

func TestTrainingEOSAdmissionProductBounds(t *testing.T) {
	state := uint32(811)
	next := func() float32 { state = state*1664525 + 1013904223; return math.Float32frombits(state & 0x7f7fffff) }
	accepted, rejected := 0, 0
	maximumRelative := float64(0)
	for i := 0; i < 100000; i++ {
		a, b := next(), next()
		remaining := uint32(1 << 30)
		units, ok := eosAdmissionUnits(a, b, remaining)
		// Product of two F32 values is exact in F64 (<=48 significand bits).
		exact := math.Abs(float64(a) * float64(b))
		upper := math.Ldexp(float64(units), 95)
		if ok {
			accepted++
			if upper < exact || units > remaining {
				t.Fatal("downward product bound", a, b, units, exact)
			}
			if exact >= math.Ldexp(1, 95) {
				maximumRelative = math.Max(maximumRelative, (upper-exact)/exact)
			}
		} else {
			rejected++
		}
	}
	t.Logf("100000 independent exact-F64 product checks accepted %d rejected %d max relative overbound for terms>=2^95 %.9g", accepted, rejected, maximumRelative)
}
func TestTrainingEOSAdmissionConservativeMargin(t *testing.T) {
	// The mantissa ceiling may reject a host-safe boundary. Quantify it rather
	// than calling this conservative predicate numerically identical admission.
	for _, width := range []int{1, 1024, 16384} {
		rows := 1024
		limit := float64(math.MaxFloat32) * .125 / float64(rows)
		input := float32(limit / float64(width))
		x, w := make([]float32, rows*width), make([]float32, width)
		for i := range x {
			x[i] = input
		}
		for i := range w {
			w[i] = 1
		}
		flag, err := EOSAdmissionFlag(x, w, 0, rows, width)
		if err != nil {
			t.Fatal(err)
		}
		safe := float64(input)*float64(width) <= limit
		if !safe { // nearest F32 may exceed the original exact bound; force safe.
			input = math.Nextafter32(input, 0)
			for i := range x {
				x[i] = input
			}
			flag, _ = EOSAdmissionFlag(x, w, 0, rows, width)
		}
		if flag != 1 {
			t.Fatal("expected conservative boundary rejection")
		}
		// Document worst geometry's additive fixed-point rounding. A 2% margin
		// clears mantissa+unit overhead across the supported domain.
		for i := range x {
			x[i] = input * .98
		}
		flag, _ = EOSAdmissionFlag(x, w, 0, rows, width)
		if flag != 0 {
			t.Fatal("2percent margin")
		}
		t.Logf("rows=%d width=%d original-host-safe nearboundary rejected; 2percent margin accepted", rows, width)
	}
}

func TestTrainingEOSAdmissionExhaustedBudget(t *testing.T) {
	if _, ok := eosAdmissionUnits(math.SmallestNonzeroFloat32, 1, 0); ok {
		t.Fatal("nonzero exhausted fixed-point budget")
	}
	for _, sign := range []float32{-1, 1} {
		if flag, err := EOSAdmissionFlag([]float32{sign * math.SmallestNonzeroFloat32, sign * .1}, []float32{math.MaxFloat32, sign * .2}, sign*.1, 1, 2); err != nil || flag != 0 {
			t.Fatal("signed finite values", flag, err)
		}
	}
}
