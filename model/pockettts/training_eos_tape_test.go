package pockettts

import (
	"math"
	"testing"
)

func TestTrainingEOSTapeReference(t *testing.T) {
	var maxLoss, maxGradient, maxTailRelative float64
	cases := 0
	for _, n := range []int{1, 2, 7, 257, 1024} {
		for _, valid := range []int{1, n} {
			for _, weight := range []float32{0, .1, 1} {
				logits := make([]float32, n)
				mask := make([]bool, n)
				pattern := []float32{-100, -40, -16, -2, 0, 2, 16, 40, 100}
				for i := range logits {
					logits[i] = pattern[i%len(pattern)]
					mask[i] = i < valid
				}
				tape, err := BuildEOSTrainingTape(logits, mask, weight, 1<<20)
				if err != nil {
					t.Fatal(err)
				}
				got := interpretFlowTape(t, tape)
				loss, gradient, err := EOSLossAndGradient(logits, mask)
				if err != nil {
					t.Fatal(err)
				}
				lossErr := math.Abs(float64(got[0][0]) - loss)
				maxLoss = math.Max(maxLoss, lossErr)
				if lossErr > 2e-6+1e-7*math.Abs(loss) {
					t.Fatal("EOS F64 loss", n, valid, weight, got[0], loss)
				}
				for i, want := range gradient {
					want *= weight
					e := math.Abs(float64(got[1][i]) - float64(want))
					maxGradient = math.Max(maxGradient, e)
					if e > 6e-8 {
						t.Fatal("EOS F64 gradient", i, got[1][i], want)
					}
				}
				cases++
			}
		}
	}
	// Tail-preserving positive-target gradient uses an independent stable F64
	// residual, not sigmoid(x)-1 cancellation in the legacy CPU implementation.
	for _, x := range []float32{0, 1, 8, 16, 20, 40, 80} {
		logits := []float32{-1, x}
		tape, err := BuildEOSTrainingTape(logits, []bool{true, false}, 1, 1024)
		if err != nil {
			t.Fatal(err)
		}
		got := interpretFlowTape(t, tape)
		want := -math.Exp(-float64(x)) / (1 + math.Exp(-float64(x))) / 2
		rel := math.Abs(float64(got[1][1])-want) / math.Abs(want)
		maxTailRelative = math.Max(maxTailRelative, rel)
		if rel > 2e-7 {
			t.Fatal("tail gradient", x, got[1][1], want, rel)
		}
	}
	// Exercise the GLSL log1p series separately against independent math.Log1p.
	maxSeries := float64(0)
	for i := 0; i <= 10000; i++ {
		e := float32(.125) * float32(i) / 10000
		var sum float32
		if e >= .125 {
			sum = float32(math.Log(float64(1 + e)))
		} else {
			term := e
			sum = e
			for k := uint32(2); k <= 12; k++ {
				term *= e
				v := term / float32(k)
				if k%2 == 0 {
					sum -= v
				} else {
					sum += v
				}
			}
		}
		err := math.Abs(float64(sum) - math.Log1p(float64(e)))
		maxSeries = math.Max(maxSeries, err)
		if err > 2e-8 {
			t.Fatal("series", e, sum, err)
		}
	}
	t.Logf("EOS %d cases: max mean-loss error=%g gradient=%g tail-relative=%g log1p-series=%g", cases, maxLoss, maxGradient, maxTailRelative, maxSeries)
}
func TestTrainingEOSTapeAdmission(t *testing.T) {
	for _, bad := range []struct {
		x []float32
		m []bool
		w float32
		b int64
	}{{nil, nil, 1, 1024}, {[]float32{0}, nil, 1, 1024}, {[]float32{0}, []bool{false}, 1, 1024}, {[]float32{0, 0}, []bool{false, true}, 1, 1024}, {[]float32{0}, []bool{true}, -1, 1024}, {[]float32{0}, []bool{true}, 1, 1}, {[]float32{float32(math.NaN())}, []bool{true}, 1, 1024}, {[]float32{float32(math.MaxFloat32)}, []bool{true}, 1, 1024}} {
		if _, err := BuildEOSTrainingTape(bad.x, bad.m, bad.w, bad.b); err == nil {
			t.Fatal("invalid tape")
		}
	}
	tape, err := BuildEOSTrainingTape([]float32{1, 2, float32(math.NaN())}, []bool{true, false, false}, 1, 1024)
	if err != nil || tape.Tensors[0].Values[2] != 0 {
		t.Fatal("ignored nonfinite padding", err)
	}
	valid := [][]float32{{1}, {.2, 0}}
	if _, _, err := validateEOSBoundary(valid, []float32{0, 0}, []bool{true, true}); err != nil {
		t.Fatal(err)
	}
	for _, v := range [][][]float32{nil, {{-1}, {0, 0}}, {{float32(math.NaN())}, {0, 0}}, {{1}, nil}, {{1}, {0, 1}}} {
		if _, _, err := validateEOSBoundary(v, []float32{0, 0}, []bool{false, false}); err == nil {
			t.Fatal("invalid result")
		}
	}
	borrowed := []float32{1, 2}
	if _, _, err := validateEOSBoundary([][]float32{{1}, borrowed}, borrowed, []bool{true, true}); err == nil {
		t.Fatal("borrowed")
	}
}
