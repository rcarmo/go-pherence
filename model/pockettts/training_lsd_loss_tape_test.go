package pockettts

import (
	"math"
	"testing"
)

func TestTrainingLSDLossTapeReference(t *testing.T) {
	var maxSquare, maxLoss, maxSeed, maxLogvar float64
	cases := 0
	for _, channels := range []int{1, 3, 32, 257, 16384} {
		for _, logvar := range []float32{-8, -1, 0, 1, 8} {
			prediction, desired := make([]float32, channels), make([]float32, channels)
			for i := range prediction {
				prediction[i] = float32(i%19-9) * .037
				desired[i] = float32(i%13-6) * .051
			}
			tape, err := BuildLSDLossTrainingTape(prediction, desired, logvar, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			got := interpretFlowTape(t, tape)
			var square float64
			for i, p := range prediction {
				d := float64(p - desired[i])
				square += d * d
			}
			scale := math.Exp(float64(logvar)) / float64(channels)
			wantLoss, wantLogvar := square*scale-float64(logvar), square*scale-1
			check := func(name string, got, want float64, absolute, relative float64) float64 {
				e := math.Abs(got - want)
				if e > absolute+relative*math.Abs(want) {
					t.Fatalf("%s c%d l%g got %.12g want %.12g error %g", name, channels, logvar, got, want, e)
				}
				return e
			}
			maxSquare = math.Max(maxSquare, check("square", float64(got[0][0]), square, 1e-7, 2e-7))
			maxLoss = math.Max(maxLoss, check("loss", float64(got[0][1]), wantLoss, 1e-6, 3e-7))
			maxLogvar = math.Max(maxLogvar, check("dLogvar", float64(got[0][2]), wantLogvar, 2e-7, 3e-7))
			for i, p := range prediction {
				want := 2 * float64(p-desired[i]) * scale
				maxSeed = math.Max(maxSeed, check("prediction seed", float64(got[1][i]), want, 1e-7, 2e-7))
			}
			cases++
		}
	}
	t.Logf("LSD %d F32/F64 cases: max square=%g loss=%g seed=%g logvar-gradient=%g", cases, maxSquare, maxLoss, maxSeed, maxLogvar)
	// Independent F64 finite difference of the normalized loss wrt prediction
	// and logvar; constants held fixed. Expected result is not a tape golden.
	p, d, l := .7, -.2, .4
	objective := func(p, l float64) float64 { return (p-d)*(p-d)*math.Exp(l) - l }
	epsilon := 1e-6
	dp := (objective(p+epsilon, l) - objective(p-epsilon, l)) / (2 * epsilon)
	dl := (objective(p, l+epsilon) - objective(p, l-epsilon)) / (2 * epsilon)
	if math.Abs(dp-2*(p-d)*math.Exp(l)) > 1e-9 || math.Abs(dl-((p-d)*(p-d)*math.Exp(l)-1)) > 1e-9 {
		t.Fatal("finite difference", dp, dl)
	}
	// Zero residual permits negative losses (not BCE); boundary +/-80 valid.
	for _, l := range []float32{-80, 80} {
		tape, err := BuildLSDLossTrainingTape([]float32{1}, []float32{1}, l, 1024)
		if err != nil {
			t.Fatal(err)
		}
		got := interpretFlowTape(t, tape)
		if got[0][0] != 0 || got[0][1] != -l || got[0][2] != -1 || got[1][0] != 0 {
			t.Fatal("zero residual", l, got)
		}
	}
}
func TestTrainingLSDLossTapeAdmission(t *testing.T) {
	for _, bad := range []struct {
		p, d []float32
		l    float32
		b    int64
	}{{nil, nil, 0, 1024}, {[]float32{1}, nil, 0, 1024}, {[]float32{1}, []float32{0}, 81, 1024}, {[]float32{1}, []float32{0}, -81, 1024}, {[]float32{1}, []float32{0}, 0, 1}, {[]float32{float32(math.NaN())}, []float32{0}, 0, 1024}, {[]float32{float32(math.MaxFloat32)}, []float32{-1}, 0, 1024}, {[]float32{1e10}, []float32{0}, 80, 1024}} {
		if _, err := BuildLSDLossTrainingTape(bad.p, bad.d, bad.l, bad.b); err == nil {
			t.Fatal("invalid LSD loss")
		}
	}
	r := LSDLossTrainingResult{RawSquare: 1, Loss: -2, DLogVariance: 0, DPrediction: []float32{1}}
	if err := validateLSDLossBoundary(r, []float32{0}, []float32{0}); err != nil {
		t.Fatal("negative loss valid", err)
	}
	for _, mutate := range []func(*LSDLossTrainingResult){func(r *LSDLossTrainingResult) { r.RawSquare = -1 }, func(r *LSDLossTrainingResult) { r.Loss = math.NaN() }, func(r *LSDLossTrainingResult) { r.DLogVariance = float32(math.Inf(1)) }, func(r *LSDLossTrainingResult) { r.DPrediction = nil }} {
		bad := r
		mutate(&bad)
		if validateLSDLossBoundary(bad, []float32{0}, []float32{0}) == nil {
			t.Fatal("bad result")
		}
	}
	if validateLSDLossBoundary(r, r.DPrediction, []float32{0}) == nil {
		t.Fatal("borrowed")
	}
}
