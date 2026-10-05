package training

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// This models the shader's explicit non-contracted arithmetic schedule on the
// host; it does NOT execute SPIR-V/PTX or qualify a device implementation.
func adamSchedule(p AdamWStep, parameter, g, m, v, ema float32) (float32, float32, float32, float32) {
	g *= p.GradientScale
	m = p.Beta1*m + (1-p.Beta1)*g
	v = p.Beta2*v + ((1-p.Beta2)*g)*g
	mh, vh := m/p.Bias1, v/p.Bias2
	parameter *= 1 - p.LearningRate*p.WeightDecay
	parameter -= p.LearningRate * mh / (float32(math.Sqrt(float64(vh))) + p.Epsilon)
	if p.EMADecay > 0 {
		ema = p.EMADecay*ema + (1-p.EMADecay)*parameter
	}
	return parameter, m, v, ema
}
func TestTrainingAdamWPyTorchSchedule(t *testing.T) {
	data, err := os.ReadFile("../../model/pockettts/testdata/training_step_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle struct {
		Schema    int                  `json:"schema"`
		Revision  string               `json:"upstream_revision"`
		Params    map[string][]float32 `json:"parameters"`
		Gradients map[string][]float32 `json:"gradients"`
		Adam      struct {
			LR      float32              `json:"learning_rate"`
			B1      float32              `json:"beta1"`
			B2      float32              `json:"beta2"`
			Eps     float32              `json:"epsilon"`
			Decay   float32              `json:"weight_decay"`
			EMA     float32              `json:"ema_decay"`
			Params  map[string][]float32 `json:"parameters_after_step"`
			Shadows map[string][]float32 `json:"ema_after_step"`
		} `json:"adamw"`
	}
	if err := json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.Revision != "0acce6b2f390150267557770d2098c5caa9a18ac" {
		t.Fatal("oracle provenance", oracle.Schema, oracle.Revision)
	}
	a := oracle.Adam
	p, err := NewAdamWStep(a.LR, a.B1, a.B2, a.Eps, a.Decay, a.EMA, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	var max, RMS float64
	for name, want := range a.Params {
		// The source parameter manifest also includes frozen buffers (latent
		// statistics/frequencies). Only the optimiser output names trainables.
		values, gradients, shadow := oracle.Params[name], oracle.Gradients[name], a.Shadows[name]
		if len(gradients) != len(values) || len(want) != len(values) || len(shadow) != len(values) {
			t.Fatal("fixture shape", name)
		}
		for i, v := range values {
			got, _, _, ema := adamSchedule(p, v, gradients[i], 0, 0, v)
			for j, pair := range [][2]float32{{got, want[i]}, {ema, shadow[i]}} {
				d := math.Abs(float64(pair[0]) - float64(pair[1]))
				max = math.Max(max, d)
				RMS += d * d
				count++
				// Reuse the pre-existing CPU/PyTorch optimiser tolerance (4e-6),
				// unchanged. Device error needs its own measured qualification.
				if d > 4e-6 || math.IsNaN(d) {
					t.Fatal("schedule differs from independent PyTorch fixture", name, i, j, d)
				}
			}
		}
	}
	if count == 0 {
		t.Fatal("empty oracle")
	}
	t.Logf("CPU schedule vs independent PyTorch: values=%d max_abs=%g RMS=%g", count, max, math.Sqrt(RMS/float64(count)))
}
func TestTrainingSiLUDerivativeSchedule(t *testing.T) {
	for _, x := range []float64{-100, -10, -1, -.01, 0, .01, 1, 10, 100} {
		// Independent central difference of the analytic forward, evaluated in
		// F64. This checks the derivative expression, not a device exp intrinsic.
		forward := func(v float64) float64 { return v / (1 + math.Exp(-v)) }
		h := 1e-4
		want := (forward(x+h) - forward(x-h)) / (2 * h)
		xf := float32(x)
		e := float32(math.Exp(-math.Abs(x)))
		s := float32(1) / (1 + e)
		if xf < 0 {
			s = e / (1 + e)
		}
		got := s * (1 + xf*(1-s))
		if math.Abs(float64(got)-want) > 2e-6 {
			t.Fatal(x, got, want)
		}
	}
}
func TestTrainingLinearAnalyticSchedule(t *testing.T) {
	// Y=X W^T, scalar objective=sum(Y*dY). Independently differentiated
	// pencil-and-paper fixture: X=[[1,2],[3,4]], W=[[5,6],[7,8]],
	// dY=[[2,3],[4,5]]. Expected dX,dW,dB follow this objective directly.
	x, w, dy := []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}, []float32{2, 3, 4, 5}
	dx, dw, db := make([]float32, 4), make([]float32, 4), make([]float32, 2)
	for i := 0; i < 4; i++ {
		row, col := i/2, i%2
		for q := 0; q < 2; q++ {
			dx[i] += dy[row*2+q] * w[q*2+col]
			dw[i] += dy[q*2+row] * x[q*2+col]
		}
	}
	for i := range db {
		for q := 0; q < 2; q++ {
			db[i] += dy[q*2+i]
		}
	}
	for _, tc := range []struct{ got, want []float32 }{{dx, []float32{31, 36, 55, 64}}, {dw, []float32{14, 20, 18, 26}}, {db, []float32{6, 8}}} {
		for i, v := range tc.want {
			if tc.got[i] != v {
				t.Fatal("exact small integer derivative", tc.got, tc.want)
			}
		}
	}
}
