package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedLayer1AttentionPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	layer0, err := LoadLayer0Complete(file)
	if err != nil {
		t.Fatal(err)
	}
	layer1, err := LoadLayer1Attention(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	for _, rows := range []int{16, 138} {
		t.Run(fmt.Sprintf("rows=%d", rows), func(t *testing.T) {
			input := append([]float32(nil), stacked[:rows*projectedWidth]...)
			hidden, err := layer0.ForwardOffline(input, rows)
			if err != nil {
				t.Fatal(err)
			}
			initial := append([]float32(nil), hidden...)
			for _, scalar := range []bool{false, true} {
				var attention, residual []float32
				if scalar {
					attention, residual, err = layer1.forwardOfflineScalar(hidden, rows)
				} else {
					attention, residual, err = layer1.ForwardOffline(hidden, rows)
				}
				if err != nil {
					t.Fatal(err)
				}
				prefix := "jfk_layer1_"
				if rows == 138 {
					prefix = "jfk_full_layer1_"
				}
				for _, item := range []struct {
					name string
					got  []float32
				}{{"attention", attention}, {"residual", residual}} {
					ref := readStackingFixture(t, "testdata/"+prefix+item.name+".f32.gz", rows*projectedWidth)
					var maxAbs, sumAbs float64
					var outside int
					for i, value := range item.got {
						delta := math.Abs(float64(value - ref[i]))
						maxAbs = math.Max(maxAbs, delta)
						sumAbs += delta
						if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
							outside++
						}
					}
					mean := sumAbs / float64(len(item.got))
					t.Logf("scalar=%t %s max_abs=%g mean_abs=%g outside=%d", scalar, item.name, maxAbs, mean, outside)
					if outside != 0 || mean > 2e-6 {
						t.Fatalf("%s differs from PyTorch", item.name)
					}
				}
				if &attention[0] == &residual[0] || &attention[0] == &hidden[0] || &residual[0] == &hidden[0] {
					t.Fatal("output aliases input or other output")
				}
				for i, value := range hidden {
					if value != initial[i] {
						t.Fatalf("mutated layer-0 output %d", i)
					}
				}
				for i, value := range input {
					if value != stacked[i] {
						t.Fatalf("mutated caller input %d", i)
					}
				}
			}
		})
	}
}

func TestLayer1AttentionRejectsMalformed(t *testing.T) {
	if _, err := LoadLayer1Attention(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	if _, _, err := (*Layer1Attention)(nil).ForwardOffline(make([]float32, projectedWidth), 1); err == nil {
		t.Fatal("accepted nil model")
	}
	m := &Layer1Attention{qkv: &Layer1QKV{gamma: make([]float32, projectedWidth), beta: make([]float32, projectedWidth), q: make([]float32, projectedWidth*projectedWidth), k: make([]float32, projectedWidth*projectedWidth), v: make([]float32, projectedWidth*projectedWidth)}, outWeight: make([]float32, projectedWidth*projectedWidth), outBias: make([]float32, projectedWidth)}
	for _, rows := range []int{0, maxPreparedDiarizationRows + 1} {
		if _, _, err := m.ForwardOffline(make([]float32, rows*projectedWidth), rows); err == nil {
			t.Fatalf("accepted rows=%d", rows)
		}
	}
	bad := make([]float32, projectedWidth)
	bad[0] = float32(math.NaN())
	if _, _, err := m.ForwardOffline(bad, 1); err == nil {
		t.Fatal("accepted non-finite input")
	}
}
