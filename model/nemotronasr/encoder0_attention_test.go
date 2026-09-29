package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedEncoder0AttentionPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadEncoder0Attention(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	positions := encoder0RelativePositions(5)
	refPositions := readStemFixture(t, "encoder0_attn_positions", 9*encoderWidth)
	var positionMax, positionSum float64
	for i, value := range positions {
		delta := math.Abs(float64(value - refPositions[i]))
		positionMax = math.Max(positionMax, delta)
		positionSum += delta
		if math.IsNaN(float64(value)) || delta > 2e-6 {
			t.Fatalf("relative position %d delta=%g", i, delta)
		}
	}
	t.Logf("relative position max_abs=%g mean_abs=%g", positionMax, positionSum/float64(len(positions)))
	input := readStemFixture(t, "encoder0_ff1_residual", 5*encoderWidth)
	original := append([]float32(nil), input...)
	attention, residual, err := m.ForwardOffline(input, 5)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct {
		name string
		got  []float32
	}{{"output", attention}, {"residual", residual}} {
		ref := readStemFixture(t, "encoder0_attn_"+item.name, 5*encoderWidth)
		var maxAbs, sumAbs float64
		var outside int
		for i, actual := range item.got {
			delta := math.Abs(float64(actual - ref[i]))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(actual)) || math.IsInf(float64(actual), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
				outside++
			}
		}
		mean := sumAbs / float64(len(item.got))
		t.Logf("encoder0 attention %s max_abs=%g mean_abs=%g outside=%d", item.name, maxAbs, mean, outside)
		if outside != 0 || mean > 2e-5 {
			t.Fatalf("attention %s differs from independent reference", item.name)
		}
	}
	for i := range input {
		if input[i] != original[i] {
			t.Fatalf("mutated caller input %d", i)
		}
	}
	if &attention[0] == &residual[0] || &attention[0] == &input[0] || &residual[0] == &input[0] {
		t.Fatal("attention outputs alias caller or one another")
	}
}

func TestReleasedEncoder0AttentionLookaheadParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	m, loadErr := LoadEncoder0Attention(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	input := readStemFixture(t, "encoder0_ff1_residual", 5*encoderWidth)
	original := append([]float32(nil), input...)
	for _, lookahead := range []int{0, 3} {
		mask := readStemFixture(t, fmt.Sprintf("encoder0_attn_mask_look%d", lookahead), 25)
		for row := 0; row < 5; row++ {
			for source := 0; source < 5; source++ {
				want := float32(0)
				if source/(lookahead+1) <= row/(lookahead+1) {
					want = 1
				}
				if mask[row*5+source] != want {
					t.Fatalf("lookahead=%d mask row=%d source=%d got=%g want=%g", lookahead, row, source, mask[row*5+source], want)
				}
			}
		}
		output, residual, err := m.ForwardOfflineLookahead(input, 5, lookahead)
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range []struct {
			name string
			got  []float32
		}{{"output", output}, {"residual", residual}} {
			ref := readStemFixture(t, fmt.Sprintf("encoder0_attn_%s_look%d", item.name, lookahead), 5*encoderWidth)
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
			t.Logf("lookahead=%d %s max_abs=%g mean_abs=%g outside=%d", lookahead, item.name, maxAbs, mean, outside)
			if outside != 0 || mean > 2e-5 {
				t.Fatalf("lookahead=%d %s differs from independent reference", lookahead, item.name)
			}
		}
	}
	for i, value := range input {
		if value != original[i] {
			t.Fatalf("mutated input at %d", i)
		}
	}
}

func TestEncoder0AttentionRejectsMalformed(t *testing.T) {
	if _, err := LoadEncoder0Attention(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	m := &Encoder0Attention{}
	if _, _, err := m.ForwardOffline(make([]float32, encoderWidth), 1); err == nil {
		t.Fatal("accepted missing weights")
	}
	m = &Encoder0Attention{qkv: &Encoder0QKV{gamma: make([]float32, encoderWidth), beta: make([]float32, encoderWidth), q: make([]float32, encoderWidth*encoderWidth), k: make([]float32, encoderWidth*encoderWidth), v: make([]float32, encoderWidth*encoderWidth)}, relativeWeight: make([]float32, encoderWidth*encoderWidth), biasU: make([]float32, encoderWidth), biasV: make([]float32, encoderWidth), outputWeight: make([]float32, encoderWidth*encoderWidth)}
	for _, rows := range []int{0, 6} {
		if _, _, err := m.ForwardOffline(make([]float32, rows*encoderWidth), rows); err == nil {
			t.Fatalf("accepted rows=%d", rows)
		}
	}
	if _, _, err := m.ForwardOffline(make([]float32, encoderWidth-1), 1); err == nil {
		t.Fatal("accepted short input")
	}
	bad := make([]float32, encoderWidth)
	bad[0] = float32(math.NaN())
	if _, _, err := m.ForwardOffline(bad, 1); err == nil {
		t.Fatal("accepted non-finite input")
	}
	if _, _, err := m.ForwardOfflineLookahead(make([]float32, encoderWidth), 1, 2); err == nil {
		t.Fatal("accepted unsupported lookahead")
	}
}

func BenchmarkReleasedEncoder0Attention5(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	m, loadErr := LoadEncoder0Attention(file)
	if err := file.Close(); err != nil {
		b.Fatal(err)
	}
	if loadErr != nil {
		b.Fatal(loadErr)
	}
	input := readStemFixture(b, "encoder0_ff1_residual", 5*encoderWidth)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, _, err := m.ForwardOffline(input, 5); err != nil {
			b.Fatal(err)
		}
	}
}
