package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedRNNTProjectionPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, err := LoadRNNTProjection(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	input := readStemFixture(t, "encoder0_block_output", 5*encoderWidth)
	original := append([]float32(nil), input...)
	decoder := readStemFixture(t, "rnnt_joint_decoder_input", rnntHidden)
	var failures []string
	for _, prompt := range []int{101, 7} {
		first, fused, encoded, err := model.Project(input, 5, prompt)
		if err != nil {
			t.Fatal(err)
		}
		logits, err := model.Joint(encoded, decoder, 5)
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range []struct {
			name string
			got  []float32
		}{{"linear1", first}, {"output", fused}, {"encoder", encoded}, {"joint_logits", logits}} {
			ref := readStemFixture(t, fmt.Sprintf("rnnt_prompt%d_%s", prompt, item.name), len(item.got))
			// The encoder stage compounds prompt-fusion error and a second
			// reduction. Against pinned PyTorch, the exact F64 dot on the
			// fixture input has zero outliers at the tighter gate; the composed
			// scalar path needs a measured 8e-4 floor near zero. Other stages
			// retain the established 3e-4 floor.
			absTol, meanTol := 3e-4, 3e-5
			switch item.name {
			case "output":
				meanTol = 1e-4
			case "encoder":
				absTol, meanTol = 8e-4, 2e-4
			case "joint_logits":
				meanTol = 2e-2
			}
			var maxAbs, sumAbs float64
			var outside, worst, firstOutside int
			for i, value := range item.got {
				delta := math.Abs(float64(value - ref[i]))
				if delta > maxAbs {
					maxAbs = delta
					worst = i
				}
				sumAbs += delta
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > absTol+2e-5*math.Abs(float64(ref[i])) {
					if outside == 0 {
						firstOutside = i
					}
					outside++
				}
			}
			mean := sumAbs / float64(len(item.got))
			t.Logf("prompt=%d stage=%s max_abs=%g mean_abs=%g outside=%d", prompt, item.name, maxAbs, mean, outside)
			if outside != 0 || mean > meanTol {
				failures = append(failures, fmt.Sprintf("prompt=%d stage=%s outside=%d mean=%g first=%d got=%g ref=%g worst=%d got=%g ref=%g", prompt, item.name, outside, mean, firstOutside, item.got[firstOutside], ref[firstOutside], worst, item.got[worst], ref[worst]))
			}
		}
	}
	if len(failures) != 0 {
		t.Fatalf("PyTorch parity: %v", failures)
	}
	for i, value := range input {
		if value != original[i] {
			t.Fatalf("mutated input %d", i)
		}
	}
	first, fused, encoded, err := model.Project(input, 5, 101)
	if err != nil {
		t.Fatal(err)
	}
	if &first[0] == &fused[0] || &first[0] == &encoded[0] || &fused[0] == &encoded[0] || &first[0] == &input[0] {
		t.Fatal("projection outputs alias")
	}
	joint, err := model.Joint(encoded, decoder, 5)
	if err != nil {
		t.Fatal(err)
	}
	if &joint[0] == &encoded[0] || &joint[0] == &decoder[0] {
		t.Fatal("joint output aliases input")
	}
	before := append([]float32(nil), joint...)
	if _, err := model.Joint(encoded, decoder, 5); err != nil {
		t.Fatal(err)
	}
	for i := range before {
		if joint[i] != before[i] {
			t.Fatal("joint output changed after another call")
		}
	}
}

func TestRNNTProjectionRejectsMalformed(t *testing.T) {
	if _, err := LoadRNNTProjection(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	if _, _, _, err := (*RNNTProjection)(nil).Project(make([]float32, encoderWidth), 1, 101); err == nil {
		t.Fatal("accepted nil projector")
	}
	if _, err := (*RNNTProjection)(nil).Joint(make([]float32, rnntHidden), make([]float32, rnntHidden), 1); err == nil {
		t.Fatal("accepted nil joint")
	}
	m := &RNNTProjection{promptFirst: make([]float32, rnntIntermediate*(encoderWidth+rnntPrompts)), promptFirstBias: make([]float32, rnntIntermediate), promptSecond: make([]float32, encoderWidth*rnntIntermediate), promptSecondBias: make([]float32, encoderWidth), encoderWeight: make([]float32, rnntHidden*encoderWidth), encoderBias: make([]float32, rnntHidden), jointWeight: make([]float32, rnntVocabulary*rnntHidden), jointBias: make([]float32, rnntVocabulary)}
	if _, _, _, err := m.Project(make([]float32, encoderWidth), 1, rnntPrompts); err == nil {
		t.Fatal("accepted out-of-range prompt")
	}
	if _, _, _, err := m.Project(make([]float32, encoderWidth-1), 1, 0); err == nil {
		t.Fatal("accepted short input")
	}
	bad := make([]float32, encoderWidth)
	bad[0] = float32(math.NaN())
	if _, _, _, err := m.Project(bad, 1, 0); err == nil {
		t.Fatal("accepted non-finite input")
	}
	if _, err := m.Joint(make([]float32, rnntHidden), make([]float32, rnntHidden-1), 1); err == nil {
		t.Fatal("accepted short decoder")
	}
	if _, err := m.Joint(nil, nil, -1); err == nil {
		t.Fatal("accepted negative rows")
	}
	if _, err := m.Joint(nil, nil, 6); err == nil {
		t.Fatal("accepted unqualified rows")
	}
	if err := m.jointTo(make([]float32, rnntVocabulary-1), make([]float32, rnntHidden), make([]float32, rnntHidden), make([]float32, rnntHidden), 1); err == nil {
		t.Fatal("accepted undersized joint scratch")
	}
}
