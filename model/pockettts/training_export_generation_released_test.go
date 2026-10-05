package pockettts

import (
	"math"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Adds a two-frame recurrent generation/reset check without repeating the
// completed single-frame tensor inventory or affine parity gate.
func TestPocketSavedEMATwoFrameGeneration(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKETTTS_SAVED_EMA_INFERENCE") != "1" {
		t.Skip("set GO_PHERENCE_POCKETTTS_SAVED_EMA_INFERENCE=1 and pinned asset paths")
	}
	exportPath := releasedAsset(t, "GO_PHERENCE_POCKETTTS_EMA_EXPORT", "3681bf13d0b6232c83731ee5806c3cb4b341bd89a2696a3e6343251ff528d12d", 438033312)
	checkpointPath := releasedAsset(t, "GO_PHERENCE_POCKETTTS_EMA_CHECKPOINT", "16391d753802996f7fded00af5f5c8391d699c69c48aa2563cc77d3659de7413", 1431217112)
	testSavedEMATwoFrameGeneration(t, exportPath, checkpointPath)
}

func TestPocketNativeEMATwoFrameGeneration(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKETTTS_NATIVE_EMA_INFERENCE") != "1" {
		t.Skip("explicit native EMA recurrent generation qualification")
	}
	exportPath := nativeEMAExportPath(t)
	checkpointPath := releasedAsset(t, "GO_PHERENCE_POCKETTTS_EMA_CHECKPOINT", "16391d753802996f7fded00af5f5c8391d699c69c48aa2563cc77d3659de7413", 1431217112)
	testSavedEMATwoFrameGeneration(t, exportPath, checkpointPath)
}

func testSavedEMATwoFrameGeneration(t *testing.T, exportPath, checkpointPath string) {
	t.Helper()
	frozenPath := releasedAsset(t, "GO_PHERENCE_POCKETTTS_VOICE_MODEL", "fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997", 219029196)
	cfg := releasedConfig(t)
	state, err := LoadFullTrainingStateBinary(checkpointPath, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	if state.Step != 2 {
		t.Fatal("checkpoint boundary")
	}
	values := map[string][]float32{}
	for _, tensor := range state.Params {
		if strings.HasPrefix(tensor.Name, "flow_lm.") {
			values[tensor.Name] = tensor.Values
		}
	}
	for _, tensor := range state.EMA {
		if strings.HasPrefix(tensor.Name, "flow_lm.") {
			values[tensor.Name] = tensor.Values
		}
	}
	for _, tensor := range state.Buffers {
		values[tensor.Name] = tensor.Values
	}
	state = FullTrainingState{}
	runtime.GC()
	lm, flow := savedEMAInferenceModels(t, cfg, values)
	frozen, err := safetensors.Open(frozenPath)
	if err != nil {
		t.Fatal(err)
	}
	defer frozen.Close()
	mimi, err := LoadMimiDecoderCPU(frozen, cfg)
	if err != nil {
		t.Fatal(err)
	}
	generator, err := LoadGeneratorCPU(exportPath, cfg)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []uint32{1, 7, 23}
	noise := func(frame int, dst []float32) error {
		for i := range dst {
			dst[i] = float32(math.Sin(float64(i+1)*.37+float64(frame)*.11) * .3)
		}
		return nil
	}
	// Independent orchestration: re-evaluate the full causal sequence for each
	// frame instead of using generation's incremental KV state.
	h, c := cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim
	sequence := make([]float32, len(tokens)*h)
	for row, id := range tokens {
		copy(sequence[row*h:(row+1)*h], lm.Embedding[int(id)*h:(int(id)+1)*h])
	}
	latent := append([]float32(nil), lm.BOS...)
	want := make([]float32, 0, 2*SamplesPerFrame)
	decoderState := mimi.NewStateForFrames(2)
	for frame := 0; frame < 2; frame++ {
		row := make([]float32, h)
		if err := lm.Input.Forward(row, latent); err != nil {
			t.Fatal(err)
		}
		sequence = append(sequence, row...)
		hidden, err := lm.Transformer.Forward(sequence, len(sequence)/h)
		if err != nil {
			t.Fatal(err)
		}
		last := hidden[len(hidden)-h:]
		if err := noise(frame, latent); err != nil {
			t.Fatal(err)
		}
		for step := 0; step < 2; step++ {
			velocity := make([]float32, c)
			if err := flow.Forward(velocity, last, []float32{float32(step) / 2, float32(step+1) / 2}, latent); err != nil {
				t.Fatal(err)
			}
			for i := range latent {
				latent[i] += .5 * velocity[i]
			}
		}
		raw := make([]float32, c)
		for i := range raw {
			raw[i] = latent[i]*values["flow_lm.emb_std"][i] + values["flow_lm.emb_mean"][i]
		}
		pcm, err := mimi.Decode(raw, 1, decoderState)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, pcm...)
	}
	voice, err := generator.FlowLM.Transformer.NewState(len(tokens) + 2)
	if err != nil {
		t.Fatal(err)
	}
	got, err := generator.Generate(voice, tokens, 2, 3, 2, float32(math.Inf(1)), noise)
	if err != nil {
		t.Fatal(err)
	}
	savedEMAErrorDistribution(t, "two_frame_generator_pcm", got, want, 3e-5)
	template, err := generator.FlowLM.Transformer.NewState(len(tokens) + 2)
	if err != nil {
		t.Fatal(err)
	}
	session, err := NewSession(generator, template, 2)
	if err != nil {
		t.Fatal(err)
	}
	for request := 0; request < 2; request++ {
		pcm := make([]float32, 2*SamplesPerFrame)
		n, err := session.GenerateInto(pcm, tokens, 2, 3, 2, float32(math.Inf(1)), noise)
		if err != nil {
			t.Fatal(err)
		}
		if n != len(pcm) || session.FlowState.Position != len(tokens)+2 {
			t.Fatal("generation/reset extent", n, session.FlowState.Position)
		}
		savedEMAErrorDistribution(t, "two_frame_session_pcm", pcm, want, 3e-5)
	}
	if template.Position != 0 {
		t.Fatal("session mutated voice template")
	}
}
