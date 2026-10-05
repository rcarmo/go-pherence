package pockettts

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// This opt-in engineering gate checks the saved synthetic-row step-two EMA
// export. It never updates, exports or publishes state. The reference models
// are assembled directly from checkpoint tensors, bypassing inference loaders
// and export selection. Inference arithmetic is shared, so this does not
// replace the independent PyTorch or native full-step numerical gates.
func TestPocketSavedEMAExportInference(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKETTTS_SAVED_EMA_INFERENCE") != "1" {
		t.Skip("set GO_PHERENCE_POCKETTTS_SAVED_EMA_INFERENCE=1 and pinned asset paths")
	}
	exportPath := releasedAsset(t, "GO_PHERENCE_POCKETTTS_EMA_EXPORT", "3681bf13d0b6232c83731ee5806c3cb4b341bd89a2696a3e6343251ff528d12d", 438033312)
	checkpointPath := releasedAsset(t, "GO_PHERENCE_POCKETTTS_EMA_CHECKPOINT", "16391d753802996f7fded00af5f5c8391d699c69c48aa2563cc77d3659de7413", 1431217112)
	testSavedEMAExportInference(t, exportPath, checkpointPath, 0)
}

// Native exports are compared numerically to the independently produced CPU
// checkpoint. File hashes are not an acceptance gate for generated output.
func TestPocketNativeEMAExportInference(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKETTTS_NATIVE_EMA_INFERENCE") != "1" {
		t.Skip("explicit native EMA export inference qualification")
	}
	exportPath := nativeEMAExportPath(t)
	checkpointPath := releasedAsset(t, "GO_PHERENCE_POCKETTTS_EMA_CHECKPOINT", "16391d753802996f7fded00af5f5c8391d699c69c48aa2563cc77d3659de7413", 1431217112)
	testSavedEMAExportInference(t, exportPath, checkpointPath, 4e-6)
}

func nativeEMAExportPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("GO_PHERENCE_POCKETTTS_EMA_EXPORT")
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() != 438033312 {
		t.Fatal("native EMA export format/extent", err)
	}
	return path
}

func testSavedEMAExportInference(t *testing.T, exportPath, checkpointPath string, tensorTolerance float64) {
	t.Helper()
	frozenPath := releasedAsset(t, "GO_PHERENCE_POCKETTTS_VOICE_MODEL", "fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997", 219029196)
	cfg := releasedConfig(t)
	state, err := LoadFullTrainingStateBinary(checkpointPath, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	if state.Step != 2 || state.EMADecay <= 0 || len(state.EMA) == 0 {
		t.Fatal("unexpected engineering checkpoint boundary")
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
	// Release unrelated P/M/V at this diagnostic-only phase boundary.
	runtime.GC()
	referenceLM, referenceFlow := savedEMAInferenceModels(t, cfg, values)
	src, err := safetensors.Open(exportPath)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	frozen, err := safetensors.Open(frozenPath)
	if err != nil {
		t.Fatal(err)
	}
	defer frozen.Close()
	var elements int64
	for _, name := range src.Names() {
		got, _, err := src.GetFloat32(name)
		if err != nil {
			t.Fatal(err)
		}
		want, ok := values[name]
		if strings.HasPrefix(name, "mimi.") {
			want, _, err = frozen.GetFloat32(name)
			if err != nil {
				t.Fatal(err)
			}
			ok = true
		}
		if !ok || len(got) != len(want) {
			t.Fatal("unexpected export tensor", name)
		}
		// CPU export retains its exact persisted tensor contract. Native output
		// uses the unchanged independent parameter gate, never output hashes.
		for i := range got {
			if !isFinite(got[i]) || math.Abs(float64(got[i])-float64(want[i])) > tensorTolerance {
				t.Fatalf("export differs from checkpoint/frozen source %s[%d]", name, i)
			}
		}
		elements += int64(len(got))
	}
	if len(src.Names()) != len(expectedFlowExportShapes(cfg))+len(expectedMimiStateShapes(cfg)) {
		t.Fatal("export inventory mismatch")
	}
	t.Logf("stored tensor values checked=%d tensors=%d tolerance=%g", elements, len(src.Names()), tensorTolerance)
	loadedLM, err := LoadFlowLMCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	loadedFlow, err := LoadFlowHeadCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Text IDs and noise are deterministic synthetic inputs, not voice-quality
	// fixtures. No imported pre-update voice cache is used for changed weights.
	tokens := []uint32{1, 7, 23}
	noise := make([]float32, cfg.Mimi.InnerDim)
	for i := range noise {
		noise[i] = float32(math.Sin(float64(i+1)*.37) * .3)
	}
	savedEMAAffineReference(t, "lm_input", loadedLM.Input, referenceLM.BOS)
	wantHidden, wantEOS, err := referenceLM.Prefill(tokens, referenceLM.BOS)
	if err != nil {
		t.Fatal(err)
	}
	gotHidden, gotEOS, loadedState, err := loadedLM.PrefillStreaming(tokens, loadedLM.BOS)
	if err != nil {
		t.Fatal(err)
	}
	// Existing released hidden/EOS streaming gate, unchanged.
	savedEMAErrorDistribution(t, "prefill_hidden", gotHidden, wantHidden, 3e-5)
	savedEMAErrorDistribution(t, "prefill_eos", []float32{gotEOS}, []float32{wantEOS}, 3e-5)
	savedEMAAffineReference(t, "eos", loadedLM.EOS, gotHidden)
	savedEMAAffineReference(t, "flow_input", loadedFlow.Input, noise)
	gotVelocity, wantVelocity := make([]float32, len(noise)), make([]float32, len(noise))
	if err := loadedFlow.Forward(gotVelocity, gotHidden, []float32{0, .5}, noise); err != nil {
		t.Fatal(err)
	}
	if err := referenceFlow.Forward(wantVelocity, wantHidden, []float32{0, .5}, noise); err != nil {
		t.Fatal(err)
	}
	// Existing synthetic flow-head gate, unchanged.
	savedEMAErrorDistribution(t, "flow_velocity", gotVelocity, wantVelocity, 2e-5)
	gotLatent, wantLatent := make([]float32, len(noise)), make([]float32, len(noise))
	for _, boundary := range []struct {
		dst, hidden []float32
		flow        *FlowHeadCPU
	}{{gotLatent, gotHidden, loadedFlow}, {wantLatent, wantHidden, referenceFlow}} {
		if err := LSDDecodeSIMD(boundary.dst, noise, 2, make([]float32, len(noise)), func(dst []float32, start, end float32, input []float32) error {
			return boundary.flow.Forward(dst, boundary.hidden, []float32{start, end}, input)
		}); err != nil {
			t.Fatal(err)
		}
	}
	savedEMAErrorDistribution(t, "lsd_latent", gotLatent, wantLatent, 3e-5)
	if loadedState.Position != len(tokens)+1 {
		t.Fatal("streaming state position")
	}
	// Decode each independently obtained latent using separately loaded frozen
	// Mimi models. This measures propagation through the common decoder; audio
	// listening and an independent decoder oracle remain separate gates.
	loadedMimi, err := LoadMimiDecoderCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	referenceMimi, err := LoadMimiDecoderCPU(frozen, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, latent := range [][]float32{gotLatent, wantLatent} {
		for i := range latent {
			latent[i] = latent[i]*values["flow_lm.emb_std"][i] + values["flow_lm.emb_mean"][i]
		}
	}
	gotPCM, err := loadedMimi.Decode(gotLatent, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantPCM, err := referenceMimi.Decode(wantLatent, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotPCM) != SamplesPerFrame {
		t.Fatal("decoded frame length", len(gotPCM))
	}
	savedEMAErrorDistribution(t, "decoded_pcm", gotPCM, wantPCM, 3e-5)
}

func savedEMAInferenceModels(t *testing.T, cfg Config, values map[string][]float32) (*FlowLMCPU, *FlowHeadCPU) {
	t.Helper()
	get := func(name string, size int) []float32 {
		v, ok := values[name]
		if !ok || len(v) != size || !finiteF32(v) {
			t.Fatal("invalid checkpoint inference tensor", name)
		}
		return v
	}
	linear := func(prefix string, in, out int, bias bool) LinearF32 {
		l := LinearF32{In: in, Out: out, Weight: get(prefix+".weight", in*out)}
		if bias {
			l.Bias = get(prefix+".bias", out)
		}
		return l
	}
	h, c, f := cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim, cfg.FlowLM.Flow.Dim
	tr := cfg.FlowLM.Transformer
	ff := h * tr.HiddenScale
	transformer := &TransformerCPU{Width: h, Heads: tr.NumHeads, HeadDim: h / tr.NumHeads, Context: tr.Context, MaxPeriod: tr.MaxPeriod, Layers: make([]TransformerLayerCPU, tr.NumLayers), FinalWeight: get("flow_lm.out_norm.weight", h), FinalBias: get("flow_lm.out_norm.bias", h)}
	for i := range transformer.Layers {
		p := fmt.Sprintf("flow_lm.transformer.layers.%d", i)
		transformer.Layers[i] = TransformerLayerCPU{Norm1Weight: get(p+".norm1.weight", h), Norm1Bias: get(p+".norm1.bias", h), Norm2Weight: get(p+".norm2.weight", h), Norm2Bias: get(p+".norm2.bias", h), InProjection: linear(p+".self_attn.in_proj", h, 3*h, false), OutProjection: linear(p+".self_attn.out_proj", h, h, false), FC1: linear(p+".linear1", h, ff, false), FC2: linear(p+".linear2", ff, h, false)}
	}
	lm := &FlowLMCPU{Config: cfg, Embedding: get("flow_lm.conditioner.embed.weight", (cfg.FlowLM.LookupTable.NBins+1)*h), BOS: get("flow_lm.bos_emb", c), Input: linear("flow_lm.input_linear", c, h, false), Transformer: transformer, EOS: linear("flow_lm.out_eos", h, 1, true)}
	flow := &FlowHeadCPU{Input: linear("flow_lm.flow_net.input_proj", c, f, true), Condition: linear("flow_lm.flow_net.cond_embed", h, f, true), Time: make([]TimestepMLP, 2), Blocks: make([]AdaLNResidual, cfg.FlowLM.Flow.Depth)}
	for i := range flow.Time {
		p := fmt.Sprintf("flow_lm.flow_net.time_embed.%d", i)
		flow.Time[i] = TimestepMLP{Frequencies: get(p+".freqs", 128), FC1: linear(p+".mlp.0", 256, f, true), FC2: linear(p+".mlp.2", f, f, true), RMSWeight: get(p+".mlp.3.alpha", f), RMSEpsilon: 1e-5}
	}
	for i := range flow.Blocks {
		p := fmt.Sprintf("flow_lm.flow_net.res_blocks.%d", i)
		flow.Blocks[i] = AdaLNResidual{NormWeight: get(p+".in_ln.weight", f), NormBias: get(p+".in_ln.bias", f), FC1: linear(p+".mlp.0", f, f, true), FC2: linear(p+".mlp.2", f, f, true), Modulation: linear(p+".adaLN_modulation.1", f, 3*f, true), Epsilon: 1e-6}
	}
	flow.Final = AdaLNFinal{Linear: linear("flow_lm.flow_net.final_layer.linear", f, c, true), Modulation: linear("flow_lm.flow_net.final_layer.adaLN_modulation.1", f, 2*f, true), Epsilon: 1e-6}
	return lm, flow
}

func savedEMAAffineReference(t *testing.T, name string, linear LinearF32, input []float32) {
	t.Helper()
	if len(linear.WeightBF16) != 0 || len(linear.Weight) != linear.In*linear.Out {
		t.Fatal(name, "F32 tensor was quantised")
	}
	got := make([]float32, linear.Out)
	if err := linear.Forward(got, input); err != nil {
		t.Fatal(err)
	}
	maxError, maxBoundRatio := float64(0), float64(0)
	// A priori F32 rounding bound: gamma_(n+2) * sum(abs(products)+bias).
	// No measured error is used to adjust the bound after the run.
	u := math.Ldexp(1, -24)
	gamma := float64(linear.In+2) * u / (1 - float64(linear.In+2)*u)
	for row := range got {
		want, magnitude := float64(0), float64(0)
		for col, x := range input {
			product := float64(x) * float64(linear.Weight[row*linear.In+col])
			want += product
			magnitude += math.Abs(product)
		}
		if linear.Bias != nil {
			want += float64(linear.Bias[row])
			magnitude += math.Abs(float64(linear.Bias[row]))
		}
		diff := math.Abs(float64(got[row]) - want)
		bound := gamma*magnitude + float64(linear.In+2)*math.Ldexp(1, -149)
		if !isFinite(got[row]) || diff > bound {
			t.Fatalf("%s[%d] F64 reference error=%g bound=%g", name, row, diff, bound)
		}
		maxError = max(maxError, diff)
		if bound > 0 {
			maxBoundRatio = max(maxBoundRatio, diff/bound)
		}
	}
	t.Logf("%s F64 affine rows=%d max_abs=%g max_rounding_bound_ratio=%g", name, len(got), maxError, maxBoundRatio)
}

func savedEMAErrorDistribution(t *testing.T, name string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) == 0 || len(got) != len(want) {
		t.Fatal(name, "invalid output extent")
	}
	maximum, sum, square := float64(0), float64(0), float64(0)
	for i := range got {
		if !isFinite(got[i]) || !isFinite(want[i]) {
			t.Fatal(name, "non-finite output", i)
		}
		d := math.Abs(float64(got[i]) - float64(want[i]))
		maximum = max(maximum, d)
		sum += d
		square += d * d
	}
	t.Logf("%s n=%d max_abs=%g mean_abs=%g rms=%g gate=%g", name, len(got), maximum, sum/float64(len(got)), math.Sqrt(square/float64(len(got))), tolerance)
	if maximum > tolerance {
		t.Fatal(name, "unchanged numerical gate exceeded")
	}
}
