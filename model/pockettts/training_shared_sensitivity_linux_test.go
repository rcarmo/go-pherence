//go:build linux

package pockettts

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"testing"
)

type sharedSensitivityPoint struct {
	Name                                                        string `json:"name"`
	Index                                                       int    `json:"index"`
	CPUParameter, VulkanParameter, CPUM, CPUV, VulkanM, VulkanV float32
}

type sharedSensitivityValue struct {
	Name                               string `json:"name"`
	Index                              int    `json:"index"`
	Gradient, M, V, PredictedParameter float32
}

// Test-only local intervention: replace just the speaker inputs to the CPU
// transformer with the precise sequential-F32 projection. All other forward/
// backward arithmetic remains CPU. This estimates sensitivity; it neither
// emulates a full Vulkan graph nor supplies an execution fallback.
type sharedSpeakerSensitivityBackend struct {
	batch         FlowLMTrainingBatch
	projection    LinearF32
	modified      int
	maxDifference float64
}

func (b *sharedSpeakerSensitivityBackend) EvaluateTransformer(m *TransformerCPU, sequence, seed []float32, rows int) (TransformerTrainingResult, error) {
	if b == nil || m == nil || b.batch.VoiceFrames <= 0 || rows != 1+b.batch.VoiceFrames+len(b.batch.TextTokens)+b.batch.Frames || len(sequence) != rows*m.Width || b.projection.In != len(b.batch.VoiceLatents)/b.batch.VoiceFrames || b.projection.Out != m.Width {
		return TransformerTrainingResult{}, fmt.Errorf("sensitivity boundary shape")
	}
	x := append([]float32(nil), sequence...)
	for row := 0; row < b.batch.VoiceFrames; row++ {
		for col := 0; col < m.Width; col++ {
			var sum float32
			for q := 0; q < b.projection.In; q++ {
				sum += float32(b.batch.VoiceLatents[row*b.projection.In+q] * b.projection.Weight[col*b.projection.In+q])
			}
			index := (1+row)*m.Width + col
			d := math.Abs(float64(sum) - float64(x[index]))
			if d > 0 {
				b.modified++
			}
			b.maxDifference = math.Max(b.maxDifference, d)
			x[index] = sum
		}
	}
	output, g, dx, err := m.ForwardBackward(x, seed, rows)
	if err != nil {
		return TransformerTrainingResult{}, err
	}
	all := map[string][]float32{}
	addFlowLMGradients(all, &FlowLMTrainingGradients{Transformer: g})
	for name := range all {
		if !strings.HasPrefix(name, "flow_lm.transformer.layers.") && !strings.HasPrefix(name, "flow_lm.out_norm.") {
			delete(all, name)
		}
	}
	return TransformerTrainingResult{Output: output, DSequence: dx, Parameters: all}, nil
}

// A test-only CPU accumulator isolates the resident branch contract: each
// branch computes local unscaled parameter gradients, then applies the objective
// scale at accumulation. Existing CPU default often scales seeds before VJP.
// Only CPU routines execute; no graph interpreter or shader fallback is used.
type sharedAccumulationSensitivityBackend struct {
	gradients *FlowHeadGradients
	calls     int
}

func (b *sharedAccumulationSensitivityBackend) EvaluateFlow(*FlowHeadCPU, []float32, []float32, []float32, int, []float32, []float32) (FlowDualTrainingResult, error) {
	return FlowDualTrainingResult{}, fmt.Errorf("explicit sensitivity accumulation required")
}
func (b *sharedAccumulationSensitivityBackend) EvaluateFlowAccumulating(m *FlowHeadCPU, condition, times, input []float32, index int, gv, gt []float32, accumulate bool, scale float32) (FlowDualTrainingResult, error) {
	if b == nil || !isFinite(scale) || scale < 0 || index < -1 || (index == -1 && !zeroTrainingSeed(gt)) {
		return FlowDualTrainingResult{}, fmt.Errorf("sensitivity scale/index/tangent seed")
	}
	var r FlowDualTrainingResult
	var g *FlowHeadGradients
	var err error
	if index < 0 {
		r.Output, g, r.DCondition, r.DTimes, r.DInput, err = m.ForwardBackward(condition, times, input, gv)
		r.Tangent = make([]float32, len(r.Output))
	} else {
		r.Output, r.Tangent, g, r.DCondition, r.DTimes, r.DInput, err = m.ForwardTimeJVPBackward(condition, times, input, index, gv, gt)
	}
	if err != nil {
		return FlowDualTrainingResult{}, err
	}
	if accumulate {
		if b.gradients == nil {
			b.gradients = newFlowHeadGradients(m)
		}
		accumulateFlowTrainingGradients(b.gradients, g, scale)
		b.calls++
	}
	return r, nil
}

func TestPocketSharedAccumulationSensitivityBoundary(t *testing.T) {
	m := tinyTrainableFlowHead()
	b := &sharedAccumulationSensitivityBackend{}
	r, err := b.EvaluateFlowAccumulating(m, []float32{.2, -.3, .5}, []float32{.3, .8}, []float32{.3, -.2}, 1, []float32{.1, .2}, []float32{.3, -.4}, true, .25)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateFlowBackendResult(m, r); err != nil {
		t.Fatal(err)
	}
	_, _, want, _, _, _, err := m.ForwardTimeJVPBackward([]float32{.2, -.3, .5}, []float32{.3, .8}, []float32{.3, -.2}, 1, []float32{.1, .2}, []float32{.3, -.4})
	if err != nil {
		t.Fatal(err)
	}
	a, c := map[string][]float32{}, map[string][]float32{}
	addFlowHeadGradientParameters(a, b.gradients)
	addFlowHeadGradientParameters(c, want)
	for name, x := range a {
		for i, v := range x {
			if v != c[name][i]*.25 {
				t.Fatal("local branch scaling", name, i)
			}
		}
	}
	before := b.calls
	if _, err := b.EvaluateFlow(m, nil, nil, nil, -1, nil, nil); err == nil {
		t.Fatal("implicit accumulation")
	}
	if _, err := b.EvaluateFlowAccumulating(m, nil, nil, nil, -1, nil, []float32{1}, true, 1); err == nil {
		t.Fatal("ordinary nonzero tangent seed")
	}
	if _, err := b.EvaluateFlowAccumulating(m, nil, nil, nil, -2, nil, nil, true, 1); err == nil {
		t.Fatal("unsupported index")
	}
	if _, err := b.EvaluateFlowAccumulating(m, nil, nil, nil, -1, nil, nil, true, -1); err == nil {
		t.Fatal("negative scale")
	}
	if b.calls != before {
		t.Fatal("invalid sensitivity mutated accumulator")
	}
}

type sharedReductionObservation struct {
	Name                           string `json:"name"`
	Index                          int    `json:"index"`
	CPUG, SequentialG, ShaderGeluG float32
	FusedG                         float32
	ReferenceG                     float64
	Inputs, Seeds                  []float32 `json:",omitempty"`
}

// Keep CPU forward and backward seeds fixed. Observe only the FC2 weight
// gradient dot product at selected indices; no observed value is used by AD.
type sharedReductionSensitivityBackend struct {
	points         []sharedSensitivityPoint
	observations   []sharedReductionObservation
	capture        bool
	captureLayer   int
	sequence, seed []float32
}

// Emulate specified F32 polynomial/exp formula at fixed CPU preactivations.
// Host exp rounding cannot qualify the actual device transcendental function.
func sharedShaderGeluDiagnostic(x float32) float32 {
	u := float32(.7978845608028654) * (x + ((float32(.044715)*x)*x)*x)
	e := float32(math.Exp(float64(-2 * float32(math.Abs(float64(u))))))
	half := float32(1) / (1 + e)
	if u < 0 {
		half = e / (1 + e)
	}
	return x * half
}

func (b *sharedReductionSensitivityBackend) EvaluateTransformer(m *TransformerCPU, sequence, seed []float32, rows int) (TransformerTrainingResult, error) {
	if b == nil {
		return TransformerTrainingResult{}, fmt.Errorf("nil reduction observer")
	}
	if err := validateTrainableTransformer(m, sequence, seed, rows); err != nil {
		return TransformerTrainingResult{}, err
	}
	tape, output := m.forwardTraining(rows, sequence)
	g := newTransformerGradients(m)
	dh := seed
	if m.FinalWeight != nil {
		dh = transformerNormBackward(rows, m.Width, tape.final, dh, m.FinalWeight, g.FinalWeight, g.FinalBias)
	}
	var observation []sharedReductionObservation
	for layer := len(m.Layers) - 1; layer >= 0; layer-- {
		l, lt := m.Layers[layer], tape.layers[layer]
		if b.capture && layer == b.captureLayer && !zeroTrainingSeed(seed) {
			b.sequence = append([]float32(nil), lt.norm1.input...)
			b.seed = append([]float32(nil), dh...)
		}
		name := fmt.Sprintf("flow_lm.transformer.layers.%d.linear2.weight", layer)
		for _, p := range b.points {
			if p.Name != name {
				continue
			}
			if p.Index < 0 || p.Index >= l.FC2.In*l.FC2.Out {
				return TransformerTrainingResult{}, fmt.Errorf("observer index")
			}
			out, in := p.Index/l.FC2.In, p.Index%l.FC2.In
			var sequential, shaderGelu, fused float32
			inputs, seeds := make([]float32, rows), make([]float32, rows)
			var reference, correction float64
			for row := 0; row < rows; row++ {
				d := dh[row*m.Width+out]
				if l.LayerScale2 != nil {
					d *= l.LayerScale2[out]
				}
				x := lt.fc1[row*l.FC2.In+in]
				inputs[row], seeds[row] = x, d
				fused = float32(math.FMA(float64(d), float64(x), float64(fused)))
				sequential += float32(d * x)
				shaderGelu += float32(d * sharedShaderGeluDiagnostic(lt.fc1Pre[row*l.FC2.In+in]))
				term := float64(d)*float64(x) - correction
				next := reference + term
				correction = (next - reference) - term
				reference = next
			}
			observation = append(observation, sharedReductionObservation{Name: name, Index: p.Index, SequentialG: sequential, ShaderGeluG: shaderGelu, FusedG: fused, ReferenceG: reference, Inputs: inputs, Seeds: seeds})
		}
		var err error
		dh, err = l.backwardTrainingWithAttention(rows, m.Width, m.Heads, m.HeadDim, m.Context, m.MaxPeriod, lt, dh, &g.Layers[layer], nil)
		if err != nil {
			return TransformerTrainingResult{}, err
		}
		for i := range observation {
			if observation[i].Name == name {
				observation[i].CPUG = g.Layers[layer].FC2.Weight[observation[i].Index]
			}
		}
	}
	// Zero-seed forward calls need no observations; the final seeded call owns
	// all diagnostic points while Parameters carry only normal CPU gradients.
	if !zeroTrainingSeed(seed) {
		b.observations = observation
	}
	all := map[string][]float32{}
	addFlowLMGradients(all, &FlowLMTrainingGradients{Transformer: g})
	for name := range all {
		if !strings.HasPrefix(name, "flow_lm.transformer.layers.") && !strings.HasPrefix(name, "flow_lm.out_norm.") {
			delete(all, name)
		}
	}
	return TransformerTrainingResult{Output: output, DSequence: dh, Parameters: all}, nil
}

func TestPocketSharedGeluSensitivityFormula(t *testing.T) {
	for _, x := range []float32{-12, -8, -5, -2, -.1, 0, .1, 2, 5, 8, 12} {
		got := sharedShaderGeluDiagnostic(x)
		if !isFinite(got) {
			t.Fatal("nonfinite formula")
		}
		if math.Abs(float64(got)-float64(geluTanh(x))) > 1e-6 {
			t.Fatal("formula ordinary range", x, got, geluTanh(x))
		}
	}
}

func TestPocketSharedReductionSensitivityBoundary(t *testing.T) {
	lm := tinyFlowLMTraining()
	rows := 4
	sequence, seed := make([]float32, rows*lm.Hidden), make([]float32, rows*lm.Hidden)
	for i := range sequence {
		sequence[i] = float32(i%7-3) * .1
		seed[i] = float32(i%3-1) * .03
	}
	b := &sharedReductionSensitivityBackend{points: []sharedSensitivityPoint{{Name: "flow_lm.transformer.layers.0.linear2.weight", Index: 1}}}
	r, err := b.EvaluateTransformer(lm.Transformer, sequence, seed, rows)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateTransformerBackendResult(lm.Transformer, r, sequence, seed, rows); err != nil {
		t.Fatal(err)
	}
	if len(b.observations) != 1 || !isFinite(b.observations[0].CPUG) || !finite64(b.observations[0].ReferenceG) {
		t.Fatal("reduction observations")
	}
	b.points[0].Index = -1
	if _, err := b.EvaluateTransformer(lm.Transformer, sequence, seed, rows); err == nil {
		t.Fatal("bad observation index")
	}
}

func sharedSensitivityObservations(t *testing.T, tr *FullTrainer, g TrainingStepGradients, points []sharedSensitivityPoint) ([]sharedSensitivityValue, float64) {
	t.Helper()
	var square, correction float64
	for _, b := range tr.bindings {
		values := b.gradient(g)
		if len(values) != len(b.parameter) || !finiteF32(values) {
			t.Fatal("sensitivity gradient shape/nonfinite", b.name)
		}
		for _, v := range values {
			term := float64(v)*float64(v) - correction
			next := square + term
			correction = (next - square) - term
			square = next
		}
	}
	norm := math.Sqrt(square)
	if !finite64(norm) || norm <= 0 {
		t.Fatal("sensitivity norm")
	}
	scale := float32(1)
	if norm > 1 {
		scale = float32(1 / norm)
	}
	// Observation only after all gradients exist; expected saved values never
	// enter forward/backward, clipping reduction or any optimizer mutation.
	named, err := fullGradientMap(g)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]sharedSensitivityValue, len(points))
	c := tr.Config
	for i, p := range points {
		values, ok := named[p.Name]
		initial, found := tr.params[p.Name]
		if !ok || !found || p.Index < 0 || p.Index >= len(values) || len(values) != len(initial) {
			t.Fatal("sensitivity observation extent")
		}
		x := values[p.Index] * scale
		m, v := (1-c.Beta1)*x, ((1-c.Beta2)*x)*x
		predicted := initial[p.Index] * (1 - c.LearningRate*c.WeightDecay)
		predicted -= c.LearningRate * (m / (1 - c.Beta1)) / (float32(math.Sqrt(float64(v/(1-c.Beta2)))) + c.Epsilon)
		out[i] = sharedSensitivityValue{Name: p.Name, Index: p.Index, Gradient: x, M: m, V: v, PredictedParameter: predicted}
	}
	return out, norm
}

func TestPocketSharedSpeakerSensitivityBoundary(t *testing.T) {
	lm := tinyFlowLMTraining()
	batch := FlowLMTrainingBatch{Frames: 1, VoiceFrames: 1, TextTokens: []uint32{1}, VoiceLatents: []float32{.25, -.5}}
	backend := &sharedSpeakerSensitivityBackend{batch: batch, projection: lm.SpeakerProjection}
	rows := 4
	sequence := make([]float32, rows*lm.Hidden)
	seed := make([]float32, len(sequence))
	for i := range seed {
		seed[i] = float32(i%3-1) * .01
	}
	before := append([]float32(nil), sequence...)
	got, err := backend.EvaluateTransformer(lm.Transformer, sequence, seed, rows)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateTransformerBackendResult(lm.Transformer, got, sequence, seed, rows); err != nil {
		t.Fatal(err)
	}
	for i, v := range sequence {
		if v != before[i] {
			t.Fatal("sensitivity intervention mutated caller input")
		}
	}
	if _, err := backend.EvaluateTransformer(lm.Transformer, sequence[:1], seed, rows); err == nil {
		t.Fatal("invalid sensitivity boundary")
	}
	backend.batch.VoiceFrames = 0
	if _, err := backend.EvaluateTransformer(lm.Transformer, sequence, seed, rows); err == nil {
		t.Fatal("zero voice boundary")
	}
	if _, err := backend.EvaluateTransformer(nil, sequence, seed, rows); err == nil {
		t.Fatal("nil transformer")
	}
}

func TestPocketSharedReleasedSpeakerSensitivity(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_SHARED_SENSITIVITY") != "1" {
		t.Skip("set GO_PHERENCE_POCKET_SHARED_SENSITIVITY=1 with pinned local assets")
	}
	mode := os.Getenv("GO_PHERENCE_POCKET_SHARED_SENSITIVITY_MODE")
	flowBackwardMode := mode == "flow-linear-backward" || mode == "flow-linear-backward-combined" || mode == "flow-silu-backward" || mode == "flow-silu-backward-combined" || mode == "flow-linear-silu-backward-combined" || mode == "flow-linear-silu-backward-combined-full"
	if !flowBackwardMode && mode != "" && mode != "speaker" && mode != "accumulation" && mode != "reduction" && mode != "gelu" && mode != "norm" && mode != "attention" && mode != "linear-forward" && mode != "linear-gelu-forward" && mode != "linear-backward" && mode != "linear-both" && mode != "linear-gelu-both" && mode != "linear-gelu-norm-forward" && mode != "flow-control" && mode != "flow-forward" && mode != "flow-transformer-forward" && mode != "flow-norm-forward" && mode != "flow-condition" && mode != "flow-condition-combined" && mode != "flow-norm-adjoint" && mode != "flow-norm-adjoint-combined" {
		t.Fatal("explicit supported sensitivity mode required")
	}
	root := os.Getenv("GO_PHERENCE_POCKET_SHARED_SENSITIVITY_OUTPUT")
	if info, err := os.Stat(root); err != nil || !info.IsDir() {
		t.Fatal("existing sensitivity evidence directory required", err)
	}
	pinned := func(path, hash string, max int64) []byte {
		t.Helper()
		info, err := os.Stat(path)
		if err != nil || info.Size() > max {
			t.Fatal("sensitivity input size", err)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		got := sha256.Sum256(raw)
		if hex.EncodeToString(got[:]) != hash {
			t.Fatal("sensitivity input provenance", path)
		}
		return raw
	}
	var corpus struct {
		Rows []PreparedTrainingRow `json:"rows"`
	}
	if err := json.Unmarshal(pinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9", 1<<20), &corpus); err != nil || len(corpus.Rows) != 1 {
		t.Fatal("sensitivity corpus", err)
	}
	var saved struct {
		Identity TrainingRunIdentity `json:"identity"`
		TimeRNG  string              `json:"time_rng"`
		NoiseRNG string              `json:"noise_rng"`
	}
	if err := json.Unmarshal(pinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CPU_RUN"), "e5c1bd21309fb893336028fe363cc192b88030727a2b310f3b0be869c1e8fdbd", 1<<16), &saved); err != nil {
		t.Fatal(err)
	}
	if saved.Identity.Backend != "simd" || saved.Identity.GoVersion != runtime.Version() || saved.Identity.GOARCH != runtime.GOARCH || saved.Identity.GOOS != runtime.GOOS {
		t.Fatal("CPU replay toolchain/platform")
	}
	var selection struct {
		Schema int                      `json:"schema"`
		Points []sharedSensitivityPoint `json:"points"`
	}
	if err := json.Unmarshal(pinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_POINTS"), "da4f9adb490a799b43c66f0269bc7d7850df9d2fb6d7e71bb2ad43b4a2d68b4a", 1<<20), &selection); err != nil || selection.Schema != 1 || len(selection.Points) != 1293 {
		t.Fatal("sensitivity observation points", err)
	}
	// Only identity/seeds/corpus enter sampling. Saved state values remain
	// isolated in selection until completed backward observations below.
	run, err := NewTrainingRunState(saved.Identity, 1, [6]uint64{42, 43, 44, 45, 46, 47})
	if err != nil {
		t.Fatal(err)
	}
	proposal, err := run.PrepareRow()
	if err != nil {
		t.Fatal(err)
	}
	row := corpus.Rows[0]
	if row.Batch.Frames != 14 || row.Batch.VoiceFrames != 13 || row.ValidFrames != 13 || len(row.Batch.TextTokens) != 1 {
		t.Fatal("released sensitivity row")
	}
	samples := runnerSampleBuffers(row.Batch.Frames, 32)
	recipe := TrainingRandomRecipe{Algorithm: TrainingRandomAlgorithm, OptionsSHA256: saved.Identity.OptionsSHA256, DistillProbability: 1, FlowBatchMultiplier: 1, StopGradient: "minimal", Normalize: true, PEqual: .75, EOSLossWeight: .1, LogitMean: .4, LogitStd: 1}
	if err := SampleTrainingInto(samples, proposal, row.Batch.Frames, row.ValidFrames, 32, recipe, TrainingRandomLimits{TrainingSampleLimits{375, 375 * 32}, 2*375*32 + 6*375}); err != nil {
		t.Fatal(err)
	}
	if hex.EncodeToString(proposal.next.timeRNG[:]) != saved.TimeRNG || hex.EncodeToString(proposal.next.noiseRNG[:]) != saved.NoiseRNG {
		t.Fatal("sampling replay mismatch")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	tr, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	workspace, err := NewTrainingWorkspace(lm, flow, row.Batch)
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Description                                                                     string `json:"description"`
		BaselineNorm, InterventionNorm                                                  float64
		BaselineMetrics, InterventionMetrics                                            TrainingStepMetrics
		ModifiedForwardValues                                                           int
		MaxForwardDifference                                                            float64
		Points                                                                          []sharedSensitivityPoint
		Baseline, Intervention                                                          []sharedSensitivityValue
		Reductions                                                                      []sharedReductionObservation
		FlowCalls, FlowOrdinary, FlowDual                                               int
		FlowOutput, FlowTangent                                                         sharedAttentionError
		FlowCondition                                                                   sharedFlowConditionObservation
		FlowNormAdjoint                                                                 sharedFlowNormAdjointObservation
		FlowBackward                                                                    *sharedFlowBackwardPolicy
		LinearAdjoints                                                                  []sharedLinearAdjointObservation
		AttentionObservations                                                           []sharedAttentionLayerObservation
		NormBoundaryValues                                                              int
		NormMaxCPUShader, NormMaxCPUReference, NormMaxShaderReference, NormCPUShaderRMS float64
	}
	result.Description = "Offline speaker-forward reduction intervention, all other arithmetic CPU; diagnostic sampled updates only, no optimizer execution, checkpoint publication or numerical qualification."
	result.Points = selection.Points
	filename := "speaker-sensitivity.json"
	if flowBackwardMode || mode == "accumulation" || mode == "reduction" || mode == "gelu" || mode == "norm" || mode == "attention" || mode == "linear-forward" || mode == "linear-gelu-forward" || mode == "linear-backward" || mode == "linear-both" || mode == "linear-gelu-both" || mode == "linear-gelu-norm-forward" || mode == "flow-control" || mode == "flow-forward" || mode == "flow-transformer-forward" || mode == "flow-norm-forward" || mode == "flow-condition" || mode == "flow-condition-combined" || mode == "flow-norm-adjoint" || mode == "flow-norm-adjoint-combined" {
		// Reuse saved diagnostic control observations, never feed them to execution.
		raw, err := os.ReadFile(root + "/speaker-sensitivity.json")
		if err != nil {
			t.Fatal(err)
		}
		var prior struct {
			BaselineNorm    float64
			BaselineMetrics TrainingStepMetrics
			Baseline        []sharedSensitivityValue
		}
		if err := json.Unmarshal(raw, &prior); err != nil || len(prior.Baseline) != len(selection.Points) {
			t.Fatal("saved diagnostic control", err)
		}
		result.BaselineNorm, result.BaselineMetrics, result.Baseline = prior.BaselineNorm, prior.BaselineMetrics, prior.Baseline
		backend := &sharedAccumulationSensitivityBackend{}
		observer := &sharedReductionSensitivityBackend{points: selection.Points}
		normObserver := &sharedNormSensitivityBackend{}
		attentionObserver := &sharedAttentionDiagnosticBackend{}
		backwardCombined := flowBackwardMode && (strings.HasSuffix(mode, "-combined") || strings.HasSuffix(mode, "-combined-full"))
		var backwardPolicy *sharedFlowBackwardPolicy
		if flowBackwardMode {
			backwardPolicy = &sharedFlowBackwardPolicy{Linear: strings.HasPrefix(mode, "flow-linear-"), SiLU: strings.HasPrefix(mode, "flow-silu-") || strings.HasPrefix(mode, "flow-linear-silu-")}
		}
		linearForwardObserver := &sharedLinearForwardSensitivityBackend{shaderGelu: mode == "linear-gelu-forward" || mode == "linear-gelu-norm-forward" || mode == "flow-transformer-forward" || mode == "flow-condition-combined" || mode == "flow-norm-adjoint-combined" || backwardCombined, shaderNorm: mode == "linear-gelu-norm-forward"}
		linearBackwardObserver := &sharedLinearBackwardSensitivityBackend{serialForward: mode == "linear-both" || mode == "linear-gelu-both", shaderDerivative: mode == "linear-gelu-both"}
		flowForwardObserver := &sharedFlowForwardSensitivityBackend{serial: mode == "flow-forward" || mode == "flow-transformer-forward" || mode == "flow-norm-forward" || mode == "flow-condition-combined" || mode == "flow-norm-adjoint-combined" || backwardCombined, backwardPolicy: backwardPolicy, shaderNorm: mode == "flow-norm-forward", serialCondition: mode == "flow-condition" || mode == "flow-condition-combined", normAdjoint: mode == "flow-norm-adjoint" || mode == "flow-norm-adjoint-combined"}
		if mode == "flow-control" || mode == "flow-forward" || mode == "flow-transformer-forward" || mode == "flow-norm-forward" || mode == "flow-condition" || mode == "flow-condition-combined" || mode == "flow-norm-adjoint" || mode == "flow-norm-adjoint-combined" || flowBackwardMode {
			flow.TrainingBackend = flowForwardObserver
			if backwardCombined || mode == "flow-transformer-forward" || mode == "flow-condition-combined" || mode == "flow-norm-adjoint-combined" {
				lm.TransformerBackend = linearForwardObserver
			}
		} else if mode == "reduction" || mode == "gelu" {
			lm.TransformerBackend = observer
		} else if mode == "norm" {
			lm.TransformerBackend = normObserver
		} else if mode == "attention" {
			lm.TransformerBackend = attentionObserver
		} else if mode == "linear-forward" || mode == "linear-gelu-forward" || mode == "linear-gelu-norm-forward" {
			lm.TransformerBackend = linearForwardObserver
		} else if mode == "linear-backward" || mode == "linear-both" || mode == "linear-gelu-both" {
			lm.TransformerBackend = linearBackwardObserver
		} else {
			flow.TrainingBackend = backend
		}
		metrics, g, err := PocketTrainingStepInto(lm, flow, w, row.Batch, samples, DefaultTrainingStepConfig(), workspace)
		if err != nil {
			t.Fatal(err)
		}
		if mode == "flow-control" || mode == "flow-forward" || mode == "flow-transformer-forward" || mode == "flow-norm-forward" || mode == "flow-condition" || mode == "flow-condition-combined" || mode == "flow-norm-adjoint" || mode == "flow-norm-adjoint-combined" || flowBackwardMode {
			result.FlowBackward = backwardPolicy
			result.FlowCondition = flowForwardObserver.condition
			result.FlowNormAdjoint = flowForwardObserver.normObservation
			if backwardCombined || mode == "flow-transformer-forward" || mode == "flow-condition-combined" || mode == "flow-norm-adjoint-combined" {
				result.AttentionObservations = linearForwardObserver.layers
				result.ModifiedForwardValues, result.MaxForwardDifference = linearForwardObserver.modified, linearForwardObserver.maxDifference
			}
			if flowForwardObserver.calls != 6*row.ValidFrames || flowForwardObserver.dual != 2*row.ValidFrames {
				t.Fatal("flow call inventory", flowForwardObserver.calls, flowForwardObserver.dual)
			}
			result.FlowCalls, result.FlowOrdinary, result.FlowDual = flowForwardObserver.calls, flowForwardObserver.ordinary, flowForwardObserver.dual
			result.FlowOutput, result.FlowTangent = flowForwardObserver.output, flowForwardObserver.tangent
		} else if mode == "accumulation" {
			if backend.calls != 2*row.ValidFrames || backend.gradients == nil {
				t.Fatal("branch contribution count", backend.calls)
			}
			g.Flow = backend.gradients // independently computed local accumulated G
		} else if mode == "linear-backward" || mode == "linear-both" || mode == "linear-gelu-both" {
			if len(linearBackwardObserver.observations) != len(lm.Transformer.Layers) {
				t.Fatal("linear adjoint observation layer count")
			}
			result.LinearAdjoints = linearBackwardObserver.observations
		} else if mode == "linear-forward" || mode == "linear-gelu-forward" || mode == "linear-gelu-norm-forward" {
			result.AttentionObservations = linearForwardObserver.layers
			result.ModifiedForwardValues, result.MaxForwardDifference = linearForwardObserver.modified, linearForwardObserver.maxDifference
		} else if mode == "attention" {
			if len(attentionObserver.observations) != len(lm.Transformer.Layers) {
				t.Fatal("attention observation layer count")
			}
			result.AttentionObservations = attentionObserver.observations
		} else if mode == "norm" {
			result.NormBoundaryValues = normObserver.values
			result.NormMaxCPUShader, result.NormMaxCPUReference, result.NormMaxShaderReference = normObserver.maxCPUShader, normObserver.maxCPUReference, normObserver.maxShaderReference
			result.NormCPUShaderRMS = math.Sqrt(normObserver.CPUShaderSquare / float64(normObserver.values))
		} else {
			if len(observer.observations) != 663 {
				t.Fatal("FC2 observation count", len(observer.observations))
			}
			result.Reductions = observer.observations
		}
		result.InterventionMetrics = metrics
		result.Intervention, result.InterventionNorm = sharedSensitivityObservations(t, tr, g, selection.Points)
		if mode == "flow-linear-silu-backward-combined-full" {
			sharedFullParameterDistribution(t, tr, g, selection.Points, result.InterventionNorm, root)
		}
		result.Description = "Offline CPU local-branch post-VJP objective scaling intervention; other paths CPU. Prior diagnostic baseline reused as observation data only. No optimizer execution or shader emulation."
		filename = "accumulation-sensitivity.json"
		if mode == "flow-control" || mode == "flow-forward" || mode == "flow-transformer-forward" || mode == "flow-norm-forward" || mode == "flow-condition" || mode == "flow-condition-combined" || mode == "flow-norm-adjoint" || mode == "flow-norm-adjoint-combined" || flowBackwardMode {
			filename = mode + "-sensitivity.json"
			result.Description = "Offline explicit non-accumulator flow backend paired control/serial-F32 forward-linear intervention. CPU ordinary/dual backward retained, local gradients receive outer objective scaling. Host time/SiLU unchanged; flow-norm-forward additionally uses forward F32 sqrt/reciprocal norm formula with CPU backward retained; flow-transformer-forward also applies prior test-only transformer serial-linear/forward-GELU. flow-condition substitutes only the last condition projection dInput reduction; flow-condition-combined adds that to combined forward. flow-norm-adjoint uses dual norm VJP GLSL expression with CPU stats; its combined variant adds this to prior combined forward, ordinary backward unchanged. Explicit flow-linear-backward/flow-silu-backward policies replace only corresponding adjoint formulas in ordinary and dual paths, combined variants add prior combined forward. No device/optimizer execution; expected observations isolated."
		} else if mode == "reduction" || mode == "gelu" {
			filename = mode + "-sensitivity.json"
			result.Description = "Offline FC2 dWeight observations at fixed CPU forward inputs/backward seeds; sequentialF32, compensatedF64 and shader-GELU F32 activation emulation are never used by AD, clipping or optimizer."
		} else if mode == "linear-backward" || mode == "linear-both" || mode == "linear-gelu-both" {
			filename = mode + "-sensitivity.json"
			result.Description = "Offline transformer serial linear input-adjoint intervention; CPU dWeight/dBias/norm/GELU derivative/RoPE retained. Explicit linear-both/linear-gelu-both additionally changes transformer forward linear/GELU formulas; linear-gelu-both also uses host shader GELU derivative. No device or optimizer execution; saved values never enter AD."
		} else if mode == "linear-forward" || mode == "linear-gelu-forward" || mode == "linear-gelu-norm-forward" {
			filename = mode + "-sensitivity.json"
			result.Description = "Offline transformer-only serial F32 linear forward reduction intervention; CPU norm/RoPE/backward retained. Explicit linear-gelu-forward/linear-gelu-norm-forward additionally uses host GELU GLSL formula; the latter also rounds sqrt before F32 reciprocal in forward LayerNorm. CPU backward retained. No device execution or optimizer update. Saved observations are never injected."
		} else if mode == "attention" {
			filename = "attention-sensitivity.json"
			result.Description = "Offline input-fixed serial F32 attention forward/adjoint and projection reduction observations, compared with F64 adjoints. Normal CPU values enter AD. Host exp cannot qualify device transcendental arithmetic; no optimizer execution."
		} else if mode == "norm" {
			filename = "norm-sensitivity.json"
			result.Description = "Offline final transformer norm adjoint intervention with CPU forward and other backward operations unchanged. F32 GLSL expression emulation compared to independent analytic F64 reference; host arithmetic cannot qualify actual device execution."
		}
	} else {
		metrics, g, err := PocketTrainingStepInto(lm, flow, w, row.Batch, samples, DefaultTrainingStepConfig(), workspace)
		if err != nil {
			t.Fatal(err)
		}
		result.BaselineMetrics = metrics
		result.Baseline, result.BaselineNorm = sharedSensitivityObservations(t, tr, g, selection.Points)
		g = TrainingStepGradients{}
		backend := &sharedSpeakerSensitivityBackend{batch: row.Batch, projection: lm.SpeakerProjection}
		lm.TransformerBackend = backend
		metrics, g, err = PocketTrainingStepInto(lm, flow, w, row.Batch, samples, DefaultTrainingStepConfig(), workspace)
		if err != nil {
			t.Fatal(err)
		}
		result.InterventionMetrics = metrics
		result.Intervention, result.InterventionNorm = sharedSensitivityObservations(t, tr, g, selection.Points)
		result.ModifiedForwardValues, result.MaxForwardDifference = backend.modified, backend.maxDifference
	}
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := root + "/" + filename
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, writeErr := file.Write(append(data, '\n'))
	closeErr := file.Close()
	if writeErr != nil || closeErr != nil {
		t.Fatal(writeErr, closeErr)
	}
	t.Logf("%s: baselineLoss=%g interventionLoss=%g baselineNorm=%g interventionNorm=%g changedForwardValuesAcross2Calls=%d maxForwardDifference=%g observedPoints=%d", filename, result.BaselineMetrics.Loss, result.InterventionMetrics.Loss, result.BaselineNorm, result.InterventionNorm, result.ModifiedForwardValues, result.MaxForwardDifference, len(selection.Points))
}
