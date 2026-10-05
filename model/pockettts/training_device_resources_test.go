package pockettts

import (
	"fmt"
	"os"
	"sort"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// Shape-only audit fixtures. No model asset loading or signal/model execution.
// Synthetic immutable positive-zero parameter inventories share host backing by
// extent to keep this audit bounded; native residency is charged independently
// for every named parameter by ResidentBytes, irrespective of host sharing.
func releasedDeviceAuditFixture(t *testing.T) (Config, *FlowHeadCPU, map[int][]float32) {
	t.Helper()
	data, err := os.ReadFile("testdata/english-upstream.yaml")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	if err = releasedTrainingTopology(cfg); err != nil {
		t.Fatal(err)
	}
	pool := map[int][]float32{}
	values := func(n int) []float32 {
		if n <= 0 {
			t.Fatal("audit extent")
		}
		if pool[n] == nil {
			pool[n] = make([]float32, n)
		}
		return pool[n]
	}
	linear := func(in, out int) LinearF32 {
		return LinearF32{In: in, Out: out, Weight: values(in * out), Bias: values(out)}
	}
	width, hidden, latent := cfg.FlowLM.Flow.Dim, cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim
	flow := &FlowHeadCPU{Input: linear(latent, width), Condition: linear(hidden, width), Time: make([]TimestepMLP, 2), Blocks: make([]AdaLNResidual, cfg.FlowLM.Flow.Depth)}
	for i := range flow.Time {
		flow.Time[i] = TimestepMLP{Frequencies: append([]float32(nil), values(128)...), FC1: linear(256, width), FC2: linear(width, width), RMSWeight: values(width), RMSEpsilon: 1e-5}
	}
	for i := range flow.Blocks {
		flow.Blocks[i] = AdaLNResidual{NormWeight: values(width), NormBias: values(width), FC1: linear(width, width), FC2: linear(width, width), Modulation: linear(width, 3*width), Epsilon: 1e-6}
	}
	flow.Final = AdaLNFinal{Linear: linear(width, latent), Modulation: linear(width, 2*width), Epsilon: 1e-6}
	return cfg, flow, pool
}
func deviceAuditTapeBytes(t *testing.T, tape train.Tape) int64 {
	t.Helper()
	var n int64
	for _, s := range tape.Tensors {
		size, err := s.Elements()
		if err != nil {
			t.Fatal(err)
		}
		n += 4 * int64(size)
	}
	return n
}
func deviceAuditStorage(t *testing.T, tape train.Tape, bindings map[string]train.ResidentTapeBinding, roots []int) int64 {
	t.Helper()
	if err := train.ValidateResidentTapeBindings(tape, bindings, 2<<30); err != nil {
		t.Fatal(err)
	}
	p, m, err := train.PruneTape(tape, roots, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	protected := map[int]bool{}
	for _, b := range bindings {
		if m[b.Parameter] >= 0 {
			protected[m[b.Parameter]] = true
		}
	}
	storage, err := train.PlanZeroTapeStorage(p, 2<<30, protected)
	if err != nil {
		t.Fatal(err)
	}
	return storage.Bytes
}
func TestTrainingReleasedDeviceGraphResources(t *testing.T) {
	cfg, flow, _ := releasedDeviceAuditFixture(t)
	h, n := cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim
	diag, err := flow.buildDiagonalDeviceTrainingTape(h, n, .3, 0, .25, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	distill, err := flow.buildDistillDeviceTrainingTape(h, n, .3, .8, 0, make([]float32, n), make([]float32, n), 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	in := DiagonalForwardInputs{Condition: make([]float32, h), Target: make([]float32, n), Noise: make([]float32, n), Time: .3}
	forwardDiag, err := flow.buildDiagonalForwardTape(in, 2<<30, true)
	if err != nil {
		t.Fatal(err)
	}
	forwardDistill, err := flow.buildDistillForwardTape(DistillForwardInputs{Condition: in.Condition, Target: in.Target, Noise: in.Noise, S: .3, T: .8}, 2<<30, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, fixture := range []struct {
		name     string
		tape     train.Tape
		bindings map[string]train.ResidentTapeBinding
		roots    []int
	}{
		{"diagonal-forward", forwardDiag.Tape, forwardDiag.ResidentBindings, forwardDiag.Tape.Outputs[:3]},
		{"distill-forward", forwardDistill.Tape, forwardDistill.ResidentBindings, forwardDistill.Tape.Outputs[:6]},
		{"diagonal-backward", diag.tape, diag.bindings, diag.tape.Outputs},
		{"distill-backward", distill.tape, distill.bindings, distill.tape.Outputs},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			bytes := deviceAuditTapeBytes(t, fixture.tape)
			if len(fixture.tape.Tensors) > 4096 || len(fixture.tape.Commands) > 4096 {
				t.Fatal("released graph exceeds metadata ceilings")
			}
			if fixture.tape.Validate(bytes-1) == nil {
				t.Fatal("fullsource onebyte under")
			}
			if err := fixture.tape.Validate(bytes); err != nil {
				t.Fatal(err)
			}
			planned := deviceAuditStorage(t, fixture.tape, fixture.bindings, fixture.roots)
			t.Logf("released shape graph %s depth%d width%d hidden%d latent%d tensors%d commands%d logicalbytes%d pruned-zero-lifetimebytes%d params%d (shape/synthetic only; no model/device execution)", fixture.name, len(flow.Blocks), flow.Input.Out, h, n, len(fixture.tape.Tensors), len(fixture.tape.Commands), bytes, planned, len(fixture.bindings))
		})
	}
}

// Complete trainable parameter extent inventory follows addFlowLMParameters /
// addTransformerParameters plus actual synthetic flow map and default weighting
// shapes. Nontrainable embedding stats/frequencies do not enter resident state.
func releasedDeviceAuditInventory(t *testing.T, cfg Config, flow *FlowHeadCPU) map[string]int {
	t.Helper()
	inventory := map[string]int{}
	flowParams := map[string][]float32{}
	addFlowHeadParameters(flowParams, flow)
	for name, p := range flowParams {
		inventory[name] = len(p)
	}
	h, n := cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim
	ff := h * cfg.FlowLM.Transformer.HiddenScale
	// Released training retains the pad-token row as part of its trainable map.
	inventory["flow_lm.conditioner.embed.weight"] = (cfg.FlowLM.LookupTable.NBins + 1) * h
	inventory["flow_lm.bos_emb"], inventory["flow_lm.bos_before_voice"] = n, h
	inventory["flow_lm.speaker_proj_weight"], inventory["flow_lm.input_linear.weight"] = h*n, h*n
	inventory["flow_lm.out_eos.weight"], inventory["flow_lm.out_eos.bias"] = h, 1
	inventory["flow_lm.out_norm.weight"], inventory["flow_lm.out_norm.bias"] = h, h
	for i := 0; i < cfg.FlowLM.Transformer.NumLayers; i++ {
		p := fmt.Sprintf("flow_lm.transformer.layers.%d", i)
		for _, suffix := range []string{"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"} {
			inventory[p+"."+suffix] = h
		}
		inventory[p+".self_attn.in_proj.weight"] = 3 * h * h
		inventory[p+".self_attn.out_proj.weight"] = h * h
		inventory[p+".linear1.weight"], inventory[p+".linear2.weight"] = ff*h, ff*h
	}
	w := NewDefaultLSDWeightMLP()
	for i, l := range w.Layers {
		p := fmt.Sprintf("flow.w_s_t.%d", 2*i)
		inventory[p+".weight"], inventory[p+".bias"] = len(l.Weight), len(l.Bias)
	}
	return inventory
}
func TestTrainingReleasedDeviceResidentResources(t *testing.T) {
	cfg, flow, pool := releasedDeviceAuditFixture(t)
	inventory := releasedDeviceAuditInventory(t, cfg, flow)
	// No constructor invoked: ResidentBytes is pure admission only. Sharing host
	// zero arrays reduces fixture storage, never the logical resident charge.
	snapshot := train.ResidentSnapshot{Config: train.ResidentConfig{LearningRate: .001, Beta1: .9, Beta2: .99, Epsilon: 1e-8, EMADecay: .9}, Parameters: map[string]train.ResidentParameter{}}
	var parameterElements, statsElements int64
	names := make([]string, 0, len(inventory))
	for name := range inventory {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		n := inventory[name]
		if pool[n] == nil {
			pool[n] = make([]float32, n)
		}
		x := pool[n]
		snapshot.Parameters[name] = train.ResidentParameter{Parameter: x, M: x, V: x, EMA: x}
		parameterElements += int64(n)
		stats, err := train.NormStatsShape(n)
		if err != nil {
			t.Fatal(err)
		}
		statsElements += int64(stats)
	}
	bytes, err := train.ResidentBytes(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	expected := parameterElements*32 + statsElements*4
	if bytes != expected {
		t.Fatal("resident inventory charge", bytes, expected)
	}
	// Global norm/device-scale/device-statecheck adds exactly 12 F32 elements.
	total := bytes + 48
	t.Logf("released synthetic inventory parameters%d elements%d F32parambytes%d resident8buffersplusstats%d globals48 total%d CLIdefault%d CLImaximum%d overmax%d (logical only; no constructor/model/device)", len(inventory), parameterElements, 4*parameterElements, bytes, total, int64(256<<20), int64(2<<30), total-int64(2<<30))
	if total <= 2<<30 {
		t.Fatal("released full resident unexpectedly fits existing CLI max; audit assumptions need review")
	}
}

func TestTrainingReleasedDeviceInventoryMapping(t *testing.T) {
	cfg, flow, pool := releasedDeviceAuditFixture(t)
	zero := func(n int) []float32 {
		if pool[n] == nil {
			pool[n] = make([]float32, n)
		}
		return pool[n]
	}
	h, n, ff := cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim, cfg.FlowLM.Transformer.DModel*cfg.FlowLM.Transformer.HiddenScale
	linear := func(in, out int) LinearF32 { return LinearF32{In: in, Out: out, Weight: zero(in * out)} }
	backbone := &TransformerCPU{Width: h, Heads: cfg.FlowLM.Transformer.NumHeads, HeadDim: h / cfg.FlowLM.Transformer.NumHeads, MaxPeriod: cfg.FlowLM.Transformer.MaxPeriod, Layers: make([]TransformerLayerCPU, cfg.FlowLM.Transformer.NumLayers), FinalWeight: zero(h), FinalBias: zero(h)}
	for i := range backbone.Layers {
		backbone.Layers[i] = TransformerLayerCPU{Norm1Weight: zero(h), Norm1Bias: zero(h), Norm2Weight: zero(h), Norm2Bias: zero(h), InProjection: linear(h, 3*h), OutProjection: linear(h, h), FC1: linear(h, ff), FC2: linear(ff, h)}
	}
	lm := &FlowLMTrainingCPU{Transformer: backbone, Embedding: zero((cfg.FlowLM.LookupTable.NBins + 1) * h), BOS: zero(n), BOSBeforeVoice: zero(h), SpeakerProjection: linear(n, h), Input: linear(n, h), EOS: linear(h, 1), Hidden: h, LatentDim: n, Vocabulary: cfg.FlowLM.LookupTable.NBins + 1}
	lm.EOS.Bias = zero(1)
	// Cross-check manual resource inventory with the actual full trainer map,
	// without constructing a trainer, evaluating a model or allocating a driver.
	actual, err := fullParameterMap(lm, flow, NewDefaultLSDWeightMLP())
	if err != nil {
		t.Fatal(err)
	}
	expected := releasedDeviceAuditInventory(t, cfg, flow)
	var elements int64
	for _, extent := range expected {
		elements += int64(extent)
	}
	planned, ok := trainingParameterElements(cfg)
	if !ok || elements != planned {
		t.Fatal("released resource manifest/planner mismatch", elements, planned)
	}
	if len(actual) != len(expected) {
		t.Fatal("released inventory names", len(actual), len(expected))
	}
	for name, extent := range expected {
		p, ok := actual[name]
		if !ok || len(p) != extent {
			t.Fatal("released inventory extent", name, len(p), extent)
		}
	}
	t.Logf("actual fullParameterMap matches independent released extent inventory: %d named parameters", len(actual))
}

func TestTrainingReleasedDeviceCombinedSourceCeilings(t *testing.T) {
	cfg, flow, _ := releasedDeviceAuditFixture(t)
	h, n := cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim
	target, noise := make([]float32, n), make([]float32, n)
	diagonal, err := flow.buildDiagonalDeviceTrainingTape(h, n, .3, 0, .25, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	distill, err := flow.buildDistillDeviceTrainingTape(h, n, .3, .8, 0, target, noise, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	df, err := flow.buildDiagonalForwardTape(DiagonalForwardInputs{Condition: make([]float32, h), Target: target, Noise: noise, Time: .3}, 2<<30, true)
	if err != nil {
		t.Fatal(err)
	}
	sf, err := flow.buildDistillForwardTape(DistillForwardInputs{Condition: make([]float32, h), Target: target, Noise: noise, S: .3, T: .8}, 2<<30, true)
	if err != nil {
		t.Fatal(err)
	}
	diagPair := deviceAuditTapeBytes(t, df.Tape) + deviceAuditTapeBytes(t, diagonal.tape)
	distillPair := deviceAuditTapeBytes(t, sf.Tape) + deviceAuditTapeBytes(t, distill.tape)
	if diagPair >= 256<<20 || distillPair <= 256<<20 {
		t.Fatal("released combinedsource assumption", diagPair, distillPair)
	}
	// These exclude the transformer parent and row-condition child. They are
	// strict lower bounds on the coordinator's live-source admission, not peaks.
	t.Logf("released live-source lowerbounds diagonal forward+backward=%d distill forward+backward=%d default=%d; transformer-parent/condition-child/native-overhead excluded", diagPair, distillPair, int64(256<<20))
}

func TestTrainingReleasedDeviceSixBufferLowerBound(t *testing.T) {
	cfg, flow, _ := releasedDeviceAuditFixture(t)
	inventory := releasedDeviceAuditInventory(t, cfg, flow)
	var elements, stats int64
	for _, n := range inventory {
		elements += int64(n)
		count, err := train.NormStatsShape(n)
		if err != nil {
			t.Fatal(err)
		}
		stats += int64(count)
	}
	// Hypothetical removal of branch-upload and zero arrays still leaves P,
	// two gradient ping-pong destinations, M, V, EMA and existing norm statistics.
	// This is an arithmetic lower bound for that proposal, not implemented state.
	sixBuffers := 24*elements + 4*stats + 48
	if sixBuffers <= 2<<30 {
		t.Fatal("six-buffer floor unexpectedly admits")
	}
	t.Logf("hypothetical six-buffer resident floor P/G/Gscratch/M/V/EMA plus stats+globals=%d CLImax=%d excess=%d; removing branch/zero alone cannot resolve released admission", sixBuffers, int64(2<<30), sixBuffers-int64(2<<30))
}

// Design arithmetic only. No proposed allocator/API is implemented or invoked.
func TestTrainingReleasedDeviceSharedWorkspaceProposal(t *testing.T) {
	cfg, flow, _ := releasedDeviceAuditFixture(t)
	inventory := releasedDeviceAuditInventory(t, cfg, flow)
	var elements, statsSum, maxElements, maxStats int64
	largest := ""
	for name, n := range inventory {
		elements += int64(n)
		stats, err := train.NormStatsShape(n)
		if err != nil {
			t.Fatal(err)
		}
		statsSum += int64(stats)
		if int64(n) > maxElements {
			maxElements = int64(n)
			largest = name
		}
		if int64(stats) > maxStats {
			maxStats = int64(stats)
		}
	}
	current := 32*elements + 4*statsSum + 48
	proposed := 20*elements + 12*maxElements + 4*maxStats + 48
	headroom := int64(2<<30) - proposed
	t.Logf("DESIGN ONLY persistent5buffers=%d sharedscratch-branch-zero=%d sharedstats=%d globals48 proposed=%d old=%d reduction=%d capheadroom=%d largest=%s elements%d maxstatselements%d", 20*elements, 12*maxElements, 4*maxStats, proposed, current, current-proposed, headroom, largest, maxElements, maxStats)
	if headroom <= 0 || maxElements != 4194304 {
		t.Fatal("design accounting assumption", headroom, maxElements)
	}
	distillPair := int64(476038888) // verified complete-source pair in audit, excludes parent/condition
	if proposed+distillPair <= 2<<30 {
		t.Fatal("coexistence unexpectedly fits total cap")
	}
	t.Logf("DESIGN ONLY proposal+known-complete-distill-source-lowerbound=%d total2GiBexcess=%d transformer-parent/condition/native/hostcopies excluded; resident fits alone, combined safety NOT established", proposed+distillPair, proposed+distillPair-int64(2<<30))
	for _, peakMiB := range []int64{64, 128, 256, 512} {
		bytes := peakMiB << 20
		t.Logf("DESIGN sensitivity unborrowed-live-tape-and-native-reserve=%dMiB remainingresidentcapheadroom=%d (scenario not measurement)", peakMiB, headroom-bytes)
	}
}

func TestTrainingReleasedDeviceProposalCoexistence(t *testing.T) {
	cfg, flow, _ := releasedDeviceAuditFixture(t)
	h, n := cfg.FlowLM.Transformer.DModel, cfg.Mimi.InnerDim
	forward, err := flow.buildDistillForwardTape(DistillForwardInputs{Condition: make([]float32, h), Target: make([]float32, n), Noise: make([]float32, n), S: .3, T: .8}, 2<<30, true)
	if err != nil {
		t.Fatal(err)
	}
	backward, err := flow.buildDistillDeviceTrainingTape(h, n, .3, .8, 0, make([]float32, n), make([]float32, n), 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	ownedSlots := func(tape train.Tape, bindings map[string]train.ResidentTapeBinding, roots []int) int64 {
		t.Helper()
		p, m, err := train.PruneTape(tape, roots, 2<<30)
		if err != nil {
			t.Fatal(err)
		}
		protected := map[int]bool{}
		borrowed := map[int]bool{}
		for _, b := range bindings {
			if m[b.Parameter] >= 0 {
				protected[m[b.Parameter]] = true
				borrowed[m[b.Parameter]] = true
			}
		}
		for id, s := range p.Tensors {
			if s.External {
				borrowed[id] = true
			}
		}
		plan, err := train.PlanZeroTapeStorage(p, 2<<30, protected)
		if err != nil {
			t.Fatal(err)
		}
		slots := map[int]bool{}
		for id := range borrowed {
			slots[plan.Slots[id]] = true
		}
		owned := plan.Bytes
		for slot := range slots {
			owned -= 4 * int64(plan.Elements[slot])
		}
		if owned < 0 {
			t.Fatal("negative slot cost")
		}
		return owned
	}
	fowned := ownedSlots(forward.Tape, forward.ResidentBindings, forward.Tape.Outputs[:6])
	bowned := ownedSlots(backward.tape, backward.bindings, backward.tape.Outputs)
	inventory := releasedDeviceAuditInventory(t, cfg, flow)
	var elements, maxN, maxStats int64
	for _, n := range inventory {
		elements += int64(n)
		if int64(n) > maxN {
			maxN = int64(n)
		}
		stats, err := train.NormStatsShape(n)
		if err != nil {
			t.Fatal(err)
		}
		if int64(stats) > maxStats {
			maxStats = int64(stats)
		}
	}
	proposed := 20*elements + 12*maxN + 4*maxStats + 48
	headroom := int64(2<<30) - proposed
	t.Logf("DESIGN synthetic-zero pruned unborrowed slots forward=%d backward=%d sum=%d residentproposal=%d remaining2GiB=%d transformer-parent/condition/allocator/host omitted; NOT native peak or complete-request estimate", fowned, bowned, fowned+bowned, proposed, headroom-fowned-bowned)
	t.Logf("DESIGN snapshot coexistence F32 P/M/V/EMA per complete host set=%d three independent live-trainer/backend-boundary/new-snapshot sets=%d graphmetadata/inputbuffers/hostgradients/checkpointcodec omitted; arrays lifetime separately budgeted", 16*elements, 48*elements)
}
