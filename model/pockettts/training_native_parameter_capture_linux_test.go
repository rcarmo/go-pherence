//go:build linux

package pockettts

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"

	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

type parameterRow struct {
	Row                                                                                                   int
	DiagonalInput, DiagonalPrediction, PrimaryVelocity, PrimaryTangent, EndpointInput, EndpointPrediction []float32
}
type parameterInputs struct {
	Base allrowInputs
	Rows []parameterRow
	Step StepReductionInputs
}

func parameterPrepare(t *testing.T) parameterInputs {
	t.Helper()
	in := allrowPrepare(t)
	var saved struct {
		Rows   []parameterRow
		Inputs StepReductionInputs
	}
	path, pin := "/workspace/exports/pocket-native-allrow-20261004/native-allrow.json", "aa7e79fdf84b727936a462e7d69cbdb31a6bf153d0183f8adbb2eccd2251d6b6"
	if p := os.Getenv("GO_PHERENCE_POCKET_PARAMETER_ALLROW_INPUT"); p != "" {
		path, pin = p, os.Getenv("GO_PHERENCE_POCKET_PARAMETER_ALLROW_SHA256")
	}
	if e := flowCapturePinned(path, pin, 8<<20, &saved); e != nil {
		t.Fatal(e)
	}
	if len(saved.Rows) != 13 {
		t.Fatal("parameter rows")
	}
	for i, r := range saved.Rows {
		if r.Row != i {
			t.Fatal("row order")
		}
		for _, v := range [][]float32{r.DiagonalInput, r.DiagonalPrediction, r.PrimaryVelocity, r.PrimaryTangent, r.EndpointInput, r.EndpointPrediction} {
			if len(v) != 32 || !finiteF32(v) {
				t.Fatal("parameter row extent")
			}
		}
	}
	if _, e := validateStepReductionInputs(saved.Inputs, DefaultTrainingStepConfig()); e != nil {
		t.Fatal(e)
	}
	return parameterInputs{in, saved.Rows, saved.Inputs}
}
func productionObjectiveScales(valid int) (float32, float32, error) {
	if valid < 1 || valid > 1024 {
		return 0, 0, fmt.Errorf("objective rows")
	}
	inv := float32(1) / float32(valid)
	return float32(.75) * inv, float32(.25) * inv, nil
}
func parameterGradientGraph(tape train.Tape, bindings map[string]train.ResidentTapeBinding, boundary int, previous map[string][]float32, scale float32) (train.Tape, []string, error) {
	if !isFinite(scale) || scale < 0 || boundary < 1 || boundary > len(tape.Outputs) {
		return train.Tape{}, nil, fmt.Errorf("parameter graph boundary/scale")
	}
	if e := train.ValidateResidentTapeBindings(tape, bindings, 2<<30); e != nil {
		return train.Tape{}, nil, e
	}
	names := make([]string, 0, len(bindings))
	for name := range bindings {
		names = append(names, name)
	}
	sort.Strings(names)
	if previous != nil && len(previous) != len(names) {
		return train.Tape{}, nil, fmt.Errorf("previous gradient inventory")
	}
	roots := append([]int(nil), tape.Outputs[:boundary]...)
	for _, name := range names {
		roots = append(roots, bindings[name].Gradient)
	}
	compact, mapping, e := train.PruneTape(tape, roots, 2<<30)
	if e != nil {
		return train.Tape{}, nil, e
	}
	g := &flowTapeGraph{tape: compact}
	g.tape.Outputs = append([]int(nil), compact.Outputs[:boundary]...)
	for _, name := range names {
		id := mapping[bindings[name].Gradient]
		n, e := g.tape.Tensors[id].Elements()
		if e != nil {
			return train.Tape{}, nil, e
		}
		prev := previous[name]
		if previous == nil {
			prev = make([]float32, n)
		}
		if len(prev) != n || !finiteF32(prev) {
			return train.Tape{}, nil, fmt.Errorf("previous gradient %s", name)
		}
		// Explicit boundary input is independently owned; tape borrows it read-only.
		input := g.tensor([]int{n}, prev)
		sum := g.command("vector", []int{input, id}, []int{n}, 0, 0, 0, scale, 0)
		g.tape.Outputs = append(g.tape.Outputs, sum)
	}
	if e = g.tape.Validate(2 << 30); e != nil {
		return train.Tape{}, nil, e
	}
	return g.tape, names, nil
}
func parameterDiagonal(m *FlowHeadCPU, c, target, noise []float32, tm float32, r parameterRow, previous map[string][]float32, scale float32) (train.Tape, []string, error) {
	desired := make([]float32, len(target))
	for i := range desired {
		desired[i] = target[i] - noise[i]
	}
	p, e := m.buildDiagonalTrainingTape(c, tm, r.DiagonalInput, r.DiagonalPrediction, desired, 0, scale, 2<<30, true)
	if e != nil {
		return train.Tape{}, nil, e
	}
	return parameterGradientGraph(p.Tape, p.ResidentBindings, 3, previous, 1)
}
func parameterDistill(m *FlowHeadCPU, c, target, noise []float32, s, tm float32, r parameterRow, previous map[string][]float32, scale float32) (train.Tape, []string, error) {
	desired := make([]float32, len(target))
	for i := range desired {
		desired[i] = r.PrimaryVelocity[i] + float32((tm-s)*r.PrimaryTangent[i])
	}
	xs := make([]float32, len(target))
	for i := range xs {
		xs[i] = float32(s*target[i]) + float32((1-s)*noise[i])
	}
	p, e := m.buildScaledDistillTrainingTape(DistillTrainingInputs{Condition: c, PrimaryInput: xs, EndpointInput: r.EndpointInput, Prediction: r.EndpointPrediction, Desired: desired, Velocity: r.PrimaryVelocity, Tangent: r.PrimaryTangent, Target: target, Noise: noise, S: s, T: tm, LogVariance: 0}, 2<<30, true, scale)
	if e != nil {
		return train.Tape{}, nil, e
	}
	return parameterGradientGraph(p.Tape, p.ResidentBindings, 5, previous, 1)
}
func parameterCPUReference(m *FlowHeadCPU, c, target, noise []float32, tm, s, dt float32, r parameterRow, distill bool) (FlowDualTrainingResult, []float32, error) {
	n := len(target)
	if !distill {
		desired := make([]float32, n)
		for i := range desired {
			desired[i] = target[i] - noise[i]
		}
		_, pred, e := m.forwardTraining(c, []float32{tm, tm}, r.DiagonalInput)
		if e != nil {
			return FlowDualTrainingResult{}, nil, e
		}
		seed, e := reverseLossSeed(pred, desired)
		if e != nil {
			return FlowDualTrainingResult{}, nil, e
		}
		diag, _, e := productionObjectiveScales(13)
		if e != nil {
			return FlowDualTrainingResult{}, nil, e
		}
		for i := range seed {
			seed[i] *= diag
		}
		ref, e := reverseCPU(m, c, []float32{tm, tm}, r.DiagonalInput, -1, seed, make([]float32, n))
		return ref, seed, e
	}
	_, ep, e := m.forwardTraining(c, []float32{dt, dt}, r.EndpointInput)
	if e != nil {
		return FlowDualTrainingResult{}, nil, e
	}
	desired := make([]float32, n)
	for i := range desired {
		desired[i] = r.PrimaryVelocity[i] + float32((dt-s)*r.PrimaryTangent[i])
	}
	dr, e := reverseLossSeed(desired, ep)
	if e != nil {
		return FlowDualTrainingResult{}, nil, e
	}
	neg := make([]float32, n)
	for i := range neg {
		neg[i] = -dr[i]
	}
	endpoint, e := reverseCPU(m, c, []float32{dt, dt}, r.EndpointInput, -1, neg, make([]float32, n))
	if e != nil {
		return FlowDualTrainingResult{}, nil, e
	}
	gv, gt, e := reversePrimarySeeds(dr, endpoint.DInput, dt-s)
	if e != nil {
		return FlowDualTrainingResult{}, nil, e
	}
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(s*target[i]) + float32((1-s)*noise[i])
	}
	_, scale, e := productionObjectiveScales(13)
	if e != nil {
		return FlowDualTrainingResult{}, nil, e
	}
	for i := range gv {
		gv[i] *= scale
		gt[i] *= scale
	}
	ref, e := reverseCPU(m, c, []float32{s, dt}, xs, 1, gv, gt)
	return ref, nil, e
}
func parameterAccumulateCPU(previous map[string][]float32, source map[string][]float32, scale float32) error {
	if !isFinite(scale) || scale < 0 {
		return fmt.Errorf("CPU accumulation scale")
	}
	for name, x := range source {
		p, ok := previous[name]
		if !ok {
			p = make([]float32, len(x))
			previous[name] = p
		}
		if len(p) != len(x) {
			return fmt.Errorf("CPU accumulation extent")
		}
		for i, v := range x {
			p[i] = p[i] + float32(v*scale)
			if !isFinite(p[i]) {
				return fmt.Errorf("CPU accumulation overflow")
			}
		}
	}
	return nil
}
func TestPocketParameterCaptureTiny(t *testing.T) {
	flow := tinyTrainableFlowHead()
	c, target, noise := []float32{.2, -.3, .5}, []float32{.3, -.2}, []float32{-.1, .4}
	tm, s, dt := float32(.7), float32(.3), float32(.8)
	xs := make([]float32, 2)
	di := make([]float32, 2)
	for i := range xs {
		xs[i] = s*target[i] + (1-s)*noise[i]
		di[i] = tm*target[i] + (1-tm)*noise[i]
	}
	_, dp, e := flow.forwardTraining(c, []float32{tm, tm}, di)
	if e != nil {
		t.Fatal(e)
	}
	v, j, e := flow.ForwardTimeJVP(c, []float32{s, dt}, xs, 1)
	if e != nil {
		t.Fatal(e)
	}
	xt := make([]float32, 2)
	for i := range xt {
		xt[i] = xs[i] + (dt-s)*v[i]
	}
	_, ep, e := flow.forwardTraining(c, []float32{dt, dt}, xt)
	if e != nil {
		t.Fatal(e)
	}
	r := parameterRow{0, di, dp, v, j, xt, ep}
	diag, dist, e := productionObjectiveScales(13)
	if e != nil {
		t.Fatal(e)
	}
	inv := float32(1) / float32(13)
	if diag == float32(.75/13) || diag != float32(.75)*inv {
		t.Fatal("production scale expression", diag)
	}
	previous := map[string][]float32(nil)
	cpu := map[string][]float32{}
	for _, d := range []bool{false, true} {
		var g train.Tape
		var names []string
		if d {
			g, names, e = parameterDistill(flow, c, target, noise, s, dt, r, previous, dist)
		} else {
			g, names, e = parameterDiagonal(flow, c, target, noise, tm, r, previous, diag)
		}
		if e != nil {
			t.Fatal(e)
		}
		values := interpretFlowTape(t, g)
		boundary := 3
		if d {
			boundary = 5
		}
		next := map[string][]float32{}
		for i, name := range names {
			next[name] = values[boundary+i]
		}
		ref, _, e := parameterCPUReference(flow, c, target, noise, tm, s, dt, r, d)
		if e != nil {
			t.Fatal(e)
		}
		scale := float32(1)
		if e = parameterAccumulateCPU(cpu, ref.Parameters, scale); e != nil {
			t.Fatal(e)
		}
		for name, x := range next {
			assertSliceClose(t, "parameter tiny "+name, x, cpu[name], 1e-5)
		}
		previous = next
	}
	if _, _, e = parameterGradientGraph(train.Tape{}, nil, 0, nil, 1); e == nil {
		t.Fatal("boundary")
	}
	if _, _, e = productionObjectiveScales(0); e == nil {
		t.Fatal("scale input")
	}
	if e = parameterAccumulateCPU(map[string][]float32{"a": {1}}, map[string][]float32{"a": {1, 2}}, 1); e == nil {
		t.Fatal("accumulator extent")
	}
}
func parameterAdmission(t *testing.T, in parameterInputs) {
	t.Helper()
	b := in.Base
	diag, dist, _ := productionObjectiveScales(b.Valid)
	r := in.Rows[0]
	c, target, noise := b.Conditions[:1024], b.Batch.NormalizedLatents[:32], b.Samples.Noise[:32]
	for _, d := range []bool{false, true} {
		var tape train.Tape
		var names []string
		var e error
		if d {
			tape, names, e = parameterDistill(b.Flow, c, target, noise, b.Samples.DistillS[0], b.Samples.DistillT[0], r, nil, dist)
		} else {
			tape, names, e = parameterDiagonal(b.Flow, c, target, noise, b.Samples.DiagonalTime[0], r, nil, diag)
		}
		if e != nil {
			t.Fatal(e)
		}
		var bytes int64
		for _, s := range tape.Tensors {
			n, _ := s.Elements()
			bytes += int64(n) * 4
		}
		t.Logf("distill=%t source=%d params=%d commands=%d", d, bytes, len(names), len(tape.Commands))
	}
}
func TestPocketParameterCapturePreparation(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_PARAMETER_PREPARE") != "1" {
		t.Skip("new missing parameter source admission")
	}
	in := parameterPrepare(t)
	parameterAdmission(t, in)
}
func TestPocketNativeParameterCapture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_PARAMETER_NATIVE") != "1" {
		t.Skip("explicit bounded missing parameter capture")
	}
	in := parameterPrepare(t)
	parameterAdmission(t, in)
	b := in.Base
	diagScale, distScale, e := productionObjectiveScales(b.Valid)
	if e != nil {
		t.Fatal(e)
	}
	if os.Getenv("GO_PHERENCE_DISABLE_VULKAN") != "" || os.Getenv("VK_DRIVER_FILES") != "/usr/share/vulkan/icd.d/nvidia_icd.json" {
		t.Fatal("explicit Vulkan")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Second)
	defer cancel()
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("runtime mismatch no fallback")
	}
	if e = vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 2 << 30, MaxAllocations: 4096}); e != nil {
		t.Fatal(e)
	}
	op, e := vulkan.NewVkTrainingF32(ctx)
	if e != nil {
		t.Fatal(e)
	}
	type phase struct {
		Name   string
		Row    int
		Native vulkan.VulkanMemoryUsage
		RSS    uint64
	}
	var phases []phase
	run := func(name string, row int, tape train.Tape) [][]float32 {
		t.Helper()
		if e := tape.Validate(2 << 30); e != nil {
			t.Fatal(e)
		}
		if rss, e := flowCaptureRSS(); e != nil || rss > 6<<30 {
			t.Fatal("host cap", rss, e)
		}
		owner, e := vulkan.NewVkTrainingTape(ctx, op, tape, 2<<30)
		if e != nil {
			t.Fatal("native construction no retry", e)
		}
		usage := vulkan.VulkanMemoryStats()
		rss, e := flowCaptureRSS()
		if e != nil || rss > 6<<30 {
			t.Fatal("RSS cap", rss, e)
		}
		v, e := owner.Run(ctx)
		if e != nil {
			t.Fatal("native failure retain no retry", e)
		}
		if e = owner.Close(); e != nil {
			t.Fatal("close failure no retry", e)
		}
		u := vulkan.VulkanMemoryStats()
		if u.Bytes != 0 || u.Allocations != 0 || u.InFlight || u.Uncertain || u.DeviceLost {
			t.Fatal("closure", u)
		}
		phases = append(phases, phase{name, row, usage, rss})
		return v
	}
	current := map[string][]float32(nil)
	reference := map[string][]float32{}
	step := in.Step
	step.DistillObjectiveScaled = true
	for r := 0; r < b.Valid; r++ {
		c, target, noise := b.Conditions[r*1024:(r+1)*1024], b.Batch.NormalizedLatents[r*32:(r+1)*32], b.Samples.Noise[r*32:(r+1)*32]
		for _, d := range []bool{false, true} {
			var tape train.Tape
			var names []string
			var err error
			kind, boundary, scale := "diagonal", 3, float32(1)
			if d {
				kind, boundary, scale = "distill", 5, 1
				tape, names, err = parameterDistill(b.Flow, c, target, noise, b.Samples.DistillS[r], b.Samples.DistillT[r], in.Rows[r], current, distScale)
			} else {
				tape, names, err = parameterDiagonal(b.Flow, c, target, noise, b.Samples.DiagonalTime[r], in.Rows[r], current, diagScale)
			}
			if err != nil {
				t.Fatal(err)
			}
			values := run(kind, r, tape)
			next := map[string][]float32{}
			for i, name := range names {
				next[name] = values[boundary+i]
			}
			current = next
			ref, _, err := parameterCPUReference(b.Flow, c, target, noise, b.Samples.DiagonalTime[r], b.Samples.DistillS[r], b.Samples.DistillT[r], in.Rows[r], d)
			if err != nil {
				t.Fatal(err)
			}
			if err = parameterAccumulateCPU(reference, ref.Parameters, scale); err != nil {
				t.Fatal(err)
			}
			if d {
				copy(step.DistillCondition[r*1024:], values[0])
				copy(step.DistillTarget[r*32:], values[1])
				step.DistillLogvar[r] = values[4][2]
				step.RowMetrics[4*r+2], step.RowMetrics[4*r+3] = values[4][1], values[4][0]
			} else {
				copy(step.DiagonalCondition[r*1024:], values[0])
				copy(step.DiagonalTarget[r*32:], values[1])
				step.DiagonalLogvar[r] = values[2][2]
				step.RowMetrics[4*r], step.RowMetrics[4*r+1] = values[2][1], values[2][0]
			}
		}
	}
	root := os.Getenv("GO_PHERENCE_POCKET_PARAMETER_OUTPUT")
	type block struct {
		Name, NativeFile, ReferenceFile string
		Manifest, ReferenceManifest     []reverseManifest
		Error                           reverseError
	}
	var blocks []block
	save := func(name string, native, cpu map[string][]float32) {
		t.Helper()
		nf, cf := name+"-native.f32", name+"-reference.f32"
		m, d, err := reverseWriteGradients(filepath.Join(root, nf), native, cpu)
		if err != nil {
			t.Fatal(err)
		}
		cm, _, err := reverseWriteGradients(filepath.Join(root, cf), cpu, cpu)
		if err != nil {
			t.Fatal(err)
		}
		blocks = append(blocks, block{name, nf, cf, m, cm, d})
		t.Logf("%s count=%d max=%g RMS=%g opposite=%d diagnosticAbove4e-6=%d", name, d.Count, d.Max, d.RMS, d.OppositeSigns, d.Over4e6)
	}
	save("flow", current, reference)
	w := NewDefaultLSDWeightMLP()
	samples := WeightSampleInputs{b.Samples.DiagonalTime, b.Samples.DistillS, b.Samples.DistillT, step.DiagonalLogvar, step.DistillLogvar}
	wp, err := w.buildStepWeightTrainingTape(step, DefaultTrainingStepConfig(), samples, 2<<30, true)
	if err != nil {
		t.Fatal(err)
	}
	wv := run("weighting-reduction", -1, wp.Tape)
	names := make([]string, 0, len(wp.ResidentBindings))
	for name := range wp.ResidentBindings {
		names = append(names, name)
	}
	sort.Strings(names)
	wg := map[string][]float32{}
	for i, name := range names {
		wg[name] = wv[5+i]
	}
	packed, seed := make([]float32, 56), make([]float32, 28)
	for r := 0; r < 14; r++ {
		packed[2*r], packed[2*r+1] = b.Samples.DiagonalTime[r], b.Samples.DiagonalTime[r]
		packed[28+2*r], packed[29+2*r] = b.Samples.DistillS[r], b.Samples.DistillT[r]
		seed[r] = step.DiagonalLogvar[r] * diagScale
		seed[14+r] = step.DistillLogvar[r] * distScale
	}
	wt, _, err := w.forward(packed, 28)
	if err != nil {
		t.Fatal(err)
	}
	wc, err := w.backward(wt, seed, 28)
	if err != nil {
		t.Fatal(err)
	}
	wref := map[string][]float32{}
	for i, l := range wc.Layers {
		prefix := fmt.Sprintf("flow.w_s_t.%d", 2*i)
		wref[prefix+".weight"], wref[prefix+".bias"] = l.Weight, l.Bias
	}
	save("weighting", wg, wref)
	ep, err := b.LM.buildEOSProjectionTrainingTape(b.Conditions, b.Samples.Mask, .1, 2<<30, true)
	if err != nil {
		t.Fatal(err)
	}
	ev := run("eos-parameters", -1, ep.Tape)
	eg := map[string][]float32{"flow_lm.out_eos.weight": ev[2], "flow_lm.out_eos.bias": ev[3]}
	logits := make([]float32, 14)
	for r := range logits {
		logits[r] = linearForwardTraining(b.LM.EOS, b.Conditions[r*1024:(r+1)*1024])[0]
	}
	_, dy, err := EOSLossAndGradient(logits, b.Samples.Mask)
	if err != nil {
		t.Fatal(err)
	}
	ec := newLinearGradient(b.LM.EOS)
	for r := range dy {
		_ = linearBackwardTraining(b.LM.EOS, b.Conditions[r*1024:(r+1)*1024], []float32{dy[r] * .1}, &ec)
	}
	eref := map[string][]float32{"flow_lm.out_eos.weight": ec.Weight, "flow_lm.out_eos.bias": ec.Bias}
	save("eos", eg, eref)
	g := &flowTapeGraph{}
	a, eo := g.constant(wv[0]), g.constant(ev[1])
	sum := g.command("vector", []int{a, eo}, []int{14336}, 0, 0, 0, 1, 0)
	pad := g.command("vector", []int{sum, g.zero(1)}, []int{29696}, 0, 2, 15360, 0, 0)
	g.tape.Outputs = []int{sum, pad}
	joined := run("corrected-seed-join", -1, g.tape)
	report := struct {
		Scope, Device                                string
		DiagonalScale, DistillScale                  float32
		Blocks                                       []block
		CorrectedInput                               StepReductionInputs
		Reduction                                    [][]float32
		JoinedConditionSeed, TrunkSeed, EOSCondition []float32
		Phases                                       []phase
		Final                                        vulkan.VulkanMemoryUsage
	}{Scope: "Allrowmissingnamedparametercapture/nativeF32boundaryaccumulation;productionobjectivescaleexpressioncorrected. No residentoptimizer/globalclip/update/fullstepacceptance;referencesatsharednativeconditions.", Device: vulkan.VulkanDeviceName(), DiagonalScale: diagScale, DistillScale: distScale, Blocks: blocks, CorrectedInput: step, Reduction: wv[:5], JoinedConditionSeed: joined[0], TrunkSeed: joined[1], EOSCondition: ev[1], Phases: phases}
	if err = op.Close(); err != nil {
		t.Fatal(err)
	}
	report.Final = vulkan.VulkanMemoryStats()
	if report.Final.Bytes != 0 || report.Final.Allocations != 0 || report.Final.InFlight || report.Final.Uncertain || report.Final.DeviceLost {
		t.Fatal("final closure", report.Final)
	}
	if err = flowCaptureWrite(filepath.Join(root, "native-parameters.json"), report); err != nil {
		t.Fatal(err)
	}
	runtime.KeepAlive(in)
}
