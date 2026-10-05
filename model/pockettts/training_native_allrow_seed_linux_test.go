//go:build linux

package pockettts

import (
	"context"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

type allrowInputs struct {
	LM         *FlowLMTrainingCPU
	Flow       *FlowHeadCPU
	Batch      FlowLMTrainingBatch
	Samples    TrainingStepSamples
	Conditions []float32
	Valid      int
}

func allrowPrepare(t *testing.T) allrowInputs {
	t.Helper()
	var conditions struct {
		Conditions             []float32
		Frames, Hidden, Prefix int
	}
	path, pin := "/workspace/exports/pocket-native-conditioner-20261004/native-conditioner.json", "633428297efdf349bd136e17262d0ee253999269d6fb3f65f78f6e805537f722"
	if p := os.Getenv("GO_PHERENCE_POCKET_CONDITIONER_INPUT"); p != "" {
		path, pin = p, os.Getenv("GO_PHERENCE_POCKET_CONDITIONER_SHA256")
	}
	if e := flowCapturePinned(path, pin, 4<<20, &conditions); e != nil {
		t.Fatal(e)
	}
	var corpus struct{ Rows []PreparedTrainingRow }
	if e := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9", 1<<20, &corpus); e != nil {
		t.Fatal(e)
	}
	var saved struct {
		Identity TrainingRunIdentity
		TimeRNG  string `json:"time_rng"`
		NoiseRNG string `json:"noise_rng"`
	}
	if e := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CPU_RUN"), "e5c1bd21309fb893336028fe363cc192b88030727a2b310f3b0be869c1e8fdbd", 1<<16, &saved); e != nil {
		t.Fatal(e)
	}
	if len(corpus.Rows) != 1 || conditions.Frames != 14 || conditions.Hidden != 1024 || conditions.Prefix != 15 || len(conditions.Conditions) != 14336 || !finiteF32(conditions.Conditions) || saved.Identity.Backend != "simd" || saved.Identity.GoVersion != runtime.Version() || saved.Identity.GOARCH != runtime.GOARCH || saved.Identity.GOOS != runtime.GOOS {
		t.Fatal("allrow identity")
	}
	row := corpus.Rows[0]
	if row.Batch.Frames != 14 || row.Batch.VoiceFrames != 13 || row.ValidFrames != 13 || len(row.Batch.TextTokens) != 1 {
		t.Fatal("allrow geometry")
	}
	run, e := NewTrainingRunState(saved.Identity, 1, [6]uint64{42, 43, 44, 45, 46, 47})
	if e != nil {
		t.Fatal(e)
	}
	p, e := run.PrepareRow()
	if e != nil {
		t.Fatal(e)
	}
	samples := runnerSampleBuffers(14, 32)
	recipe := TrainingRandomRecipe{Algorithm: TrainingRandomAlgorithm, OptionsSHA256: saved.Identity.OptionsSHA256, DistillProbability: 1, FlowBatchMultiplier: 1, StopGradient: "minimal", Normalize: true, PEqual: .75, EOSLossWeight: .1, LogitMean: .4, LogitStd: 1}
	if e = SampleTrainingInto(samples, p, 14, 13, 32, recipe, TrainingRandomLimits{TrainingSampleLimits{375, 375 * 32}, 2*375*32 + 6*375}); e != nil {
		t.Fatal(e)
	}
	if hex.EncodeToString(p.next.timeRNG[:]) != saved.TimeRNG || hex.EncodeToString(p.next.noiseRNG[:]) != saved.NoiseRNG {
		t.Fatal("allrow replay")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	for _, l := range w.Layers {
		for _, v := range l.Weight {
			if v != 0 {
				t.Fatal("nonzero initial logvar")
			}
		}
		for _, v := range l.Bias {
			if v != 0 {
				t.Fatal("nonzero initial logvar bias")
			}
		}
	}
	return allrowInputs{lm, flow, row.Batch, samples, conditions.Conditions, 13}
}

type allrowGraph struct {
	Tape            train.Tape
	Source, Payload int64
	Stages          int
}

func allrowBoundary(tape train.Tape, bindings map[string]train.ResidentTapeBinding, count int) (allrowGraph, error) {
	var out allrowGraph
	if count < 1 || count > len(tape.Outputs) {
		return out, fmt.Errorf("allrow boundary count")
	}
	var e error
	if bindings != nil {
		e = train.ValidateResidentTapeBindings(tape, bindings, 2<<30)
	} else {
		e = tape.Validate(2 << 30)
	}
	if e != nil {
		return out, e
	}
	for _, s := range tape.Tensors {
		n, e := s.Elements()
		if e != nil {
			return out, e
		}
		out.Source += int64(n) * 4
	}
	out.Tape, _, e = train.PruneTape(tape, append([]int(nil), tape.Outputs[:count]...), 2<<30)
	if e != nil {
		return out, e
	}
	alias := map[int]bool{}
	for _, c := range out.Tape.Commands {
		if c.Op == "reshape" {
			alias[c.Output] = true
		} else {
			out.Stages++
		}
	}
	for id, s := range out.Tape.Tensors {
		if alias[id] {
			continue
		}
		n, e := s.Elements()
		if e != nil {
			return out, e
		}
		out.Payload += int64(n) * 4
	}
	if out.Payload > 2<<30 {
		return out, fmt.Errorf("allrow payload")
	}
	return out, nil
}

// Complete existing fused row graphs, with all native boundary arithmetic.
func allrowDiagForward(m *FlowHeadCPU, in DiagonalForwardInputs) (allrowGraph, error) {
	p, e := m.buildDiagonalForwardTape(in, 2<<30, true)
	if e != nil {
		return allrowGraph{}, e
	}
	return allrowBoundary(p.Tape, p.ResidentBindings, 3)
}
func allrowDiagBackward(m *FlowHeadCPU, in DiagonalForwardInputs, v [][]float32, scale float32) (allrowGraph, error) {
	if len(v) != 3 {
		return allrowGraph{}, fmt.Errorf("diagonal forward extent")
	}
	p, e := m.buildDiagonalTrainingTape(in.Condition, in.Time, v[0], v[2], v[1], 0, scale, 2<<30, true)
	if e != nil {
		return allrowGraph{}, e
	}
	return allrowBoundary(p.Tape, p.ResidentBindings, 3)
}
func allrowDistForward(m *FlowHeadCPU, in DistillForwardInputs) (allrowGraph, error) {
	p, e := m.buildDistillForwardTape(in, 2<<30, true)
	if e != nil {
		return allrowGraph{}, e
	}
	return allrowBoundary(p.Tape, p.ResidentBindings, 6)
}
func allrowEndpoint(m *FlowHeadCPU, in DistillForwardInputs, v [][]float32) (allrowGraph, error) {
	if len(v) != 6 {
		return allrowGraph{}, fmt.Errorf("distill boundary extent")
	}
	p, e := m.buildEndpointTrainingTape(in.Condition, in.T, v[3], v[5], v[4], v[1], v[2], 0, in.T-in.S, 2<<30, true)
	if e != nil {
		return allrowGraph{}, e
	}
	return allrowBoundary(p.Tape, p.ResidentBindings, 5)
}
func allrowPrimary(m *FlowHeadCPU, in DistillForwardInputs, f, e [][]float32) (allrowGraph, error) {
	if len(f) != 6 || len(e) != 5 || len(e[4]) != 1 {
		return allrowGraph{}, fmt.Errorf("endpoint boundary extent")
	}
	p, err := m.buildPrimaryTrainingTape(PrimaryTrainingInputs{Condition: in.Condition, Input: f[0], GV: e[1], GT: e[2], EndpointInput: e[0], Target: in.Target, Noise: in.Noise, S: in.S, T: in.T, DDelta: e[4][0]}, 2<<30, true)
	if err != nil {
		return allrowGraph{}, err
	}
	return allrowBoundary(p.Tape, p.ResidentBindings, 4)
}
func allrowStepInput(in allrowInputs) StepReductionInputs {
	b := in.Batch
	return StepReductionInputs{Frames: b.Frames, Hidden: in.LM.Hidden, Latent: in.LM.LatentDim, Mask: append([]bool(nil), in.Samples.Mask...), DiagonalCondition: make([]float32, b.Frames*in.LM.Hidden), DistillCondition: make([]float32, b.Frames*in.LM.Hidden), DiagonalTarget: make([]float32, b.Frames*in.LM.LatentDim), DistillTarget: make([]float32, b.Frames*in.LM.LatentDim), DiagonalLogvar: make([]float32, b.Frames), DistillLogvar: make([]float32, b.Frames), RowMetrics: make([]float32, b.Frames*4)}
}
func allrowTinyReference(t *testing.T, flow *FlowHeadCPU, di DiagonalForwardInputs, ds DistillForwardInputs) {
	t.Helper()
	run := func(g allrowGraph, e error) [][]float32 {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
		return interpretFlowTape(t, g.Tape)
	}
	df := run(allrowDiagForward(flow, di))
	db := run(allrowDiagBackward(flow, di, df, .25))
	wantSeed, e := reverseLossSeed(df[2], df[1])
	if e != nil {
		t.Fatal(e)
	}
	for i := range wantSeed {
		wantSeed[i] *= .25
	}
	ref, e := reverseCPU(flow, di.Condition, []float32{di.Time, di.Time}, df[0], -1, wantSeed, make([]float32, len(wantSeed)))
	if e != nil {
		t.Fatal(e)
	}
	assertSliceClose(t, "allrow tiny diagonal", db[0], ref.DCondition, 1e-5)
	f := run(allrowDistForward(flow, ds))
	ep := run(allrowEndpoint(flow, ds, f))
	pr := run(allrowPrimary(flow, ds, f, ep))
	_, g, e := flow.LSDDistillForwardBackward(ds.Condition, ds.S, ds.T, ds.Noise, ds.Target, 0)
	if e != nil {
		t.Fatal(e)
	}
	assertSliceClose(t, "allrow tiny condition", pr[0], g.Condition, 1e-5)
	assertSliceClose(t, "allrow tiny target", pr[1], g.Target, 1e-5)
	assertSliceClose(t, "allrow tiny times", pr[3], []float32{g.DS, g.DT}, 2e-5)
}
func TestPocketAllrowSeedGraphTiny(t *testing.T) {
	flow := tinyTrainableFlowHead()
	di := DiagonalForwardInputs{Condition: []float32{.2, -.3, .5}, Target: []float32{.3, -.2}, Noise: []float32{-.1, .4}, Time: .7}
	ds := DistillForwardInputs{Condition: di.Condition, Target: di.Target, Noise: di.Noise, S: .3, T: .8}
	allrowTinyReference(t, flow, di, ds)
	if _, e := allrowBoundary(train.Tape{}, nil, 1); e == nil {
		t.Fatal("bad boundary")
	}
	if _, e := allrowPrimary(flow, ds, nil, nil); e == nil {
		t.Fatal("bad primary boundary")
	}
	ds.S = .9
	if _, e := allrowDistForward(flow, ds); e == nil {
		t.Fatal("bad time")
	}
}
func allrowAdmission(t *testing.T, in allrowInputs) []allrowGraph {
	t.Helper()
	h, n := in.LM.Hidden, in.LM.LatentDim
	di := DiagonalForwardInputs{in.Conditions[:h], in.Batch.NormalizedLatents[:n], in.Samples.Noise[:n], in.Samples.DiagonalTime[0]}
	ds := DistillForwardInputs{in.Conditions[:h], di.Target, di.Noise, in.Samples.DistillS[0], in.Samples.DistillT[0]}
	var graphs []allrowGraph
	appendGraph := func(g allrowGraph, e error) {
		if e != nil {
			t.Fatal(e)
		}
		graphs = append(graphs, g)
	}
	appendGraph(allrowDiagForward(in.Flow, di))
	zero := make([]float32, n)
	appendGraph(allrowDiagBackward(in.Flow, di, [][]float32{zero, zero, zero}, float32(.75)/13))
	appendGraph(allrowDistForward(in.Flow, ds))
	six := make([][]float32, 6)
	for i := range six {
		six[i] = zero
	}
	appendGraph(allrowEndpoint(in.Flow, ds, six))
	five := [][]float32{zero, zero, zero, make([]float32, 3), {0}}
	appendGraph(allrowPrimary(in.Flow, ds, six, five))
	return graphs
}
func TestPocketAllrowSeedPreparation(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_ALLROW_PREPARE") != "1" {
		t.Skip("new allrow flow source admission")
	}
	in := allrowPrepare(t)
	g := allrowAdmission(t, in)
	type row struct {
		Source, Payload int64
		Stages          int
	}
	var rows []row
	for i, x := range g {
		rows = append(rows, row{x.Source, x.Payload, x.Stages})
		t.Logf("graph%d source=%d payload=%d stages=%d", i, x.Source, x.Payload, x.Stages)
	}
	if e := flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_ALLROW_OUTPUT"), "preparation.json"), rows); e != nil {
		t.Fatal(e)
	}
}

type allrowExecution struct {
	Kind            string
	Row             int
	Source, Payload int64
	Native          vulkan.VulkanMemoryUsage
	RSS             uint64
}

func TestPocketNativeAllrowSeedCapture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_ALLROW_NATIVE") != "1" {
		t.Skip("explicit bounded allrow flow/seed native capture")
	}
	in := allrowPrepare(t)
	_ = allrowAdmission(t, in)
	if os.Getenv("GO_PHERENCE_DISABLE_VULKAN") != "" || os.Getenv("VK_DRIVER_FILES") != "/usr/share/vulkan/icd.d/nvidia_icd.json" {
		t.Fatal("explicit Vulkan")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Second)
	defer cancel()
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("runtime mismatch no fallback")
	}
	if e := vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 2 << 30, MaxAllocations: 4096}); e != nil {
		t.Fatal(e)
	}
	op, e := vulkan.NewVkTrainingF32(ctx)
	if e != nil {
		t.Fatal(e)
	}
	var executions []allrowExecution
	run := func(kind string, row int, g allrowGraph, err error) [][]float32 {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		if rss, e := flowCaptureRSS(); e != nil || rss > 6<<30 {
			t.Fatal("hostcap", rss, e)
		}
		tape, e := vulkan.NewVkTrainingTape(ctx, op, g.Tape, 2<<30)
		if e != nil {
			t.Fatal("native construction failure no retry", e)
		}
		usage := vulkan.VulkanMemoryStats()
		rss, e := flowCaptureRSS()
		if e != nil || rss > 6<<30 {
			t.Fatal("RSScap", rss, e)
		}
		out, e := tape.Run(ctx)
		if e != nil {
			t.Fatal("nativefailure retain no retry", e)
		}
		if e = tape.Close(); e != nil {
			t.Fatal("closefailure no retry", e)
		}
		if u := vulkan.VulkanMemoryStats(); u.Bytes != 0 || u.Allocations != 0 || u.InFlight || u.Uncertain || u.DeviceLost {
			t.Fatal("closure", u)
		}
		executions = append(executions, allrowExecution{kind, row, g.Source, g.Payload, usage, rss})
		return out
	}
	exec := func(kind string, row int, graph func() (allrowGraph, error)) [][]float32 {
		g, e := graph()
		return run(kind, row, g, e)
	}
	nativeInput, cpuInput := allrowStepInput(in), allrowStepInput(in)
	h, n := in.LM.Hidden, in.LM.LatentDim
	type rowResult struct {
		Row                                                                                                   int
		DiagonalInput, DiagonalPrediction, PrimaryVelocity, PrimaryTangent, EndpointInput, EndpointPrediction []float32
		DiagonalCondition, PrimaryCondition, DiagonalTarget, PrimaryTarget                                    []float32
		DiagonalError, PrimaryError, DiagonalTargetError, PrimaryTargetError                                  reverseError
	}
	var rowResults []rowResult
	for r := 0; r < in.Valid; r++ {
		c := in.Conditions[r*h : (r+1)*h]
		target, noise := in.Batch.NormalizedLatents[r*n:(r+1)*n], in.Samples.Noise[r*n:(r+1)*n]
		di := DiagonalForwardInputs{c, target, noise, in.Samples.DiagonalTime[r]}
		ds := DistillForwardInputs{c, target, noise, in.Samples.DistillS[r], in.Samples.DistillT[r]}
		df := exec("diagonal-forward", r, func() (allrowGraph, error) { return allrowDiagForward(in.Flow, di) })
		db := exec("diagonal-backward", r, func() (allrowGraph, error) {
			return allrowDiagBackward(in.Flow, di, df, float32(.75)/float32(in.Valid))
		})
		f := exec("distill-forward", r, func() (allrowGraph, error) { return allrowDistForward(in.Flow, ds) })
		ep := exec("endpoint-vjp", r, func() (allrowGraph, error) { return allrowEndpoint(in.Flow, ds, f) })
		pr := exec("primary-vjp", r, func() (allrowGraph, error) { return allrowPrimary(in.Flow, ds, f, ep) })
		copy(nativeInput.DiagonalCondition[r*h:], db[0])
		copy(nativeInput.DiagonalTarget[r*n:], db[1])
		nativeInput.DiagonalLogvar[r] = db[2][2]
		nativeInput.RowMetrics[4*r], nativeInput.RowMetrics[4*r+1] = db[2][1], db[2][0]
		copy(nativeInput.DistillCondition[r*h:], pr[0])
		copy(nativeInput.DistillTarget[r*n:], pr[1])
		nativeInput.DistillLogvar[r] = ep[3][2]
		nativeInput.RowMetrics[4*r+2], nativeInput.RowMetrics[4*r+3] = ep[3][1], ep[3][0]
		// CPU reference uses identical actual native condition, independent rowflow.
		xt, desired := make([]float32, n), make([]float32, n)
		for i := range xt {
			xt[i] = di.Time*target[i] + (1-di.Time)*noise[i]
			desired[i] = target[i] - noise[i]
		}
		_, pred, e := in.Flow.forwardTraining(c, []float32{di.Time, di.Time}, xt)
		if e != nil {
			t.Fatal(e)
		}
		seed, e := reverseLossSeed(pred, desired)
		if e != nil {
			t.Fatal(e)
		}
		for i := range seed {
			seed[i] *= float32(.75) / float32(in.Valid)
		}
		ref, e := reverseCPU(in.Flow, c, []float32{di.Time, di.Time}, xt, -1, seed, make([]float32, n))
		if e != nil {
			t.Fatal(e)
		}
		copy(cpuInput.DiagonalCondition[r*h:], ref.DCondition)
		for i := 0; i < n; i++ {
			cpuInput.DiagonalTarget[r*n+i] = di.Time*ref.DInput[i] - seed[i]
		}
		var square float32
		for i := range pred {
			d := pred[i] - desired[i]
			square += d * d
		}
		cpuInput.DiagonalLogvar[r] = square/float32(n) - 1
		cpuInput.RowMetrics[4*r], cpuInput.RowMetrics[4*r+1] = square/float32(n), square
		loss, grad, e := in.Flow.LSDDistillForwardBackward(c, ds.S, ds.T, noise, target, 0)
		if e != nil {
			t.Fatal(e)
		}
		copy(cpuInput.DistillCondition[r*h:], grad.Condition)
		copy(cpuInput.DistillTarget[r*n:], grad.Target)
		cpuInput.DistillLogvar[r] = grad.DLogVariance
		cpuInput.RowMetrics[4*r+2], cpuInput.RowMetrics[4*r+3] = float32(loss.Loss), float32(loss.RawSquare)
		rr := rowResult{Row: r, DiagonalInput: df[0], DiagonalPrediction: df[2], PrimaryVelocity: f[1], PrimaryTangent: f[2], EndpointInput: f[3], EndpointPrediction: f[5], DiagonalCondition: db[0], PrimaryCondition: pr[0], DiagonalTarget: db[1], PrimaryTarget: pr[1]}
		rr.DiagonalError, e = reverseCompare(db[0], ref.DCondition)
		if e != nil {
			t.Fatal(e)
		}
		rr.PrimaryError, e = reverseCompare(pr[0], grad.Condition)
		if e != nil {
			t.Fatal(e)
		}
		rr.DiagonalTargetError, e = reverseCompare(db[1], cpuInput.DiagonalTarget[r*n:(r+1)*n])
		if e != nil {
			t.Fatal(e)
		}
		rr.PrimaryTargetError, e = reverseCompare(pr[1], grad.Target)
		if e != nil {
			t.Fatal(e)
		}
		rowResults = append(rowResults, rr)
	}
	eos, e := in.LM.buildEOSProjectionTrainingTape(in.Conditions, in.Samples.Mask, .1, 2<<30, true)
	if e != nil {
		t.Fatal(e)
	}
	eosGraph, e := allrowBoundary(eos.Tape, eos.ResidentBindings, 2)
	eosValues := run("eos-condition", -1, eosGraph, e)
	nativeInput.EOS = float64(eosValues[0][0])
	cpuInput.EOS = nativeInput.EOS
	cfg := DefaultTrainingStepConfig()
	reduce := func(kind string, input StepReductionInputs) [][]float32 {
		tape, e := BuildStepReductionTrainingTape(input, cfg, 2<<30)
		if e != nil {
			t.Fatal(e)
		}
		g, e := allrowBoundary(tape, nil, 5)
		return run(kind, -1, g, e)
	}
	nativeReduced := reduce("native-row-reduction", nativeInput)
	// CPU reduction is independent scalar formula, not a native self-reference.
	cpuDZ, cpuTarget := make([]float32, 14*h), make([]float32, 14*n)
	for i := range cpuDZ {
		cpuDZ[i] = cpuInput.DiagonalCondition[i] + cpuInput.DistillCondition[i]*(float32(.25)/float32(in.Valid))
	}
	for i := range cpuTarget {
		cpuTarget[i] = cpuInput.DiagonalTarget[i] + cpuInput.DistillTarget[i]*(float32(.25)/float32(in.Valid))
	}
	join := &flowTapeGraph{}
	a, b := join.constant(nativeReduced[0]), join.constant(eosValues[1])
	sum := join.command("vector", []int{a, b}, []int{14 * h}, 0, 0, 0, 1, 0)
	padded := join.command("vector", []int{sum, join.zero(1)}, []int{29 * h}, 0, 2, 15*h, 0, 0)
	join.tape.Outputs = []int{sum, padded}
	joinGraph, e := allrowBoundary(join.tape, nil, 2)
	joined := run("flow-plus-eos-pad", -1, joinGraph, e)
	cpuEOS, cpuEOSLoss, e := allrowCPUReferenceEOS(in.LM, in.Conditions, in.Samples.Mask)
	if e != nil {
		t.Fatal(e)
	}
	cpuJoined := make([]float32, len(cpuDZ))
	for i := range cpuJoined {
		cpuJoined[i] = cpuDZ[i] + cpuEOS[i]
	}
	if len(joined[1]) != 29*h || !zeroTrainingSeed(joined[1][:15*h]) {
		t.Fatal("joined seed prefix geometry")
	}
	type reportType struct {
		Scope, Device                                      string
		Conditions                                         []float32
		Rows                                               []rowResult
		Inputs, CPUInputs                                  StepReductionInputs
		Reduced                                            [][]float32
		CPUConditionSeed, CPUTarget                        []float32
		EOSConditionSeed, JoinedConditionSeed, TrunkSeed   []float32
		CPUEOSConditionSeed, CPUJoinedConditionSeed        []float32
		CPUEOSLoss                                         float64
		ConditionError, TargetError, EOSError, JoinedError reverseError
		Executions                                         []allrowExecution
		Final                                              vulkan.VulkanMemoryUsage
	}
	report := reportType{Scope: "13actualnativeconditionrows;fusednativeflow-primary-endpoint/loss/reverse boundaries andweightedallrowseeds/EOS. No parametergradaccumulation/optimizer/globalclip/update/fullstepacceptance.", Device: vulkan.VulkanDeviceName(), Conditions: in.Conditions, Rows: rowResults, Inputs: nativeInput, CPUInputs: cpuInput, Reduced: nativeReduced, CPUConditionSeed: cpuDZ, CPUTarget: cpuTarget, EOSConditionSeed: eosValues[1], JoinedConditionSeed: joined[0], TrunkSeed: joined[1], Executions: executions}
	report.ConditionError, e = reverseCompare(nativeReduced[0], cpuDZ)
	if e != nil {
		t.Fatal(e)
	}
	report.TargetError, e = reverseCompare(nativeReduced[1], cpuTarget)
	if e != nil {
		t.Fatal(e)
	}
	report.CPUEOSConditionSeed, report.CPUJoinedConditionSeed, report.CPUEOSLoss = cpuEOS, cpuJoined, cpuEOSLoss
	report.EOSError, e = reverseCompare(eosValues[1], cpuEOS)
	if e != nil {
		t.Fatal(e)
	}
	report.JoinedError, e = reverseCompare(joined[0], cpuJoined)
	if e != nil {
		t.Fatal(e)
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	report.Final = vulkan.VulkanMemoryStats()
	if report.Final.Bytes != 0 || report.Final.Allocations != 0 || report.Final.InFlight || report.Final.Uncertain || report.Final.DeviceLost {
		t.Fatal("final closure", report.Final)
	}
	if e = flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_ALLROW_OUTPUT"), "native-allrow.json"), report); e != nil {
		t.Fatal(e)
	}
	t.Logf("rows=%d graphs=%d conditionmax=%g RMS=%g signs=%d targetmax=%g", len(rowResults), len(executions), report.ConditionError.Max, report.ConditionError.RMS, report.ConditionError.OppositeSigns, report.TargetError.Max)
	runtime.KeepAlive(in)
}

// CPU EOS head and BCE derivatives are comparison-only, never native inputs.
func allrowCPUReferenceEOS(lm *FlowLMTrainingCPU, conditions []float32, mask []bool) ([]float32, float64, error) {
	if lm == nil || len(conditions) != len(mask)*lm.Hidden || !finiteF32(conditions) {
		return nil, 0, fmt.Errorf("CPU EOS condition shape")
	}
	logits := make([]float32, len(mask))
	for row := range mask {
		logits[row] = linearForwardTraining(lm.EOS, conditions[row*lm.Hidden:(row+1)*lm.Hidden])[0]
	}
	loss, seeds, err := EOSLossAndGradient(logits, mask)
	if err != nil {
		return nil, 0, err
	}
	out := make([]float32, len(conditions))
	for row, seed := range seeds {
		weighted := seed * .1
		for i, weight := range lm.EOS.Weight {
			out[row*lm.Hidden+i] = weighted * weight
		}
	}
	return out, loss, nil
}
func TestPocketAllrowEOSReductionTiny(t *testing.T) {
	lm := tinyFlowLMTraining()
	conditions := make([]float32, 3*lm.Hidden)
	for i := range conditions {
		conditions[i] = float32(i%5-2) * .1
	}
	mask := []bool{true, true, false}
	graph, err := lm.buildEOSProjectionTrainingTape(conditions, mask, .1, 64<<20, true)
	if err != nil {
		t.Fatal(err)
	}
	bound, err := allrowBoundary(graph.Tape, graph.ResidentBindings, 2)
	if err != nil {
		t.Fatal(err)
	}
	values := interpretFlowTape(t, bound.Tape)
	want, loss, err := allrowCPUReferenceEOS(lm, conditions, mask)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "allrow tiny EOS input", values[1], want, 4e-6)
	if math.Abs(float64(values[0][0])-loss) > 4e-6 {
		t.Fatal("EOS loss", values[0], loss)
	}
	in := StepReductionInputs{Frames: 3, Hidden: lm.Hidden, Latent: lm.LatentDim, Mask: mask, DiagonalCondition: make([]float32, 3*lm.Hidden), DistillCondition: make([]float32, 3*lm.Hidden), DiagonalTarget: make([]float32, 3*lm.LatentDim), DistillTarget: make([]float32, 3*lm.LatentDim), DiagonalLogvar: make([]float32, 3), DistillLogvar: make([]float32, 3), RowMetrics: make([]float32, 12), EOS: loss}
	for i := 0; i < 2*lm.Hidden; i++ {
		in.DiagonalCondition[i] = .2
		in.DistillCondition[i] = -.3
	}
	tape, err := BuildStepReductionTrainingTape(in, DefaultTrainingStepConfig(), 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	reduced := interpretFlowTape(t, tape)
	for i, v := range reduced[0] {
		expected := float32(0)
		if i < 2*lm.Hidden {
			expected = .2 + float32(-.3)*float32(.125)
		}
		if v != expected {
			t.Fatal("weight/padding", i, v, expected)
		}
	}
	in.DistillCondition[len(in.DistillCondition)-1] = 1
	if _, err := BuildStepReductionTrainingTape(in, DefaultTrainingStepConfig(), 64<<20); err == nil {
		t.Fatal("nonzero padding admitted")
	}
}
