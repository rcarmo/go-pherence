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
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type joinedCaptureInputs struct {
	LM                   *FlowLMTrainingCPU
	Batch                FlowLMTrainingBatch
	Seeds, DirectTargets map[string][]float32
}

func joinedCapturePrepare(t *testing.T) joinedCaptureInputs {
	t.Helper()
	var captured struct {
		TrunkSeed, CPUJoinedConditionSeed, CPUTarget []float32
		Reduced                                      [][]float32
	}
	if e := flowCapturePinned("/workspace/exports/pocket-native-allrow-20261004/native-allrow.json", "aa7e79fdf84b727936a462e7d69cbdb31a6bf153d0183f8adbb2eccd2251d6b6", 8<<20, &captured); e != nil {
		t.Fatal(e)
	}
	var corpus struct{ Rows []PreparedTrainingRow }
	if e := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9", 1<<20, &corpus); e != nil {
		t.Fatal(e)
	}
	if len(corpus.Rows) != 1 || len(captured.TrunkSeed) != 29696 || len(captured.CPUJoinedConditionSeed) != 14336 || len(captured.CPUTarget) != 448 || len(captured.Reduced) != 5 || len(captured.Reduced[1]) != 448 || !finiteF32(captured.TrunkSeed) || !finiteF32(captured.CPUJoinedConditionSeed) || !finiteF32(captured.CPUTarget) || !finiteF32(captured.Reduced[1]) || !zeroTrainingSeed(captured.TrunkSeed[:15360]) {
		t.Fatal("joined seed fixture")
	}
	b := corpus.Rows[0].Batch
	if b.Frames != 14 || b.VoiceFrames != 13 || len(b.TextTokens) != 1 {
		t.Fatal("joined geometry")
	}
	path := releasedAsset(t, "GO_PHERENCE_POCKETTTS_VOICE_MODEL", "fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997", 219029196)
	f, e := safetensors.Open(path)
	if e != nil {
		t.Fatal(e)
	}
	lm, e := LoadFlowLMTrainingCPU(f, releasedConfig(t))
	ce := f.Close()
	if e != nil || ce != nil {
		t.Fatal(e, ce)
	}
	cpuSeed := make([]float32, 29696)
	copy(cpuSeed[15360:], captured.CPUJoinedConditionSeed)
	return joinedCaptureInputs{lm, b, map[string][]float32{"native-seed": captured.TrunkSeed, "reference-seed": cpuSeed}, map[string][]float32{"native-seed": captured.Reduced[1], "reference-seed": captured.CPUTarget}}
}
func joinedCaptureBuild(lm *FlowLMTrainingCPU, b FlowLMTrainingBatch, seed []float32) (BackboneTrainingTape, int64, int64, error) {
	p, e := lm.buildBackboneTrainingTape(b, seed, 2<<30, true)
	if e != nil {
		return p, 0, 0, e
	}
	if e = train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 2<<30); e != nil {
		return p, 0, 0, e
	}
	var source, payload int64
	aliases := map[int]bool{}
	for _, c := range p.Tape.Commands {
		if c.Op == "reshape" {
			aliases[c.Output] = true
		}
	}
	for id, s := range p.Tape.Tensors {
		n, e := s.Elements()
		if e != nil {
			return p, 0, 0, e
		}
		source += int64(n) * 4
		if !aliases[id] {
			payload += int64(n) * 4
		}
	}
	if payload > 2<<30 {
		return p, source, payload, fmt.Errorf("joined payload")
	}
	return p, source, payload, nil
}
func joinedCaptureCPU(lm *FlowLMTrainingCPU, b FlowLMTrainingBatch, seed []float32) (FlowLMTrainingOutput, map[string][]float32, FlowLMTrainingInputGradients, error) {
	prefix := 1 + b.VoiceFrames + len(b.TextTokens)
	if len(seed) != (prefix+b.Frames)*lm.Hidden || !zeroTrainingSeed(seed[:prefix*lm.Hidden]) {
		return FlowLMTrainingOutput{}, nil, FlowLMTrainingInputGradients{}, fmt.Errorf("joined CPU seed")
	}
	out, g, inputs, e := lm.ForwardBackward(b, seed[prefix*lm.Hidden:], make([]float32, b.Frames))
	if e != nil {
		return out, nil, inputs, e
	}
	named := map[string][]float32{}
	addFlowLMGradients(named, g)
	delete(named, "flow_lm.out_eos.weight")
	delete(named, "flow_lm.out_eos.bias")
	return out, named, inputs, nil
}
func joinedCaptureTargetTape(backbone, direct []float32) (train.Tape, error) {
	if len(backbone) == 0 || len(backbone) != len(direct) || !finiteF32(backbone) || !finiteF32(direct) {
		return train.Tape{}, fmt.Errorf("joined target extent")
	}
	g := &flowTapeGraph{}
	a, b := g.constant(backbone), g.constant(direct)
	id := g.command("vector", []int{a, b}, []int{len(direct)}, 0, 0, 0, 1, 0)
	g.tape.Outputs = []int{id}
	return g.tape, g.tape.Validate(2 << 30)
}
func TestPocketJoinedBackboneCaptureTiny(t *testing.T) {
	lm := tinyFlowLMTraining()
	b := FlowLMTrainingBatch{Frames: 3, VoiceFrames: 2, TextTokens: []uint32{1}, NormalizedLatents: make([]float32, 3*lm.LatentDim), VoiceLatents: make([]float32, 2*lm.LatentDim)}
	prefix := 1 + b.VoiceFrames + len(b.TextTokens)
	seed := make([]float32, (prefix+b.Frames)*lm.Hidden)
	for i := prefix * lm.Hidden; i < len(seed); i++ {
		seed[i] = float32(i%5-2) * .01
	}
	p, s, n, e := joinedCaptureBuild(lm, b, seed)
	if e != nil || s < n || n <= 0 {
		t.Fatal(s, n, e)
	}
	v := interpretFlowTape(t, p.Tape)
	out, w, inputs, e := joinedCaptureCPU(lm, b, seed)
	if e != nil {
		t.Fatal(e)
	}
	assertSliceClose(t, "joined tiny seq", v[0], out.Sequence, 4e-6)
	assertSliceClose(t, "joined tiny voice", v[2], inputs.VoiceLatents, 8e-6)
	assertSliceClose(t, "joined tiny audio", v[3], inputs.NormalizedLatents, 8e-6)
	names := make([]string, 0, len(p.ResidentBindings))
	for name := range p.ResidentBindings {
		names = append(names, name)
	}
	sort.Strings(names)
	for i, name := range names {
		assertSliceClose(t, "joined tiny "+name, v[4+i], w[name], 1e-5)
	}
	direct := make([]float32, len(v[3]))
	for i := range direct {
		direct[i] = .1
	}
	tape, e := joinedCaptureTargetTape(v[3], direct)
	if e != nil {
		t.Fatal(e)
	}
	joined := interpretFlowTape(t, tape)[0]
	for i, x := range joined {
		if x != v[3][i]+direct[i] {
			t.Fatal("target added more than once")
		}
	}
	seed[0] = 1
	if _, _, _, e := joinedCaptureCPU(lm, b, seed); e == nil {
		t.Fatal("prefix")
	}
	if _, e := joinedCaptureTargetTape(nil, direct); e == nil {
		t.Fatal("target shape")
	}
}
func TestPocketJoinedBackbonePreparation(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_JOINED_PREPARE") != "1" {
		t.Skip("joined complete source admission")
	}
	in := joinedCapturePrepare(t)
	type row struct {
		Mode                 string
		Source, Payload      int64
		Parameters, Elements int
	}
	var rows []row
	for _, mode := range []string{"native-seed", "reference-seed"} {
		p, s, n, e := joinedCaptureBuild(in.LM, in.Batch, in.Seeds[mode])
		if e != nil {
			t.Fatal(e)
		}
		count := 0
		for _, b := range p.ResidentBindings {
			count += len(p.Tape.Tensors[b.Parameter].Values)
		}
		rows = append(rows, row{mode, s, n, len(p.ResidentBindings), count})
		t.Logf("%s source=%d nativePayload=%d parameters=%d elements=%d", mode, s, n, len(p.ResidentBindings), count)
	}
	if e := flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_JOINED_OUTPUT"), "preparation.json"), rows); e != nil {
		t.Fatal(e)
	}
}
func TestPocketNativeJoinedBackboneCapture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_JOINED_NATIVE") != "1" {
		t.Skip("explicit no-update complete backbone VJP")
	}
	in := joinedCapturePrepare(t)
	graphs := map[string]BackboneTrainingTape{}
	for _, mode := range []string{"native-seed", "reference-seed"} {
		p, _, _, e := joinedCaptureBuild(in.LM, in.Batch, in.Seeds[mode])
		if e != nil {
			t.Fatal(e)
		}
		graphs[mode] = p
	}
	if os.Getenv("GO_PHERENCE_DISABLE_VULKAN") != "" || os.Getenv("VK_DRIVER_FILES") != "/usr/share/vulkan/icd.d/nvidia_icd.json" {
		t.Fatal("explicit Vulkan")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
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
	type result struct {
		Mode, GradientFile                                                                       string
		Seed, Sequence, Output, DVoice, DNormalized, DirectTarget, JoinedTarget, CPUJoinedTarget []float32
		SequenceError, ConditionError, VoiceError, TargetError, JoinedTargetError, GradientError reverseError
		Manifest                                                                                 []reverseManifest
		Native                                                                                   vulkan.VulkanMemoryUsage
		RSS                                                                                      uint64
	}
	report := struct {
		Scope, Device string
		Results       []result
		Final         vulkan.VulkanMemoryUsage
	}{Scope: "Complete nativeconditioner+trunkVJPatactualallrowflow/EOSjoinedseed;reference-seeddiagnosticonly. Directtargetaddedonce. Excludesflow/weight/EOSparametercontributions/globalnorm/optimizer/Pupdate/fullstepqualification.", Device: vulkan.VulkanDeviceName()}
	for _, mode := range []string{"native-seed", "reference-seed"} {
		p := graphs[mode]
		out, allRef, inputs, e := joinedCaptureCPU(in.LM, in.Batch, in.Seeds[mode])
		if e != nil {
			t.Fatal(e)
		}
		reference := map[string][]float32{}
		for name := range p.ResidentBindings {
			reference[name] = allRef[name]
		}
		if rss, e := flowCaptureRSS(); e != nil || rss > 6<<30 {
			t.Fatal("hostcap", rss, e)
		}
		tape, e := vulkan.NewVkTrainingTape(ctx, op, p.Tape, 2<<30)
		if e != nil {
			t.Fatal("nativeconstruction failure no retry", e)
		}
		usage := vulkan.VulkanMemoryStats()
		rss, e := flowCaptureRSS()
		if e != nil || rss > 6<<30 {
			t.Fatal("RSScap", rss, e)
		}
		values, e := tape.Run(ctx)
		if e != nil {
			t.Fatal("nativefailure retain no retry", e)
		}
		names := make([]string, 0, len(p.ResidentBindings))
		for name := range p.ResidentBindings {
			names = append(names, name)
		}
		sort.Strings(names)
		if len(values) != 4+len(names) {
			t.Fatal("joined output count")
		}
		native := map[string][]float32{}
		for i, name := range names {
			native[name] = values[4+i]
		}
		r := result{Mode: mode, GradientFile: mode + "-backbone.f32", Seed: in.Seeds[mode], Sequence: values[0], Output: values[1], DVoice: values[2], DNormalized: values[3], DirectTarget: in.DirectTargets[mode], Native: usage, RSS: rss}
		r.SequenceError, e = reverseCompare(values[0], out.Sequence)
		if e != nil {
			t.Fatal(e)
		}
		r.ConditionError, e = reverseCompare(values[1][15*1024:], out.Z)
		if e != nil {
			t.Fatal(e)
		}
		r.VoiceError, e = reverseCompare(values[2], inputs.VoiceLatents)
		if e != nil {
			t.Fatal(e)
		}
		r.TargetError, e = reverseCompare(values[3], inputs.NormalizedLatents)
		if e != nil {
			t.Fatal(e)
		}
		r.Manifest, r.GradientError, e = reverseWriteGradients(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_JOINED_OUTPUT"), r.GradientFile), native, reference)
		if e != nil {
			t.Fatal(e)
		}
		if e = tape.Close(); e != nil {
			t.Fatal("close failure no retry", e)
		}
		if u := vulkan.VulkanMemoryStats(); u.Bytes != 0 || u.Allocations != 0 || u.InFlight || u.Uncertain || u.DeviceLost {
			t.Fatal("closure", u)
		}
		targetTape, e := joinedCaptureTargetTape(values[3], in.DirectTargets[mode])
		if e != nil {
			t.Fatal(e)
		}
		targetOwner, e := vulkan.NewVkTrainingTape(ctx, op, targetTape, 2<<30)
		if e != nil {
			t.Fatal(e)
		}
		joined, e := targetOwner.Run(ctx)
		if e != nil {
			t.Fatal("join failure no retry", e)
		}
		r.JoinedTarget = joined[0]
		r.CPUJoinedTarget = make([]float32, len(inputs.NormalizedLatents))
		for i, v := range inputs.NormalizedLatents {
			r.CPUJoinedTarget[i] = v + in.DirectTargets[mode][i]
		}
		r.JoinedTargetError, e = reverseCompare(r.JoinedTarget, r.CPUJoinedTarget)
		if e != nil {
			t.Fatal(e)
		}
		if e = targetOwner.Close(); e != nil {
			t.Fatal(e)
		}
		if u := vulkan.VulkanMemoryStats(); u.Bytes != 0 || u.Allocations != 0 {
			t.Fatal("join closure", u)
		}
		report.Results = append(report.Results, r)
		t.Logf("%s gradientvalues=%d max=%g RMS=%g signs=%d diagnosticAbove4e-6=%d voice=%g normalized=%g joined=%g native=%d RSS=%d", mode, r.GradientError.Count, r.GradientError.Max, r.GradientError.RMS, r.GradientError.OppositeSigns, r.GradientError.Over4e6, r.VoiceError.Max, r.TargetError.Max, r.JoinedTargetError.Max, usage.Bytes, rss)
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	report.Final = vulkan.VulkanMemoryStats()
	if e = flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_JOINED_OUTPUT"), "native-joined-backbone.json"), report); e != nil {
		t.Fatal(e)
	}
	runtime.KeepAlive(in)
}
