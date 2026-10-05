//go:build linux

package pockettts

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Assemble only CPU conditioner inputs, without evaluating the CPU/native trunk.
func trunkCaptureSequence(m *FlowLMTrainingCPU, b FlowLMTrainingBatch) ([]float32, int, error) {
	rows, prefix, e := m.validateMode(b, make([]float32, b.Frames*m.Hidden), make([]float32, b.Frames), false)
	if e != nil {
		return nil, 0, e
	}
	x := make([]float32, rows*m.Hidden)
	copy(x, m.BOSBeforeVoice)
	for r := 0; r < b.VoiceFrames; r++ {
		copy(x[(1+r)*m.Hidden:], linearForwardTraining(m.SpeakerProjection, b.VoiceLatents[r*m.LatentDim:(r+1)*m.LatentDim]))
	}
	for r, token := range b.TextTokens {
		copy(x[(1+b.VoiceFrames+r)*m.Hidden:], m.Embedding[int(token)*m.Hidden:(int(token)+1)*m.Hidden])
	}
	for r := 0; r < b.Frames; r++ {
		audio := m.BOS
		if r > 0 {
			audio = b.NormalizedLatents[(r-1)*m.LatentDim : r*m.LatentDim]
		}
		copy(x[(prefix+r)*m.Hidden:], linearForwardTraining(m.Input, audio))
	}
	return x, prefix, nil
}
func trunkCaptureSeed(diagonal, primary []float32, rows, width, prefix, valid int) ([]float32, error) {
	if rows < 1 || width < 1 || prefix < 0 || prefix >= rows || valid < 1 || valid > rows-prefix || len(diagonal) != width || len(primary) != width || !finiteF32(diagonal) || !finiteF32(primary) {
		return nil, fmt.Errorf("trunk seed geometry")
	}
	out := make([]float32, rows*width)
	inv := float32(1) / float32(valid)
	ds, ps := float32(.75)*inv, float32(.25)*inv
	for i := range diagonal {
		out[prefix*width+i] = diagonal[i]*ds + primary[i]*ps
		if !isFinite(out[prefix*width+i]) {
			return nil, fmt.Errorf("trunk seed overflow")
		}
	}
	return out, nil
}

type trunkCaptureInputs struct {
	LM                     *FlowLMTrainingCPU
	Sequence, CPUCondition []float32
	Seeds                  map[string][]float32
	Rows, Prefix           int
}

func trunkCapturePrepare(t *testing.T) trunkCaptureInputs {
	t.Helper()
	var corpus struct{ Rows []PreparedTrainingRow }
	if e := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9", 1<<20, &corpus); e != nil {
		t.Fatal(e)
	}
	var captured struct {
		Results []struct {
			Name, Mode      string
			DCondition      []float32
			StoppedEndpoint bool
		}
	}
	if e := flowCapturePinned("/workspace/exports/pocket-flow-native-reverse-20261004/native-reverse.json", "231b9f3b981e4044fcc64adff0d25a3879db7a3e951cadc2fbda80b926282f7e", 1<<20, &captured); e != nil {
		t.Fatal(e)
	}
	var prior struct{ Condition []float32 }
	if e := flowCapturePinned("/workspace/exports/pocket-flow-native-boundary-20261003/preparation.json", "cce846c5fc433e70a1ccd39efd3d97c0215e804b8348778105d69b293ec21cde", 1<<20, &prior); e != nil {
		t.Fatal(e)
	}
	if len(corpus.Rows) != 1 || corpus.Rows[0].Batch.Frames != 14 || corpus.Rows[0].Batch.VoiceFrames != 13 || corpus.Rows[0].ValidFrames != 13 || len(corpus.Rows[0].Batch.TextTokens) != 1 || len(prior.Condition) != 1024 || !finiteF32(prior.Condition) {
		t.Fatal("trunk capture fixture")
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
	x, prefix, e := trunkCaptureSequence(lm, corpus.Rows[0].Batch)
	if e != nil {
		t.Fatal(e)
	}
	in := trunkCaptureInputs{lm, x, prior.Condition, map[string][]float32{}, len(x) / lm.Hidden, prefix}
	if in.Rows != 29 || prefix != 15 {
		t.Fatal("sequence geometry")
	}
	for _, mode := range []string{"device-time", "host-time-control"} {
		var diag, primary []float32
		for _, r := range captured.Results {
			if r.Mode != mode {
				continue
			}
			if r.Name == "diagonal" {
				diag = r.DCondition
			} else if r.Name == "primary" {
				primary = r.DCondition
			} else if r.Name == "endpoint" && !r.StoppedEndpoint {
				t.Fatal("endpoint policy")
			}
		}
		seed, e := trunkCaptureSeed(diag, primary, in.Rows, lm.Hidden, prefix, 13)
		if e != nil {
			t.Fatal(e)
		}
		in.Seeds[mode] = seed
	}
	return in
}
func trunkCaptureBuild(m *TransformerCPU, x, seed []float32, rows int) (TransformerTrainingTape, int64, int64, int, error) {
	g, e := m.buildTransformerTrainingTape(x, seed, rows, 2<<30, true)
	if e != nil {
		return g, 0, 0, 0, e
	}
	if e = train.ValidateResidentTapeBindings(g.Tape, g.ResidentBindings, 2<<30); e != nil {
		return g, 0, 0, 0, e
	}
	var source, payload int64
	stages := 0
	aliases := map[int]bool{}
	for _, c := range g.Tape.Commands {
		if c.Op == "reshape" {
			aliases[c.Output] = true
		} else {
			stages++
		}
	}
	for id, s := range g.Tape.Tensors {
		n, e := s.Elements()
		if e != nil {
			return g, 0, 0, 0, e
		}
		source += int64(n) * 4
		if !aliases[id] {
			payload += int64(n) * 4
		}
	}
	if payload > 2<<30 {
		return g, source, payload, stages, fmt.Errorf("trunk native payload")
	}
	return g, source, payload, stages, nil
}
func TestPocketTrunkCaptureTiny(t *testing.T) {
	lm := tinyFlowLMTraining()
	b := FlowLMTrainingBatch{Frames: 3, VoiceFrames: 2, TextTokens: []uint32{1}, NormalizedLatents: make([]float32, 3*lm.LatentDim), VoiceLatents: make([]float32, 2*lm.LatentDim)}
	x, p, e := trunkCaptureSequence(lm, b)
	if e != nil {
		t.Fatal(e)
	}
	diag, primary := make([]float32, lm.Hidden), make([]float32, lm.Hidden)
	for i := range diag {
		diag[i] = float32(i+1) * .01
		primary[i] = -float32(i+1) * .005
	}
	seed, e := trunkCaptureSeed(diag, primary, len(x)/lm.Hidden, lm.Hidden, p, 2)
	if e != nil {
		t.Fatal(e)
	}
	for i, v := range seed {
		if i < p*lm.Hidden || i >= (p+1)*lm.Hidden {
			if v != 0 {
				t.Fatal("seed padding")
			}
		} else if v != diag[i%lm.Hidden]*.375+primary[i%lm.Hidden]*.125 {
			t.Fatal("objective scaling")
		}
	}
	source, e := lm.forwardTrainingTape(b, make([]float32, len(b.NormalizedLatents)/lm.LatentDim*lm.Hidden), make([]float32, b.Frames))
	if e != nil {
		t.Fatal(e)
	}
	assertSliceClose(t, "trunk sequence assembly", x, source.output.Sequence, 0)
	g, s, n, _, e := trunkCaptureBuild(lm.Transformer, x, seed, len(x)/lm.Hidden)
	if e != nil || s < n || n <= 0 {
		t.Fatal(s, n, e)
	}
	got, e := g.Decode(interpretFlowTape(t, g.Tape))
	if e != nil {
		t.Fatal(e)
	}
	out, grad, dx, e := lm.Transformer.ForwardBackward(x, seed, len(x)/lm.Hidden)
	if e != nil {
		t.Fatal(e)
	}
	assertSliceClose(t, "trunk tiny output", got.Output, out, 4e-6)
	assertSliceClose(t, "trunk tiny dx", got.DSequence, dx, 8e-6)
	ref := map[string][]float32{}
	addFlowLMGradients(ref, &FlowLMTrainingGradients{Transformer: grad})
	for name, v := range got.Parameters {
		assertSliceClose(t, "trunk tiny "+name, v, ref[name], 1e-5)
	}
	for _, args := range []struct{ rows, prefix, valid int }{{0, p, 2}, {len(x) / lm.Hidden, -1, 2}, {len(x) / lm.Hidden, p, 0}, {len(x) / lm.Hidden, p, 100}} {
		if _, e := trunkCaptureSeed(diag, primary, args.rows, lm.Hidden, args.prefix, args.valid); e == nil {
			t.Fatal("seed admission")
		}
	}
}
func TestPocketTrunkCaptureReleasedPreparation(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_TRUNK_PREPARE") != "1" {
		t.Skip("new released trunk source admission only")
	}
	in := trunkCapturePrepare(t)
	type row struct {
		Mode                         string
		SourceBytes, NativePayload   int64
		Stages, Parameters, Elements int
	}
	var inventory []row
	for _, mode := range []string{"device-time", "host-time-control"} {
		g, s, n, st, e := trunkCaptureBuild(in.LM.Transformer, in.Sequence, in.Seeds[mode], in.Rows)
		if e != nil {
			t.Fatal(e)
		}
		count := 0
		for _, name := range g.ParameterNames {
			count += len(g.Tape.Tensors[g.ResidentBindings[name].Parameter].Values)
		}
		inventory = append(inventory, row{mode, s, n, st, len(g.ParameterNames), count})
		t.Logf("%s source=%d nativePayload=%d stages=%d parameters=%d elements=%d", mode, s, n, st, len(g.ParameterNames), count)
	}
	if e := flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_TRUNK_OUTPUT"), "preparation.json"), struct {
		Scope        string
		Rows, Prefix int
		Sequence     []float32
		Seeds        map[string][]float32
		Inventory    []row
	}{"CPUconditioner assembly/source admission only; no CPU/native trunk evaluation", in.Rows, in.Prefix, in.Sequence, in.Seeds, inventory}); e != nil {
		t.Fatal(e)
	}
}
func TestPocketNativeTrunkCapture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_TRUNK_NATIVE") != "1" {
		t.Skip("explicit bounded standalone native trunk VJP")
	}
	in := trunkCapturePrepare(t)
	// Admit both complete sources before runtime; graph parameters are borrowed.
	graphs := map[string]TransformerTrainingTape{}
	for _, mode := range []string{"device-time", "host-time-control"} {
		g, _, _, _, e := trunkCaptureBuild(in.LM.Transformer, in.Sequence, in.Seeds[mode], in.Rows)
		if e != nil {
			t.Fatal(e)
		}
		graphs[mode] = g
	}
	if os.Getenv("GO_PHERENCE_DISABLE_VULKAN") != "" || os.Getenv("VK_DRIVER_FILES") != "/usr/share/vulkan/icd.d/nvidia_icd.json" {
		t.Fatal("explicit Vulkan ICD")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("runtime/device mismatch no fallback")
	}
	if e := vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 2 << 30, MaxAllocations: 4096}); e != nil {
		t.Fatal(e)
	}
	op, e := vulkan.NewVkTrainingF32(ctx)
	if e != nil {
		t.Fatal(e)
	}
	type result struct {
		Mode, GradientFile                                     string
		Seed, Output, DSequence                                []float32
		OutputError, InputError, ConditionError, GradientError reverseError
		Manifest                                               []reverseManifest
		Native                                                 vulkan.VulkanMemoryUsage
		RSS                                                    uint64
	}
	report := struct {
		Scope, Device string
		Rows, Prefix  int
		Results       []result
		Final         vulkan.VulkanMemoryUsage
	}{Scope: "Standalone native 29-row transformer at CPUconditioner sequence and objective-weighted first-row captured native-flow seeds; excludes nativeconditioner/allrows/EOS/update/fullstep acceptance.", Device: vulkan.VulkanDeviceName(), Rows: in.Rows, Prefix: in.Prefix}
	for _, mode := range []string{"device-time", "host-time-control"} {
		seed, g := in.Seeds[mode], graphs[mode]
		expected, cpuGrad, dx, e := in.LM.Transformer.ForwardBackward(in.Sequence, seed, in.Rows)
		if e != nil {
			t.Fatal(e)
		}
		cpuAll := map[string][]float32{}
		addFlowLMGradients(cpuAll, &FlowLMTrainingGradients{Transformer: cpuGrad})
		reference := map[string][]float32{}
		for _, name := range g.ParameterNames {
			reference[name] = cpuAll[name]
		}
		if rss, e := flowCaptureRSS(); e != nil || rss > 6<<30 {
			t.Fatal("host cap before native", rss, e)
		}
		tape, e := vulkan.NewVkTrainingTape(ctx, op, g.Tape, 2<<30)
		if e != nil {
			t.Fatal("native construction no retry", e)
		}
		usage := vulkan.VulkanMemoryStats()
		rss, e := flowCaptureRSS()
		if e != nil || rss > 6<<30 || usage.Bytes > 2<<30 {
			t.Fatal("native/host cap", rss, usage, e)
		}
		values, e := tape.Run(ctx)
		if e != nil {
			t.Fatal("native failure retained no retry", e)
		}
		if len(values) != 2+len(g.ParameterNames) {
			t.Fatal("trunk outputs")
		}
		native := map[string][]float32{}
		for i, name := range g.ParameterNames {
			native[name] = values[i+2]
		}
		r := result{Mode: mode, GradientFile: mode + "-trunk.f32", Seed: seed, Output: values[0], DSequence: values[1], Native: usage, RSS: rss}
		r.OutputError, e = reverseCompare(values[0], expected)
		if e != nil {
			t.Fatal(e)
		}
		r.InputError, e = reverseCompare(values[1], dx)
		if e != nil {
			t.Fatal(e)
		}
		r.ConditionError, e = reverseCompare(values[0][in.Prefix*1024:(in.Prefix+1)*1024], in.CPUCondition)
		if e != nil {
			t.Fatal(e)
		}
		r.Manifest, r.GradientError, e = reverseWriteGradients(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_TRUNK_OUTPUT"), r.GradientFile), native, reference)
		if e != nil {
			t.Fatal(e)
		}
		if e = tape.Close(); e != nil {
			t.Fatal("close failure no retry", e)
		}
		if u := vulkan.VulkanMemoryStats(); u.Bytes != 0 || u.Allocations != 0 || u.InFlight || u.Uncertain || u.DeviceLost {
			t.Fatal("trunk closure", u)
		}
		report.Results = append(report.Results, r)
		t.Logf("%s outputmax=%g conditionmax=%g inputmax=%g grads=%d max=%g RMS=%g signs=%d diagnosticAbove4e-6=%d native=%d RSS=%d", mode, r.OutputError.Max, r.ConditionError.Max, r.InputError.Max, r.GradientError.Count, r.GradientError.Max, r.GradientError.RMS, r.GradientError.OppositeSigns, r.GradientError.Over4e6, usage.Bytes, rss)
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	report.Final = vulkan.VulkanMemoryStats()
	if e = flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_TRUNK_OUTPUT"), "native-trunk.json"), report); e != nil {
		t.Fatal(e)
	}
	runtime.KeepAlive(in)
}
