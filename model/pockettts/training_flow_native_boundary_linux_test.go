//go:build linux

package pockettts

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

type flowCaptureBoundary struct {
	Name, Op                       string
	Kind                           uint32
	ExpectedValue, ExpectedTangent []float32
}
type flowCaptureGraph struct {
	Name, Mode                 string
	Tape                       train.Tape
	Boundaries                 []flowCaptureBoundary
	SourceBytes, NativePayload int64
	Stages                     int
}

// Enumerate CPU forward boundaries in production graph order. Comparison-only
// arrays never become TensorSpec.Values in the device-time graph.
func flowCaptureExpected(m *FlowHeadCPU, t *dualFlowTape, deviceTime bool) []flowCaptureBoundary {
	var out []flowCaptureBoundary
	add := func(name, op string, kind uint32, x dualVector) {
		out = append(out, flowCaptureBoundary{name, op, kind, x.value, x.tangent})
	}
	add("input-projection", "linear-forward", 0, t.input.output)
	add("condition-projection", "linear-forward", 0, t.condition.output)
	combined := cloneDual(t.condition.output)
	for i, v := range t.times {
		p := fmt.Sprintf("time-%d/", i)
		if deviceTime {
			add(p+"embedding", "time-embedding", 0, v.embedding)
		}
		add(p+"fc1", "linear-forward", 0, v.fc1Pre)
		add(p+"silu", "silu-forward", 0, v.fc1)
		add(p+"fc2", "linear-forward", 0, v.fc2Pre)
		add(p+"variance-norm", "norm-forward", 1, v.output)
		for j := range combined.value {
			combined.value[j] += v.output.value[j] / float32(len(t.times))
			combined.tangent[j] += v.output.tangent[j] / float32(len(t.times))
		}
		add(p+"condition-sum", "vector", 0, cloneDual(combined))
	}
	slice := func(x dualVector, at, n int) dualVector { return dualVector{x.value[at : at+n], x.tangent[at : at+n]} }
	for i, b := range t.blocks {
		p := fmt.Sprintf("block-%d/", i)
		d := len(b.input.value)
		add(p+"condition-silu", "silu-forward", 0, b.condAct)
		add(p+"modulation", "linear-forward", 0, b.mod)
		add(p+"norm", "norm-forward", 0, b.norm)
		add(p+"shift-scale", "vector", 0, slice(b.mod, 0, 2*d))
		add(p+"gate", "vector", 0, slice(b.mod, 2*d, d))
		add(p+"modulated", "mod-forward", 0, b.modulated)
		add(p+"fc1", "linear-forward", 0, b.fc1Pre)
		add(p+"silu", "silu-forward", 0, b.fc1)
		add(p+"fc2", "linear-forward", 0, b.update)
		gated := dualVector{make([]float32, d), make([]float32, d)}
		for j := 0; j < d; j++ {
			gated.value[j] = b.update.value[j] * b.mod.value[2*d+j]
			gated.tangent[j] = b.update.tangent[j]*b.mod.value[2*d+j] + b.update.value[j]*b.mod.tangent[2*d+j]
		}
		add(p+"gated-update", "mod-forward", 1, gated)
		add(p+"output", "vector", 0, b.output)
	}
	f := t.final
	add("final/condition-silu", "silu-forward", 0, f.condAct)
	add("final/modulation", "linear-forward", 0, f.mod)
	add("final/norm", "norm-forward", 0, f.norm)
	add("final/modulated", "mod-forward", 0, f.modulated)
	add("final/output", "linear-forward", 0, f.output)
	return out
}
func flowCaptureBuild(m *FlowHeadCPU, name string, c, times, x []float32, index int, deviceTime bool) (flowCaptureGraph, error) {
	const capBytes = int64(2 << 30)
	complete, err := m.buildFlowDualTrainingTape(c, times, x, index, make([]float32, m.Final.Linear.Out), make([]float32, m.Final.Linear.Out), capBytes, deviceTime, true)
	if err != nil {
		return flowCaptureGraph{}, err
	}
	var source int64
	for _, s := range complete.Tape.Tensors {
		n, e := s.Elements()
		if e != nil {
			return flowCaptureGraph{}, e
		}
		source += int64(n) * 4
	}
	if err = train.ValidateResidentTapeBindings(complete.Tape, complete.ResidentBindings, capBytes); err != nil {
		return flowCaptureGraph{}, err
	}
	cpu := m.forwardDualTape(c, times, x, index)
	expected := flowCaptureExpected(m, cpu, deviceTime)
	// All forward commands precede the final output view; reject any unexpected
	// op/pair rather than accepting positional metadata drift.
	last := complete.Tape.Outputs[1]
	var commands []train.Command
	auxiliary := map[int]bool{}
	for i := 3; i < len(complete.Tape.Commands); i++ {
		first, second, partial, last := complete.Tape.Commands[i-3], complete.Tape.Commands[i-2], complete.Tape.Commands[i-1], complete.Tape.Commands[i]
		if first.Op == "mod-forward" && first.Kind == 1 && first.Mode == 0 && second.Op == "mod-forward" && second.Kind == 1 && second.Mode == 0 && partial.Op == "vector" && len(partial.Inputs) == 2 && partial.Inputs[1] == first.Output && last.Op == "vector" && len(last.Inputs) == 2 && last.Inputs[0] == partial.Output && last.Inputs[1] == second.Output {
			auxiliary[first.Output] = true
			auxiliary[second.Output] = true
			auxiliary[partial.Output] = true
		}
	}
	for _, cmd := range complete.Tape.Commands {
		if cmd.Output > last {
			break
		}
		if auxiliary[cmd.Output] || cmd.Op == "reshape" || cmd.Op == "norm-forward" && cmd.Mode == 0 {
			continue
		}
		commands = append(commands, cmd)
	}
	if len(commands) != 2*len(expected) {
		return flowCaptureGraph{}, fmt.Errorf("forward boundary count %d want %d", len(commands), 2*len(expected))
	}
	var roots []int
	for i, b := range expected {
		a, z := commands[2*i], commands[2*i+1]
		vMode, tMode := uint32(0), uint32(1)
		if b.Op == "norm-forward" {
			vMode, tMode = 1, 2
		}
		if b.Op == "vector" {
			vMode, tMode = 0, 0
			if strings.HasSuffix(b.Name, "shift-scale") || strings.HasSuffix(b.Name, "gate") {
				vMode, tMode = 1, 1
			}
		}
		if a.Op != b.Op || z.Op != b.Op || a.Kind != b.Kind || z.Kind != b.Kind || a.Mode != vMode || z.Mode != tMode {
			return flowCaptureGraph{}, fmt.Errorf("boundary pairing %s %v %v", b.Name, a, z)
		}
		vn, _ := complete.Tape.Tensors[a.Output].Elements()
		tn, _ := complete.Tape.Tensors[z.Output].Elements()
		if vn != len(b.ExpectedValue) || tn != len(b.ExpectedTangent) {
			return flowCaptureGraph{}, fmt.Errorf("boundary extent %s", b.Name)
		}
		roots = append(roots, a.Output, z.Output)
	}
	complete.Tape.Outputs = append([]int(nil), roots...) // declared diagnostic roots after full-source admission
	forward, _, err := train.PruneTape(complete.Tape, roots, capBytes)
	if err != nil {
		return flowCaptureGraph{}, err
	}
	var payload int64
	stages := 0
	aliases := map[int]bool{}
	for _, cmd := range forward.Commands {
		if cmd.Op == "reshape" {
			aliases[cmd.Output] = true
		} else {
			stages++
			if strings.Contains(cmd.Op, "backward") || cmd.Op == "lsd-loss" {
				return flowCaptureGraph{}, fmt.Errorf("reverse/loss survived")
			}
		}
	}
	for id, s := range forward.Tensors {
		if aliases[id] {
			continue
		}
		n, e := s.Elements()
		if e != nil {
			return flowCaptureGraph{}, e
		}
		payload += int64(n) * 4
	}
	if payload > capBytes {
		return flowCaptureGraph{}, fmt.Errorf("forward native payload cap")
	}
	mode := "host-time-control"
	if deviceTime {
		mode = "device-time"
	}
	return flowCaptureGraph{name, mode, forward, expected, source, payload, stages}, nil
}
func flowCapturePinned(path, pin string, limit int64, out any) error {
	info, e := os.Stat(path)
	if e != nil {
		return e
	}
	if !info.Mode().IsRegular() || info.Size() > limit {
		return fmt.Errorf("input bound")
	}
	raw, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	sum := sha256.Sum256(raw)
	if hex.EncodeToString(sum[:]) != pin {
		return fmt.Errorf("input pin")
	}
	return json.Unmarshal(raw, out)
}
func flowCaptureWrite(path string, x any) error {
	raw, e := json.MarshalIndent(x, "", "  ")
	if e != nil {
		return e
	}
	f, e := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if e != nil {
		return e
	}
	_, we := f.Write(append(raw, '\n'))
	ce := f.Close()
	if we != nil {
		return we
	}
	return ce
}
func flowCaptureRSS() (uint64, error) {
	raw, e := os.ReadFile("/proc/self/status")
	if e != nil {
		return 0, e
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			f := strings.Fields(line)
			if len(f) != 3 {
				return 0, fmt.Errorf("RSS")
			}
			n, e := strconv.ParseUint(f[1], 10, 64)
			return n * 1024, e
		}
	}
	return 0, fmt.Errorf("RSS absent")
}

func TestPocketFlowCaptureTinyAdmission(t *testing.T) {
	m := tinyTrainableFlowHead()
	for _, device := range []bool{false, true} {
		g, e := flowCaptureBuild(m, "tiny", []float32{.2, -.3, .5}, []float32{.3, .8}, []float32{.3, -.2}, 1, device)
		if e != nil {
			t.Fatal(e)
		}
		if len(g.Boundaries) < 20 || g.SourceBytes <= g.NativePayload || len(g.Tape.Outputs) != 2*len(g.Boundaries) {
			t.Fatal("inventory", g.SourceBytes, g.NativePayload)
		}
		if e = g.Tape.Validate(2 << 30); e != nil {
			t.Fatal(e)
		}
		found := false
		for _, c := range g.Tape.Commands {
			if c.Op == "time-embedding" {
				found = true
			}
			if strings.Contains(c.Op, "backward") {
				t.Fatal("reverse survived")
			}
		}
		if found != device {
			t.Fatal("time mode")
		}
	}
	if _, e := flowCaptureBuild(m, "bad", nil, []float32{.3, .8}, []float32{.3, -.2}, 1, true); e == nil {
		t.Fatal("shape")
	}
	if _, e := flowCaptureBuild(m, "bad", []float32{.2, -.3, .5}, []float32{.3, .8}, []float32{.3, -.2}, 2, true); e == nil {
		t.Fatal("index")
	}
}
func flowCapturePrepare(t *testing.T) ([]flowCaptureGraph, []float32) {
	t.Helper()
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
	if len(corpus.Rows) != 1 || saved.Identity.Backend != "simd" || saved.Identity.GoVersion != runtime.Version() {
		t.Fatal("replay identity")
	}
	row := corpus.Rows[0]
	if row.Batch.Frames != 14 || row.Batch.VoiceFrames != 13 || row.ValidFrames != 13 || len(row.Batch.TextTokens) != 1 {
		t.Fatal("geometry")
	}
	run, e := NewTrainingRunState(saved.Identity, 1, [6]uint64{42, 43, 44, 45, 46, 47})
	if e != nil {
		t.Fatal(e)
	}
	proposal, e := run.PrepareRow()
	if e != nil {
		t.Fatal(e)
	}
	samples := runnerSampleBuffers(14, 32)
	recipe := TrainingRandomRecipe{Algorithm: TrainingRandomAlgorithm, OptionsSHA256: saved.Identity.OptionsSHA256, DistillProbability: 1, FlowBatchMultiplier: 1, StopGradient: "minimal", Normalize: true, PEqual: .75, EOSLossWeight: .1, LogitMean: .4, LogitStd: 1}
	if e = SampleTrainingInto(samples, proposal, 14, 13, 32, recipe, TrainingRandomLimits{TrainingSampleLimits{375, 375 * 32}, 2*375*32 + 6*375}); e != nil {
		t.Fatal(e)
	}
	if hex.EncodeToString(proposal.next.timeRNG[:]) != saved.TimeRNG || hex.EncodeToString(proposal.next.noiseRNG[:]) != saved.NoiseRNG {
		t.Fatal("sample drift")
	}
	lm, flow, _, _ := releasedTrainingModels(t)
	trunk, e := lm.forwardTrainingTape(row.Batch, make([]float32, 14*lm.Hidden), make([]float32, 14))
	if e != nil {
		t.Fatal(e)
	}
	condition := append([]float32(nil), trunk.output.Z[:lm.Hidden]...)
	noise, target := samples.Noise[:32], row.Batch.NormalizedLatents[:32]
	interpolate := func(tm float32) []float32 {
		x := make([]float32, 32)
		for i := range x {
			x[i] = tm*target[i] + (1-tm)*noise[i]
		}
		return x
	}
	s, tm := samples.DistillS[0], samples.DistillT[0]
	xs := interpolate(s)
	primary := flow.forwardDualTape(condition, []float32{s, tm}, xs, 1)
	xt := make([]float32, 32)
	for i := range xt {
		xt[i] = xs[i] + (tm-s)*primary.final.output.value[i]
	}
	var graphs []flowCaptureGraph
	for _, branch := range []struct {
		name     string
		times, x []float32
		index    int
	}{{"diagonal", []float32{samples.DiagonalTime[0], samples.DiagonalTime[0]}, interpolate(samples.DiagonalTime[0]), -1}, {"primary", []float32{s, tm}, xs, 1}, {"endpoint", []float32{tm, tm}, xt, -1}} {
		for _, device := range []bool{true, false} {
			g, e := flowCaptureBuild(flow, branch.name, condition, branch.times, branch.x, branch.index, device)
			if e != nil {
				t.Fatal(e)
			}
			graphs = append(graphs, g)
		}
	}
	runtime.KeepAlive(lm)
	runtime.KeepAlive(trunk)
	return graphs, condition
}
func TestPocketFlowCaptureReleasedPreparation(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_FLOW_CAPTURE_PREPARE") != "1" {
		t.Skip("explicit released capture preparation")
	}
	graphs, condition := flowCapturePrepare(t)
	type inventory struct {
		Name, Mode                 string
		SourceBytes, NativePayload int64
		Stages, Boundaries         int
	}
	var rows []inventory
	for _, g := range graphs {
		rows = append(rows, inventory{g.Name, g.Mode, g.SourceBytes, g.NativePayload, g.Stages, len(g.Boundaries)})
		t.Logf("%s/%s source=%d nativePayload=%d stages=%d boundaries=%d", g.Name, g.Mode, g.SourceBytes, g.NativePayload, g.Stages, len(g.Boundaries))
	}
	if e := flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_FLOW_CAPTURE_OUTPUT"), "preparation.json"), struct {
		Scope     string
		Rows      []inventory
		Condition []float32
	}{"CPU preparation/source admission only; no AD/backward/native execution or updates", rows, condition}); e != nil {
		t.Fatal(e)
	}
	runtime.KeepAlive(graphs)
}

type flowCaptureError struct {
	Count, Changed, Over4e6 int
	Max, RMS                float64
}

func flowCaptureCompare(a, b []float32) (flowCaptureError, error) {
	var out flowCaptureError
	if len(a) != len(b) {
		return out, fmt.Errorf("compare extent")
	}
	var square float64
	for i, v := range a {
		d := math.Abs(float64(v) - float64(b[i]))
		if math.IsNaN(d) || math.IsInf(d, 0) {
			return out, fmt.Errorf("compare nonfinite")
		}
		out.Count++
		if d != 0 {
			out.Changed++
		}
		if d > 4e-6 {
			out.Over4e6++
		}
		out.Max = math.Max(out.Max, d)
		square += d * d
	}
	if out.Count > 0 {
		out.RMS = math.Sqrt(square / float64(out.Count))
	}
	return out, nil
}
func TestPocketFlowCaptureEvidenceValidation(t *testing.T) {
	got, err := flowCaptureCompare([]float32{1, 2, 3}, []float32{1, 2 + 1e-6, 3 + 1e-5})
	if err != nil || got.Count != 3 || got.Changed != 2 || got.Over4e6 != 1 || got.Max < 9e-6 || got.RMS <= 0 {
		t.Fatal(got, err)
	}
	for _, values := range [][]float32{{1, 2}, {1, float32(math.NaN()), 3}, {1, float32(math.Inf(1)), 3}} {
		if _, err := flowCaptureCompare(values, []float32{1, 2, 3}); err == nil {
			t.Fatal("invalid comparison admitted")
		}
	}
	path := filepath.Join(t.TempDir(), "evidence.json")
	if err := flowCaptureWrite(path, got); err != nil {
		t.Fatal(err)
	}
	if err := flowCaptureWrite(path, got); err == nil {
		t.Fatal("evidence overwrite admitted")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	var decoded flowCaptureError
	if err := flowCapturePinned(path, hex.EncodeToString(sum[:]), 4096, &decoded); err != nil || decoded.Count != 3 {
		t.Fatal(decoded, err)
	}
	if err := flowCapturePinned(path, "bad-pin", 4096, &decoded); err == nil {
		t.Fatal("pin mismatch admitted")
	}
	if err := flowCapturePinned(path, hex.EncodeToString(sum[:]), 1, &decoded); err == nil {
		t.Fatal("input ceiling admitted")
	}
}

func TestPocketNativeFlowBoundaryCapture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_FLOW_CAPTURE_NATIVE") != "1" {
		t.Skip("explicit one-shot bounded native flow capture")
	}
	graphs, condition := flowCapturePrepare(t) // all source admission before runtime
	if os.Getenv("GO_PHERENCE_DISABLE_VULKAN") != "" || os.Getenv("VK_DRIVER_FILES") != "/usr/share/vulkan/icd.d/nvidia_icd.json" {
		t.Fatal("explicit Vulkan/ICD required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 75*time.Second)
	defer cancel()
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("device unavailable/mismatch, no fallback")
	}
	if e := vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 2 << 30, MaxAllocations: 4096}); e != nil {
		t.Fatal(e)
	}
	op, e := vulkan.NewVkTrainingF32(ctx)
	if e != nil {
		t.Fatal(e)
	}
	type boundary struct {
		Name                     string
		Value, Tangent           []float32
		ValueError, TangentError flowCaptureError
	}
	type result struct {
		Name, Mode                 string
		SourceBytes, NativePayload int64
		Native                     vulkan.VulkanMemoryUsage
		RSS                        uint64
		Boundaries                 []boundary
	}
	report := struct {
		Scope, Device string
		Condition     []float32
		Graphs        []result
		FinalNative   vulkan.VulkanMemoryUsage
	}{Scope: "First-row composed forward flow/JVP at shared CPU condition/latent boundaries; no reverse/training update. Host-time native control diagnostic only. 4e-6 counts are intermediate diagnostics, not acceptance.", Device: vulkan.VulkanDeviceName(), Condition: condition}
	for _, g := range graphs {
		if rss, e := flowCaptureRSS(); e != nil || rss > 6<<30 {
			t.Fatal("host admission", rss, e)
		}
		tape, e := vulkan.NewVkTrainingTape(ctx, op, g.Tape, 2<<30)
		if e != nil {
			t.Fatal("construction failure; no retry/cleanup uncertain owners", e)
		}
		usage := vulkan.VulkanMemoryStats()
		rss, e := flowCaptureRSS()
		if e != nil || rss > 6<<30 || usage.Bytes > 2<<30 {
			t.Fatal("phase caps", rss, usage, e)
		}
		values, e := tape.Run(ctx)
		if e != nil {
			t.Fatal("native failure retain owners, no retry", e)
		}
		r := result{Name: g.Name, Mode: g.Mode, SourceBytes: g.SourceBytes, NativePayload: g.NativePayload, Native: usage, RSS: rss}
		if len(values) != 2*len(g.Boundaries) {
			t.Fatal("capture extent")
		}
		for i, b := range g.Boundaries {
			ve, e := flowCaptureCompare(values[2*i], b.ExpectedValue)
			if e != nil {
				t.Fatal(e)
			}
			te, e := flowCaptureCompare(values[2*i+1], b.ExpectedTangent)
			if e != nil {
				t.Fatal(e)
			}
			r.Boundaries = append(r.Boundaries, boundary{b.Name, values[2*i], values[2*i+1], ve, te})
		}
		if e = tape.Close(); e != nil {
			t.Fatal("close failed no retry", e)
		}
		if u := vulkan.VulkanMemoryStats(); u.Bytes != 0 || u.Allocations != 0 || u.InFlight || u.Uncertain || u.DeviceLost {
			t.Fatal("native closure", u)
		}
		report.Graphs = append(report.Graphs, r)
		last := r.Boundaries[len(r.Boundaries)-1]
		t.Logf("%s/%s output max=%g RMS=%g over4e-6=%d tangent max=%g RMS=%g over4e-6=%d payload=%d actual=%d RSS=%d", g.Name, g.Mode, last.ValueError.Max, last.ValueError.RMS, last.ValueError.Over4e6, last.TangentError.Max, last.TangentError.RMS, last.TangentError.Over4e6, g.NativePayload, usage.Bytes, rss)
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	report.FinalNative = vulkan.VulkanMemoryStats()
	if e = flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_FLOW_CAPTURE_OUTPUT"), "native-flow.json"), report); e != nil {
		t.Fatal(e)
	}
	runtime.KeepAlive(graphs)
}
