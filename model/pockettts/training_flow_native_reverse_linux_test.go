//go:build linux

package pockettts

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
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

type reverseInputs struct {
	Condition, Target, Noise, DiagonalX, PrimaryX, EndpointX []float32
	DiagonalTime, S, T                                       float32
	Flow                                                     *FlowHeadCPU
	Forward                                                  map[string]map[string]dualVector
}

func reverseLossSeed(prediction, target []float32) ([]float32, error) {
	if len(prediction) == 0 || len(prediction) != len(target) || !finiteF32(prediction) || !finiteF32(target) {
		return nil, fmt.Errorf("reverse seed extent/nonfinite")
	}
	out := make([]float32, len(prediction))
	scale := float32(1) / float32(len(out))
	for i := range out {
		out[i] = (2 * (prediction[i] - target[i])) * scale
		if !isFinite(out[i]) {
			return nil, fmt.Errorf("reverse seed overflow")
		}
	}
	return out, nil
}
func reversePrimarySeeds(dr, dx []float32, delta float32) ([]float32, []float32, error) {
	if len(dr) == 0 || len(dr) != len(dx) || !finiteF32(dr) || !finiteF32(dx) || !isFinite(delta) || delta < 0 || delta > 1 {
		return nil, nil, fmt.Errorf("primary seed inputs")
	}
	v, t := make([]float32, len(dr)), make([]float32, len(dr))
	for i := range v {
		v[i] = dr[i] + delta*dx[i]
		t[i] = delta * dr[i]
		if !isFinite(v[i]) || !isFinite(t[i]) {
			return nil, nil, fmt.Errorf("primary seed overflow")
		}
	}
	return v, t, nil
}
func reversePrepare(t *testing.T) reverseInputs {
	t.Helper()
	var preparation struct{ Condition []float32 }
	if e := flowCapturePinned("/workspace/exports/pocket-flow-native-boundary-20261003/preparation.json", "cce846c5fc433e70a1ccd39efd3d97c0215e804b8348778105d69b293ec21cde", 1<<20, &preparation); e != nil {
		t.Fatal(e)
	}
	var previous struct {
		Graphs []struct {
			Name, Mode string
			Boundaries []struct {
				Name           string
				Value, Tangent []float32
			}
		}
	}
	if e := flowCapturePinned("/workspace/exports/pocket-flow-native-reverse-20261003/forward-final-inputs.json", "0fcac2f7fe8d7720124cb18b0e62404794cc54bc59c449c16ad4eaf815459a20", 1<<16, &previous); e != nil {
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
	if len(corpus.Rows) != 1 || len(preparation.Condition) != 1024 || !finiteF32(preparation.Condition) || saved.Identity.GoVersion != runtime.Version() || saved.Identity.Backend != "simd" || saved.Identity.GOOS != runtime.GOOS || saved.Identity.GOARCH != runtime.GOARCH {
		t.Fatal("reverse replay")
	}
	row := corpus.Rows[0]
	if row.Batch.Frames != 14 || row.Batch.VoiceFrames != 13 || row.ValidFrames != 13 || len(row.Batch.TextTokens) != 1 {
		t.Fatal("reverse row")
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
		t.Fatal("sample replay")
	}
	path := releasedAsset(t, "GO_PHERENCE_POCKETTTS_VOICE_MODEL", "fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997", 219029196)
	file, e := safetensors.Open(path)
	if e != nil {
		t.Fatal(e)
	}
	flow, e := LoadFlowHeadTrainingCPU(file, releasedConfig(t))
	ce := file.Close()
	if e != nil || ce != nil {
		t.Fatal(e, ce)
	}
	out := reverseInputs{Condition: preparation.Condition, Target: append([]float32(nil), row.Batch.NormalizedLatents[:32]...), Noise: append([]float32(nil), samples.Noise[:32]...), DiagonalTime: samples.DiagonalTime[0], S: samples.DistillS[0], T: samples.DistillT[0], Flow: flow, Forward: map[string]map[string]dualVector{}}
	interpolate := func(tm float32) []float32 {
		x := make([]float32, 32)
		for i := range x {
			x[i] = tm*out.Target[i] + (1-tm)*out.Noise[i]
		}
		return x
	}
	out.DiagonalX = interpolate(out.DiagonalTime)
	out.PrimaryX = interpolate(out.S)
	v, _, e := flow.ForwardTimeJVP(out.Condition, []float32{out.S, out.T}, out.PrimaryX, 1)
	if e != nil {
		t.Fatal(e)
	}
	out.EndpointX = make([]float32, 32)
	for i := range v {
		out.EndpointX[i] = out.PrimaryX[i] + (out.T-out.S)*v[i]
	}
	for _, g := range previous.Graphs {
		if out.Forward[g.Mode] == nil {
			out.Forward[g.Mode] = map[string]dualVector{}
		}
		if _, ok := out.Forward[g.Mode][g.Name]; ok {
			t.Fatal("duplicate forward")
		}
		found := false
		for _, b := range g.Boundaries {
			if b.Name == "final/output" {
				if len(b.Value) != 32 || len(b.Tangent) != 32 || !finiteF32(b.Value) || !finiteF32(b.Tangent) {
					t.Fatal("forward boundary")
				}
				out.Forward[g.Mode][g.Name] = dualVector{b.Value, b.Tangent}
				found = true
			}
		}
		if !found {
			t.Fatal("missing final")
		}
	}
	for _, mode := range []string{"device-time", "host-time-control"} {
		if len(out.Forward[mode]) != 3 {
			t.Fatal("forward inventory")
		}
	}
	return out
}

type reverseGraph struct {
	FlowDualTrainingTape
	SourceBytes, NativePayload int64
	Stages                     int
	EndpointStopped            bool
}

func reverseBuild(m *FlowHeadCPU, c, times, x []float32, index int, gv, gt []float32, device, endpoint bool) (reverseGraph, error) {
	p, e := m.buildFlowDualTrainingTape(c, times, x, index, gv, gt, 2<<30, device, true)
	if e != nil {
		return reverseGraph{}, e
	}
	if e = train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 2<<30); e != nil {
		return reverseGraph{}, e
	}
	r := reverseGraph{FlowDualTrainingTape: p, EndpointStopped: endpoint}
	for _, s := range p.Tape.Tensors {
		n, e := s.Elements()
		if e != nil {
			return r, e
		}
		r.SourceBytes += int64(n) * 4
	}
	if endpoint {
		roots := append([]int(nil), p.Tape.Outputs[:p.boundarySize()]...)
		t, _, e := train.PruneTape(p.Tape, roots, 2<<30)
		if e != nil {
			return r, e
		}
		r.Tape = t
	}
	aliases := map[int]bool{}
	for _, cmd := range r.Tape.Commands {
		if cmd.Op == "reshape" {
			aliases[cmd.Output] = true
		} else {
			r.Stages++
		}
	}
	for id, s := range r.Tape.Tensors {
		if aliases[id] {
			continue
		}
		n, e := s.Elements()
		if e != nil {
			return r, e
		}
		r.NativePayload += int64(n) * 4
	}
	if r.NativePayload > 2<<30 {
		return r, fmt.Errorf("reverse native payload")
	}
	return r, nil
}
func reverseCPU(m *FlowHeadCPU, c, times, x []float32, index int, gv, gt []float32) (FlowDualTrainingResult, error) {
	var r FlowDualTrainingResult
	var g *FlowHeadGradients
	var e error
	if index < 0 {
		r.Output, g, r.DCondition, r.DTimes, r.DInput, e = m.ForwardBackward(c, times, x, gv)
		r.Tangent = make([]float32, len(r.Output))
	} else {
		r.Output, r.Tangent, g, r.DCondition, r.DTimes, r.DInput, e = m.ForwardTimeJVPBackward(c, times, x, index, gv, gt)
	}
	if e != nil {
		return r, e
	}
	r.Parameters = map[string][]float32{}
	addFlowHeadGradientParameters(r.Parameters, g)
	return r, nil
}

type reverseError struct {
	flowCaptureError
	OppositeSigns int
	MaxReference  float64
}

func reverseCompare(a, b []float32) (reverseError, error) {
	d, e := flowCaptureCompare(a, b)
	r := reverseError{flowCaptureError: d}
	if e != nil {
		return r, e
	}
	for i, v := range a {
		r.MaxReference = math.Max(r.MaxReference, math.Abs(float64(b[i])))
		if v != 0 && b[i] != 0 && (v < 0) != (b[i] < 0) {
			r.OppositeSigns++
		}
	}
	return r, nil
}

type reverseManifest struct {
	Name        string
	OffsetBytes int64
	Elements    int
	Error       reverseError
}

func reverseWriteGradients(path string, native, reference map[string][]float32) ([]reverseManifest, reverseError, error) {
	names := make([]string, 0, len(reference))
	for name := range reference {
		names = append(names, name)
	}
	sort.Strings(names)
	if len(native) != len(reference) {
		return nil, reverseError{}, fmt.Errorf("gradient inventory")
	}
	f, e := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if e != nil {
		return nil, reverseError{}, e
	}
	defer f.Close()
	var rows []reverseManifest
	var all reverseError
	var squares float64
	var offset int64
	chunk := make([]byte, 16384)
	for _, name := range names {
		x, ok := native[name]
		if !ok {
			return nil, all, fmt.Errorf("missing gradient %s", name)
		}
		d, e := reverseCompare(x, reference[name])
		if e != nil {
			return nil, all, e
		}
		rows = append(rows, reverseManifest{name, offset, len(x), d})
		all.Count += d.Count
		all.Changed += d.Changed
		all.Over4e6 += d.Over4e6
		all.OppositeSigns += d.OppositeSigns
		all.Max = math.Max(all.Max, d.Max)
		all.MaxReference = math.Max(all.MaxReference, d.MaxReference)
		squares += d.RMS * d.RMS * float64(d.Count)
		for at := 0; at < len(x); {
			n := min(4096, len(x)-at)
			for i := 0; i < n; i++ {
				binary.LittleEndian.PutUint32(chunk[4*i:], math.Float32bits(x[at+i]))
			}
			if _, e = f.Write(chunk[:4*n]); e != nil {
				return nil, all, e
			}
			at += n
		}
		offset += int64(len(x)) * 4
	}
	if all.Count > 0 {
		all.RMS = math.Sqrt(squares / float64(all.Count))
	}
	if e = f.Close(); e != nil {
		return nil, all, e
	}
	return rows, all, nil
}
func TestPocketReverseCaptureTiny(t *testing.T) {
	m := tinyTrainableFlowHead()
	c, times, x := []float32{.2, -.3, .5}, []float32{.3, .8}, []float32{.3, -.2}
	gv, e := reverseLossSeed([]float32{.4, .6}, []float32{.1, .2})
	if e != nil || gv[0] != (2*float32(.4-.1))/2 {
		t.Fatal(gv, e)
	}
	v, tg, e := reversePrimarySeeds(gv, []float32{.1, -.2}, .5)
	if e != nil || tg[0] != .5*gv[0] || v[0] != gv[0]+.5*.1 {
		t.Fatal(v, tg, e)
	}
	for _, device := range []bool{false, true} {
		for _, endpoint := range []bool{false, true} {
			g, e := reverseBuild(m, c, times, x, 1, v, tg, device, endpoint)
			if e != nil {
				t.Fatal(e)
			}
			if g.SourceBytes < g.NativePayload || g.NativePayload == 0 {
				t.Fatal("resource")
			}
			vals := interpretFlowTape(t, g.Tape)
			r, e := reverseCPU(m, c, times, x, 1, v, tg)
			if e != nil {
				t.Fatal(e)
			}
			if endpoint {
				if len(vals) != g.boundarySize() {
					t.Fatal("stopped inventory")
				}
				continue
			}
			n, e := g.Decode(vals)
			if e != nil {
				t.Fatal(e)
			}
			for name, w := range r.Parameters {
				d, e := reverseCompare(n.Parameters[name], w)
				if e != nil || d.Max > 1e-5 {
					t.Fatal("tiny independent CPU differential", name, d, e)
				}
			}
		}
	}
	for _, bad := range []struct{ a, b []float32 }{{nil, nil}, {[]float32{1}, nil}, {[]float32{float32(math.NaN())}, []float32{0}}} {
		if _, e := reverseLossSeed(bad.a, bad.b); e == nil {
			t.Fatal("seed admission")
		}
	}
	if _, _, e := reversePrimarySeeds([]float32{1}, []float32{1}, -1); e == nil {
		t.Fatal("delta")
	}
	n, r := map[string][]float32{"a": {1, 2}, "b": {-.1}}, map[string][]float32{"a": {1, 2}, "b": {.1}}
	path := filepath.Join(t.TempDir(), "gradients.f32")
	rows, all, e := reverseWriteGradients(path, n, r)
	if e != nil || len(rows) != 2 || all.Count != 3 || all.OppositeSigns != 1 || rows[1].OffsetBytes != 8 {
		t.Fatal(rows, all, e)
	}
	if _, _, e = reverseWriteGradients(path, n, r); e == nil {
		t.Fatal("overwrite")
	}
	if _, e := reverseCompare([]float32{1}, nil); e == nil {
		t.Fatal("extent")
	}
}
func reverseSeeds(in reverseInputs, mode, branch string, endpointDX []float32) (gv, gt []float32) {
	f := in.Forward[mode]
	n := len(in.Target)
	gt = make([]float32, n)
	switch branch {
	case "diagonal":
		desired := make([]float32, n)
		for i := range desired {
			desired[i] = in.Target[i] - in.Noise[i]
		}
		gv, _ = reverseLossSeed(f[branch].value, desired)
	case "endpoint", "primary":
		dxdt := make([]float32, n)
		for i := range dxdt {
			dxdt[i] = f["primary"].value[i] + (in.T-in.S)*f["primary"].tangent[i]
		}
		dr, _ := reverseLossSeed(dxdt, f["endpoint"].value)
		if branch == "endpoint" {
			gv = make([]float32, n)
			for i := range gv {
				gv[i] = -dr[i]
			}
		} else {
			gv, gt, _ = reversePrimarySeeds(dr, endpointDX, in.T-in.S)
		}
	}
	return
}
func reverseBranch(in reverseInputs, name string) ([]float32, []float32, int) {
	switch name {
	case "diagonal":
		return []float32{in.DiagonalTime, in.DiagonalTime}, in.DiagonalX, -1
	case "endpoint":
		return []float32{in.T, in.T}, in.EndpointX, -1
	default:
		return []float32{in.S, in.T}, in.PrimaryX, 1
	}
}
func TestPocketReverseStoppedBoundaryAndSeedOverflow(t *testing.T) {
	if _, err := reverseLossSeed([]float32{math.MaxFloat32}, []float32{-math.MaxFloat32}); err == nil {
		t.Fatal("overflow seed admitted")
	}
	if _, _, err := reversePrimarySeeds([]float32{math.MaxFloat32}, []float32{math.MaxFloat32}, 1); err == nil {
		t.Fatal("mixed seed overflow admitted")
	}
	m := tinyTrainableFlowHead()
	c, times, x := []float32{.2, -.3, .5}, []float32{.8, .8}, []float32{.3, -.2}
	gv, gt := []float32{-.1, .2}, []float32{0, 0}
	for _, device := range []bool{false, true} {
		g, err := reverseBuild(m, c, times, x, -1, gv, gt, device, true)
		if err != nil {
			t.Fatal(err)
		}
		got, err := g.decodeBoundary(interpretFlowTape(t, g.Tape))
		if err != nil {
			t.Fatal(err)
		}
		want, err := reverseCPU(m, c, times, x, -1, gv, gt)
		if err != nil {
			t.Fatal(err)
		}
		d, err := reverseCompare(got.DInput, want.DInput)
		if err != nil || d.Max > 1e-5 || len(got.Parameters) != 0 {
			t.Fatal("stopped endpoint input differential", d, err)
		}
	}
	path := filepath.Join(t.TempDir(), "gradients.f32")
	rows, _, err := reverseWriteGradients(path, map[string][]float32{"z": {-.25}, "a": {1, 2}}, map[string][]float32{"z": {-.25}, "a": {1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil || len(raw) != 12 || rows[0].Name != "a" || rows[1].OffsetBytes != 8 || math.Float32frombits(binary.LittleEndian.Uint32(raw[8:])) != -.25 {
		t.Fatal("named F32 manifest", rows, err)
	}
}

func TestPocketReverseCaptureReleasedPreparation(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_REVERSE_PREPARE") != "1" {
		t.Skip("new released reverse admission only")
	}
	in := reversePrepare(t)
	type row struct {
		Name, Mode         string
		Source, Payload    int64
		Stages, Parameters int
	}
	var rows []row
	for _, mode := range []string{"device-time", "host-time-control"} {
		for _, name := range []string{"diagonal", "endpoint", "primary"} {
			times, x, index := reverseBranch(in, name)
			gv, gt := reverseSeeds(in, mode, name, make([]float32, 32))
			g, e := reverseBuild(in.Flow, in.Condition, times, x, index, gv, gt, mode == "device-time", name == "endpoint")
			if e != nil {
				t.Fatal(e)
			}
			rows = append(rows, row{name, mode, g.SourceBytes, g.NativePayload, g.Stages, len(g.ParameterNames)})
			t.Logf("%s/%s source=%d payload=%d stages=%d namedParameters=%d endpointStopped=%t", name, mode, g.SourceBytes, g.NativePayload, g.Stages, len(g.ParameterNames), name == "endpoint")
		}
	}
	if e := flowCaptureWrite(filepath.Join(os.Getenv("GO_PHERENCE_POCKET_REVERSE_OUTPUT"), "preparation.json"), rows); e != nil {
		t.Fatal(e)
	}
}
func TestPocketNativeReverseCapture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_REVERSE_NATIVE") != "1" {
		t.Skip("explicit one-shot native reverse capture")
	}
	in := reversePrepare(t)
	if os.Getenv("GO_PHERENCE_DISABLE_VULKAN") != "" || os.Getenv("VK_DRIVER_FILES") != "/usr/share/vulkan/icd.d/nvidia_icd.json" {
		t.Fatal("explicit Vulkan ICD")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("native unavailable no fallback")
	}
	if e := vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 2 << 30, MaxAllocations: 4096}); e != nil {
		t.Fatal(e)
	}
	op, e := vulkan.NewVkTrainingF32(ctx)
	if e != nil {
		t.Fatal(e)
	}
	type result struct {
		Name, Mode                                             string
		GV, GT, DCondition, DInput, DTimes                     []float32
		CPUCondition, CPUInput, CPUTimes                       []float32
		ConditionError, InputError, TimeError, ParametersError reverseError
		Manifest                                               []reverseManifest
		GradientFile                                           string
		Native                                                 vulkan.VulkanMemoryUsage
		RSS                                                    uint64
		StoppedEndpoint                                        bool
	}
	report := struct {
		Scope, Device                                            string
		Condition, Target, Noise, DiagonalX, PrimaryX, EndpointX []float32
		DiagonalTime, S, T                                       float32
		Results                                                  []result
		Final                                                    vulkan.VulkanMemoryUsage
	}{Scope: "First-row loss-derived reverse chains using saved native forward at CPU-defined latent/condition boundaries. Named gradients comparison-only CPU oracle; endpoint direct derivatives stopped. No optimizer/all-row/fullstep qualification.", Device: vulkan.VulkanDeviceName(), Condition: in.Condition, Target: in.Target, Noise: in.Noise, DiagonalX: in.DiagonalX, PrimaryX: in.PrimaryX, EndpointX: in.EndpointX, DiagonalTime: in.DiagonalTime, S: in.S, T: in.T}
	root := os.Getenv("GO_PHERENCE_POCKET_REVERSE_OUTPUT")
	for _, mode := range []string{"device-time", "host-time-control"} {
		var endpointDX []float32
		for _, name := range []string{"diagonal", "endpoint", "primary"} {
			times, x, index := reverseBranch(in, name)
			gv, gt := reverseSeeds(in, mode, name, endpointDX)
			g, e := reverseBuild(in.Flow, in.Condition, times, x, index, gv, gt, mode == "device-time", name == "endpoint")
			if e != nil {
				t.Fatal(e)
			}
			reference, e := reverseCPU(in.Flow, in.Condition, times, x, index, gv, gt)
			if e != nil {
				t.Fatal(e)
			}
			if rss, e := flowCaptureRSS(); e != nil || rss > 6<<30 {
				t.Fatal("preconstruction host cap", rss, e)
			}
			tape, e := vulkan.NewVkTrainingTape(ctx, op, g.Tape, 2<<30)
			if e != nil {
				t.Fatal("native construction failure no retry", e)
			}
			usage := vulkan.VulkanMemoryStats()
			rss, e := flowCaptureRSS()
			if e != nil || rss > 6<<30 || usage.Bytes > 2<<30 {
				t.Fatal("capture caps", rss, usage, e)
			}
			values, e := tape.Run(ctx)
			if e != nil {
				t.Fatal("native failure retain owners no retry", e)
			}
			var native FlowDualTrainingResult
			if name == "endpoint" {
				native, e = g.decodeBoundary(values)
			} else {
				native, e = g.Decode(values)
			}
			if e != nil {
				t.Fatal(e)
			}
			r := result{Name: name, Mode: mode, GV: gv, GT: gt, DCondition: native.DCondition, DInput: native.DInput, DTimes: native.DTimes, CPUCondition: reference.DCondition, CPUInput: reference.DInput, CPUTimes: reference.DTimes, Native: usage, RSS: rss, StoppedEndpoint: name == "endpoint"}
			r.ConditionError, e = reverseCompare(native.DCondition, reference.DCondition)
			if e != nil {
				t.Fatal(e)
			}
			r.InputError, e = reverseCompare(native.DInput, reference.DInput)
			if e != nil {
				t.Fatal(e)
			}
			r.TimeError, e = reverseCompare(native.DTimes, reference.DTimes)
			if e != nil {
				t.Fatal(e)
			}
			if name == "endpoint" {
				endpointDX = append([]float32(nil), native.DInput...)
				r.DCondition = nil
				r.DTimes = nil // stopped values never become primary contribution
			} else {
				r.GradientFile = mode + "-" + name + ".f32"
				r.Manifest, r.ParametersError, e = reverseWriteGradients(filepath.Join(root, r.GradientFile), native.Parameters, reference.Parameters)
				if e != nil {
					t.Fatal(e)
				}
			}
			if e = tape.Close(); e != nil {
				t.Fatal("native close failure no retry", e)
			}
			if u := vulkan.VulkanMemoryStats(); u.Bytes != 0 || u.Allocations != 0 || u.InFlight || u.Uncertain || u.DeviceLost {
				t.Fatal("capture closure", u)
			}
			report.Results = append(report.Results, r)
			t.Logf("%s/%s condition max=%g signs=%d input max=%g parameters=%d max=%g RMS=%g opposite=%d over4e-6=%d native=%d RSS=%d stopped=%t", name, mode, r.ConditionError.Max, r.ConditionError.OppositeSigns, r.InputError.Max, r.ParametersError.Count, r.ParametersError.Max, r.ParametersError.RMS, r.ParametersError.OppositeSigns, r.ParametersError.Over4e6, usage.Bytes, rss, r.StoppedEndpoint)
		}
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	report.Final = vulkan.VulkanMemoryStats()
	if e = flowCaptureWrite(filepath.Join(root, "native-reverse.json"), report); e != nil {
		t.Fatal(e)
	}
	runtime.KeepAlive(in)
}
