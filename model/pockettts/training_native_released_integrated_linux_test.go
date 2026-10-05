//go:build linux

package pockettts

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Actual released-size runner/update/checkpoint, behind explicit hardware opt-in.
// Independent saved CPU checkpoint is comparison-only and never supplies model
// values or gradients to native execution.
func TestPocketNativeReleasedIntegrated(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_RELEASED_INTEGRATED_NATIVE") != "1" {
		t.Skip("explicit released shared-workspace integrated qualification")
	}
	root := os.Getenv("GO_PHERENCE_POCKET_RELEASED_INTEGRATED_OUTPUT")
	if root == "" {
		t.Fatal("exclusive output directory")
	}
	in := allrowPrepare(t)
	w := NewDefaultLSDWeightMLP()
	var corpus struct{ Rows []PreparedTrainingRow }
	if e := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9", 1<<20, &corpus); e != nil {
		t.Fatal(e)
	}
	var saved struct{ Identity TrainingRunIdentity }
	if e := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CPU_RUN"), "e5c1bd21309fb893336028fe363cc192b88030727a2b310f3b0be869c1e8fdbd", 1<<16, &saved); e != nil {
		t.Fatal(e)
	}
	if os.Getenv("GO_PHERENCE_POCKET_RELEASED_INTEGRATED_LONG") == "1" {
		corpus.Rows[0] = integratedLongSyntheticRow(corpus.Rows[0])
	}
	// No copied condition fixtures are used by the runner; it recomputes forward.
	identity := saved.Identity
	identity.Backend = "vulkan-resident-sharedworkspace-hybrid-v1"
	lm, flow := in.LM, in.Flow
	trainer, e := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .999)
	if e != nil {
		t.Fatal(e)
	}
	stepBudget := 180 * time.Second
	if value := os.Getenv("GO_PHERENCE_POCKET_RELEASED_INTEGRATED_TIMEOUT_SECONDS"); value != "" {
		seconds, err := strconv.Atoi(value)
		if err != nil || seconds < 1 || seconds > 1800 {
			t.Fatal("explicit bounded qualification deadline")
		}
		stepBudget = time.Duration(seconds) * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), stepBudget)
	defer cancel()
	var peak atomic.Uint64
	var capHit atomic.Bool
	done := make(chan struct{})
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		ticker := time.NewTicker(10 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				rss, e := flowCaptureRSS()
				if e != nil {
					capHit.Store(true)
					cancel()
					return
				}
				for old := peak.Load(); rss > old; old = peak.Load() {
					if peak.CompareAndSwap(old, rss) {
						break
					}
				}
				if rss > 6<<30 {
					capHit.Store(true)
					cancel()
					return
				}
			}
		}
	}()
	defer func() { close(done); <-stopped }()
	sample := func(label string) {
		rss, e := flowCaptureRSS()
		if e != nil || rss > 6<<30 {
			t.Fatal("RSS admission", label, rss, e)
		}
		for old := peak.Load(); rss > old; old = peak.Load() {
			if peak.CompareAndSwap(old, rss) {
				break
			}
		}
		t.Logf("phase=%s RSS=%d native=%+v", label, rss, vulkan.VulkanMemoryStats())
	}
	sample("models-owned")
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("runtime no fallback")
	}
	if e = vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 2 << 30, MaxAllocations: 4096}); e != nil {
		t.Fatal(e)
	}
	op, e := vulkan.NewVkTrainingF32(ctx)
	if e != nil {
		t.Fatal(e)
	}
	backend := &ResidentTrainingBackend{Driver: vulkan.NewVulkanSharedWorkspaceDriver(op), Scope: identity.Backend, MaxResidentBytes: 2 << 30, MaxTapeBytes: 2 << 30}
	trainer.OptimizerBackend = backend
	flow.TrainingBackend = backend
	flow.LossBackend = backend
	flow.BranchBackend = backend
	flow.StepReductionBackend = backend
	lm.TransformerBackend = backend
	lm.ProjectionBackend = backend
	lm.SequenceBackend = backend
	lm.EOSBackend = backend
	w.TrainingBackend = backend
	runner, e := NewTrainingRunner(trainer, identity, corpus.Rows, [6]uint64{42, 43, 44, 45, 46, 47}, TrainingRandomLimits{TrainingSampleLimits{375, 375 * 32}, 2*375*32 + 6*375}, 1, true)
	if e != nil {
		t.Fatal(e)
	}
	sample("runner-admitted")
	expectedStep := 1
	if restore := os.Getenv("GO_PHERENCE_POCKET_RELEASED_INTEGRATED_RESTORE"); restore != "" {
		if e = runner.Restore(restore, TrainingBundleLimits{MaxModelBytes: 2 << 30, MaxSidecarBytes: MaxTrainingRunStateBytes}); e != nil {
			t.Fatal("fresh restore failure no retry", e)
		}
		expectedStep = 2
		sample("restore-complete")
	}
	defer func() {
		file, err := os.OpenFile(filepath.Join(root, "live-exit.pprof"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err == nil {
			err = pprof.Lookup("heap").WriteTo(file, 0)
			closeErr := file.Close()
			if err != nil || closeErr != nil {
				t.Log("live profile error", err, closeErr)
			}
		} else {
			t.Log("live profile open", err)
		}
		var mem runtime.MemStats
		runtime.ReadMemStats(&mem)
		t.Logf("profile exit peakRSS=%d capHit=%t HeapAlloc=%d HeapInuse=%d HeapSys=%d pooledWorkspace=%t native=%+v", peak.Load(), capHit.Load(), mem.HeapAlloc, mem.HeapInuse, mem.HeapSys, runner.workspace != nil && runner.workspace.zeroGradientOwner != nil, vulkan.VulkanMemoryStats())
		runtime.KeepAlive(runner)
		runtime.KeepAlive(trainer)
		runtime.KeepAlive(backend)
	}()
	metrics, e := runner.Step(ctx)
	if e != nil {
		t.Fatal("integrated step failed, retained owner/no retry", e)
	}
	sample("completed-step")
	if trainer.StepCount != expectedStep || backend.ValidateBoundary(trainer) != nil || capHit.Load() {
		t.Fatal("completed boundary/cap")
	}
	cpuHash := "d0d366f789a6ab0078bd0f1534d16a38dddb4e08a3193797be23c9f6b1794e5c"
	if expectedStep == 2 {
		cpuHash = "16391d753802996f7fded00af5f5c8391d699c69c48aa2563cc77d3659de7413"
	}
	if os.Getenv("GO_PHERENCE_POCKET_RELEASED_INTEGRATED_LONG") == "1" {
		cpuHash = os.Getenv("GO_PHERENCE_POCKET_LONG_CPU_SHA256")
		if len(cpuHash) != 64 {
			t.Fatal("long CPU reference provenance required")
		}
	}
	cpuPath := releasedAsset(t, "GO_PHERENCE_POCKET_SHARED_FULL_CPU", cpuHash, 1431217112)
	cpu, e := sharedOpenCheckpointP(cpuPath)
	if e != nil {
		t.Fatal(e)
	}
	defer cpu.file.Close()
	var raw [32768]byte
	var max float64
	var above, count int64
	outliers := map[string][]int{}
	for i := 0; i < cpu.count; i++ {
		name, n, e := cpu.next()
		if e != nil || len(trainer.params[name]) != n {
			t.Fatal("parameter inventory", name, e)
		}
		for off := 0; off < n; {
			c := min(n-off, len(raw)/4)
			if _, e = io.ReadFull(cpu.file, raw[:4*c]); e != nil {
				t.Fatal(e)
			}
			for j := 0; j < c; j++ {
				want := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*j:]))
				d := math.Abs(float64(trainer.params[name][off+j]) - float64(want))
				max = math.Max(max, d)
				if d > 4e-6 {
					above++
					if above <= 16 {
						index := off + j
						outliers[name] = append(outliers[name], index)
						moment := trainer.Moments[name]
						t.Logf("parameter outlier name=%s index=%d got=%.10g want=%.10g abs=%g nativeM=%.10g nativeV=%.10g", name, index, trainer.params[name][index], want, d, moment.M[index], moment.V[index])
					}
				}
				count++
			}
			off += c
		}
	}
	t.Logf("actual updated CPUparameters=%d max=%g above4e-6=%d", count, max, above)
	if above != 0 {
		logIntegratedCPUMoments(t, cpuPath, outliers)
		t.Fatal("actual independent parameter gate", above, max)
	}
	compareIntegratedEMA(t, cpuPath, trainer.EMA)
	bundle, e := runner.Checkpoint(root, fmt.Sprintf("step-%d", expectedStep), TrainingBundleLimits{MaxModelBytes: 2 << 30, MaxSidecarBytes: MaxTrainingRunStateBytes})
	if e != nil {
		t.Fatal("checkpoint failure no retry", e)
	}
	sample("checkpoint-complete")
	if os.Getenv("GO_PHERENCE_POCKET_RELEASED_INTEGRATED_SOAK") == "1" {
		for extra := 1; extra <= 3; extra++ {
			if _, e = runner.Step(ctx); e != nil {
				t.Fatal("soak step failure no retry", extra, e)
			}
			if trainer.StepCount != expectedStep+extra || backend.ValidateBoundary(trainer) != nil || capHit.Load() {
				t.Fatal("soak boundary", extra)
			}
			sample(fmt.Sprintf("soak-step-%d", trainer.StepCount))
		}
		if _, e = runner.Checkpoint(root, fmt.Sprintf("step-%d", trainer.StepCount), TrainingBundleLimits{MaxModelBytes: 2 << 30, MaxSidecarBytes: MaxTrainingRunStateBytes}); e != nil {
			t.Fatal("soak checkpoint", e)
		}
		sample("soak-checkpoint-complete")
	}
	if e = backend.Close(); e != nil {
		t.Fatal(e)
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	u := vulkan.VulkanMemoryStats()
	if u.Bytes != 0 || u.Allocations != 0 || u.PeakBytes > 2<<30 || u.PeakAllocations > 4096 || u.InFlight || u.Uncertain || u.DeviceLost {
		t.Fatal("closure", u)
	}
	if expectedStep == 2 && os.Getenv("GO_PHERENCE_POCKET_RELEASED_INTEGRATED_SOAK") != "1" {
		frozen, err := safetensors.Open(os.Getenv("GO_PHERENCE_POCKETTTS_VOICE_MODEL"))
		if err != nil {
			t.Fatal(err)
		}
		defer frozen.Close()
		exportPath := filepath.Join(root, "ema.safetensors")
		if err = ExportPocketSafetensors(exportPath, releasedConfig(t), trainer, frozen, true); err != nil {
			t.Fatal("EMA export", err)
		}
		verifyIntegratedExport(t, exportPath, trainer, frozen)
		sample("EMA-export-verified")
	}
	if capHit.Load() {
		t.Fatal("RSS sampled cap exceeded")
	}
	report := struct {
		Scope                string
		Metrics              TrainingStepMetrics
		Parameters, Above4e6 int64
		Max, PeakRSS         float64
		Checkpoint           string
		Final                vulkan.VulkanMemoryUsage
	}{fmt.Sprintf("released shared runner step %d/update/checkpoint/closure; independent CPU parameter/EMA gate; export on resumed step", expectedStep), metrics, count, above, max, float64(peak.Load()), fmt.Sprint(bundle), u}
	if e = flowCaptureWrite(filepath.Join(root, "integrated-step.json"), report); e != nil {
		t.Fatal(e)
	}
	_ = in // admitted pinned sampling/reference fixture, no native inputs from it
}

// Diagnostic-only bounded random-access reads from the already pinned oracle.
// Never load a full second optimizer snapshot into the measured resource peak.
func integratedCPUMomentPoints(path string, points map[string][]int, visit func(string, int, float32, float32)) error {
	r, err := sharedOpenCheckpointP(path)
	if err != nil {
		return err
	}
	defer r.file.Close()
	for group := 0; group < 3; group++ {
		if group > 0 {
			var raw [4]byte
			if _, err = io.ReadFull(r.file, raw[:]); err != nil {
				return err
			}
			r.count = int(binary.LittleEndian.Uint32(raw[:]))
			if r.count < 0 || r.count > maxBinaryTrainingTensors {
				return fmt.Errorf("diagnostic inventory")
			}
		}
		for i := 0; i < r.count; i++ {
			name, n, err := r.next()
			if err != nil {
				return err
			}
			start, err := r.file.Seek(0, io.SeekCurrent)
			if err != nil {
				return err
			}
			bytes := int64(n) * 4
			if group == 2 {
				bytes *= 2
				for _, index := range points[name] {
					if index < 0 || index >= n {
						return fmt.Errorf("diagnostic moment index")
					}
					var m, v [4]byte
					if _, err = r.file.ReadAt(m[:], start+int64(index)*4); err != nil {
						return err
					}
					if _, err = r.file.ReadAt(v[:], start+int64(n+index)*4); err != nil {
						return err
					}
					visit(name, index, math.Float32frombits(binary.LittleEndian.Uint32(m[:])), math.Float32frombits(binary.LittleEndian.Uint32(v[:])))
				}
			}
			if _, err = r.file.Seek(bytes, io.SeekCurrent); err != nil {
				return err
			}
		}
	}
	return nil
}

func logIntegratedCPUMoments(t *testing.T, path string, points map[string][]int) {
	t.Helper()
	if err := integratedCPUMomentPoints(path, points, func(name string, index int, m, v float32) {
		t.Logf("independent CPU outlier moments name=%s index=%d M=%.10g V=%.10g", name, index, m, v)
	}); err != nil {
		t.Fatal("outlier diagnostic", err)
	}
}

func TestIntegratedCPUMomentPoints(t *testing.T) {
	r, _, _ := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
	state, err := r.trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	name, n := state.Params[0].Name, len(state.Params[0].Values)
	m, v := make([]float32, n), make([]float32, n)
	m[n-1], v[n-1] = .125, .25
	for _, parameter := range state.Params {
		state.Adam = append(state.Adam, NamedAdamState{Name: parameter.Name, M: make([]float32, len(parameter.Values)), V: make([]float32, len(parameter.Values))})
	}
	state.Adam[0].M, state.Adam[0].V = m, v
	path := filepath.Join(t.TempDir(), "moments.bin")
	if err = SaveFullTrainingStateBinary(path, state); err != nil {
		t.Fatal(err)
	}
	seen := 0
	if err = integratedCPUMomentPoints(path, map[string][]int{name: {n - 1}}, func(got string, index int, m, v float32) {
		if got != name || index != n-1 || m != .125 || v != .25 {
			t.Fatal("moment stream association", got, index, m, v)
		}
		seen++
	}); err != nil || seen != 1 {
		t.Fatal("moment stream", err, seen)
	}
}

// Stream the independent checkpoint so reference state never adds a model-sized
// snapshot to the integrated resource peak. The already pinned file contains
// parameters, buffers, Adam pairs and EMA in that order.
func compareIntegratedEMA(t *testing.T, path string, ema map[string][]float32) {
	t.Helper()
	r, e := sharedOpenCheckpointP(path)
	if e != nil {
		t.Fatal(e)
	}
	defer r.file.Close()
	for group := 0; group < 3; group++ {
		if group > 0 {
			var b [4]byte
			if _, e = io.ReadFull(r.file, b[:]); e != nil {
				t.Fatal(e)
			}
			r.count = int(binary.LittleEndian.Uint32(b[:]))
		}
		for i := 0; i < r.count; i++ {
			_, n, e := r.next()
			if e != nil {
				t.Fatal(e)
			}
			bytes := int64(n) * 4
			if group == 2 {
				bytes *= 2
			}
			if _, e = r.file.Seek(bytes, io.SeekCurrent); e != nil {
				t.Fatal(e)
			}
		}
	}
	var b [4]byte
	if _, e = io.ReadFull(r.file, b[:]); e != nil {
		t.Fatal(e)
	}
	r.count = int(binary.LittleEndian.Uint32(b[:]))
	if r.count != len(ema) {
		t.Fatal("EMA inventory", r.count, len(ema))
	}
	var raw [32768]byte
	var maximum float64
	var count, above int64
	for i := 0; i < r.count; i++ {
		name, n, e := r.next()
		if e != nil || len(ema[name]) != n {
			t.Fatal("EMA extent", name, e)
		}
		for off := 0; off < n; {
			c := min(n-off, len(raw)/4)
			if _, e = io.ReadFull(r.file, raw[:4*c]); e != nil {
				t.Fatal(e)
			}
			for j := 0; j < c; j++ {
				want := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*j:]))
				d := math.Abs(float64(ema[name][off+j]) - float64(want))
				maximum = math.Max(maximum, d)
				if d > 4e-6 {
					above++
				}
				count++
			}
			off += c
		}
	}
	t.Logf("independent CPU EMA values=%d max=%g above4e-6=%d", count, maximum, above)
	if above != 0 {
		t.Fatal("independent EMA gate", above, maximum)
	}
}

func verifyIntegratedExport(t *testing.T, path string, trainer *FullTrainer, frozen *safetensors.File) {
	t.Helper()
	output, e := safetensors.Open(path)
	if e != nil {
		t.Fatal(e)
	}
	defer output.Close()
	flow, e := pocketFlowExportTensors(releasedConfig(t), trainer, true)
	if e != nil {
		t.Fatal(e)
	}
	var count int64
	var maximum float64
	compare := func(name string, want []float32) {
		raw, dtype, _, e := output.GetRaw(name)
		if e != nil || dtype != "F32" || len(raw) != 4*len(want) {
			t.Fatal("export extent", name, e)
		}
		for i, x := range want {
			got := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
			d := math.Abs(float64(got) - float64(x))
			if !isFinite(got) || d > 6e-8 {
				t.Fatalf("export %s[%d] got=%g want=%g", name, i, got, x)
			}
			maximum = math.Max(maximum, d)
			count++
		}
	}
	for _, tensor := range flow {
		compare(tensor.name, tensor.values)
	}
	mimi := expectedMimiStateShapes(releasedConfig(t))
	for name := range mimi {
		values, _, e := frozen.GetFloat32(name)
		if e != nil {
			t.Fatal(e)
		}
		compare(name, values)
	}
	if len(output.Names()) != len(flow)+len(mimi) {
		t.Fatal("export inventory")
	}
	t.Logf("EMA export tensors=%d storedValues=%d max=%g tolerance=6e-8", len(output.Names()), count, maximum)
}

// Labelled synthetic expansion; source recordings and frozen inputs stay intact.
func integratedLongSyntheticRow(row PreparedTrainingRow) PreparedTrainingRow {
	channels := len(row.Batch.NormalizedLatents) / row.Batch.Frames
	frames := 32
	if value := os.Getenv("GO_PHERENCE_POCKET_LONG_FRAMES"); value != "" {
		n, err := strconv.Atoi(value)
		if err != nil || n < 15 || n > 375 {
			panic("explicit synthetic long frame bound")
		}
		frames = n
	}
	target := make([]float32, frames*channels)
	for i := 0; i < (frames-1)*channels; i++ {
		target[i] = row.Batch.NormalizedLatents[i%(row.ValidFrames*channels)]
	}
	row.Batch.Frames = frames
	row.ValidFrames = frames - 1
	row.Batch.NormalizedLatents = target
	if os.Getenv("GO_PHERENCE_POCKET_LONG_JOINT_MAX") == "1" {
		voice := make([]float32, 62*channels)
		if len(row.Batch.VoiceLatents) == 0 {
			panic("joint synthetic reference requires original voice rows")
		}
		for i := range voice {
			voice[i] = row.Batch.VoiceLatents[i%len(row.Batch.VoiceLatents)]
		}
		row.Batch.VoiceFrames = 62
		row.Batch.VoiceLatents = voice
		row.Batch.TextTokens = make([]uint32, 512)
		for i := range row.Batch.TextTokens {
			row.Batch.TextTokens[i] = uint32(i + 1)
		}
	}
	return row
}

func TestPocketJointFixedLayerNative(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_JOINT_FIXED_LAYER_NATIVE") != "1" {
		t.Skip("explicit fixed layer native diagnostic")
	}
	root := os.Getenv("GO_PHERENCE_POCKET_JOINT_LAYER_INPUT")
	lm, _, _, _ := releasedTrainingModels(t)
	read := func(name string) []float32 {
		raw, err := os.ReadFile(filepath.Join(root, name))
		if err != nil || len(raw) != 950*lm.Hidden*4 {
			t.Fatal("layer boundary", name, err)
		}
		v := make([]float32, len(raw)/4)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		if !finiteF32(v) {
			t.Fatal("nonfinite capture")
		}
		return v
	}
	sequence, seed := read("cpu-layer2-input.f32"), read("cpu-layer2-seed.f32")
	stage, err := buildTransformerLayerStage(lm.Transformer, 2, sequence, seed, 950, false, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	binding := stage.ResidentBindings["flow_lm.transformer.layers.2.linear2.weight"]
	roots := []int{binding.Gradient}
	for _, cmd := range stage.Tape.Commands {
		if cmd.Output == binding.Gradient {
			roots = append(roots, cmd.Inputs[0], cmd.Inputs[3])
		}
	}
	if len(roots) != 3 {
		t.Fatal("FC2 gradient producers", roots)
	}
	stage.Tape.Outputs = append(stage.Tape.Outputs, roots[1:]...)
	tape, _, err := train.PruneTape(stage.Tape, roots, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("native layer no fallback")
	}
	if err = vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 2 << 30, MaxAllocations: 4096}); err != nil {
		t.Fatal(err)
	}
	op, err := vulkan.NewVkTrainingF32(ctx)
	if err != nil {
		t.Fatal(err)
	}
	owner, err := vulkan.NewVkTrainingTape(ctx, op, tape, 2<<30)
	if err != nil {
		t.Fatal(err)
	}
	values, err := owner.Run(ctx)
	if err != nil || len(values) != 3 {
		t.Fatal("fixed layer outputs", err)
	}
	index := 120489
	inDim := lm.Transformer.Layers[2].FC2.In
	outDim := lm.Hidden
	col, row := index%inDim, index/inDim
	var dot float32
	for q := 0; q < 950; q++ {
		dot = float32(math.FMA(float64(values[2][q*outDim+row]), float64(values[1][q*inDim+col]), float64(dot)))
	}
	raw, err := os.ReadFile(filepath.Join(root, "cpu-fixed-reduction.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cpu struct{ Observation sharedReductionObservation }
	if err = json.Unmarshal(raw, &cpu); err != nil {
		t.Fatal(err)
	}
	var maxX, maxSeed float64
	for q := 0; q < 950; q++ {
		maxX = max(maxX, math.Abs(float64(values[1][q*inDim+col])-float64(cpu.Observation.Inputs[q])))
		maxSeed = max(maxSeed, math.Abs(float64(values[2][q*outDim+row])-float64(cpu.Observation.Seeds[q])))
	}
	t.Logf("fixed layer nativeG=%g hostFMA=%g CPU=%g maxActivation=%g maxSeed=%g native=%+v", values[0][index], dot, cpu.Observation.CPUG, maxX, maxSeed, vulkan.VulkanMemoryStats())
	if !isFinite(values[0][index]) || math.Abs(float64(values[0][index])-float64(dot)) > 1e-10 {
		t.Fatal("native layer reduction differs from its inputs")
	}
	if err = owner.Close(); err != nil {
		t.Fatal(err)
	}
	if err = op.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestPocketJointFixedReductionNative(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_JOINT_FIXED_REDUCTION_NATIVE") != "1" {
		t.Skip("explicit fixed-vector native diagnostic")
	}
	raw, err := os.ReadFile(os.Getenv("GO_PHERENCE_POCKET_JOINT_REDUCTION_INPUT"))
	if err != nil || len(raw) > 1<<20 {
		t.Fatal("bounded fixed input", err)
	}
	var report struct{ Observation sharedReductionObservation }
	if err = json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	o := report.Observation
	if len(o.Inputs) != 950 || len(o.Seeds) != 950 || !finiteF32(o.Inputs) || !finiteF32(o.Seeds) {
		t.Fatal("fixed vector geometry")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("fixed native no fallback")
	}
	if err = vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 8 << 20, MaxAllocations: 64}); err != nil {
		t.Fatal(err)
	}
	op, err := vulkan.NewVkTrainingF32(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var owners []*vulkan.VkTensorArena
	makeTensor := func(values []float32, shape ...int) *vulkan.VkTensorF32 {
		n := 1
		for _, d := range shape {
			n *= d
		}
		a, err := vulkan.NewVkTensorArena(ctx, n*4)
		if err != nil {
			t.Fatal(err)
		}
		owners = append(owners, a)
		v, err := a.AllocF32(ctx, shape...)
		if err != nil {
			t.Fatal(err)
		}
		if values != nil {
			if err = v.Upload(ctx, values); err != nil {
				t.Fatal(err)
			}
		}
		return v
	}
	x, xt := makeTensor(o.Inputs, 950, 1), makeTensor(make([]float32, 950), 950, 1)
	w, gv, gt, out := makeTensor([]float32{0}, 1, 1), makeTensor(o.Seeds, 950, 1), makeTensor(make([]float32, 950), 950, 1), makeTensor(nil, 1, 1)
	stage, err := op.LinearDualBackwardStage(ctx, out, x, xt, w, gv, gt, 2)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := vulkan.NewVkF32Plan(ctx, []vulkan.VkF32Stage{stage})
	if err != nil {
		t.Fatal(err)
	}
	if err = plan.Run(ctx); err != nil {
		t.Fatal("single submitted fixed reduction no retry", err)
	}
	got := make([]float32, 1)
	err = out.Download(ctx, got)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || !isFinite(got[0]) || math.Abs(float64(got[0])-float64(o.CPUG)) > 1e-10 {
		t.Fatal("fixed reduction CPU drift", got, o.CPUG, o.FusedG, o.ReferenceG)
	}
	t.Logf("fixed native=%g CPU=%g fused=%g F64=%g", got[0], o.CPUG, o.FusedG, o.ReferenceG)
	if err = plan.Close(); err != nil {
		t.Fatal(err)
	}
	for i := len(owners) - 1; i >= 0; i-- {
		if err = owners[i].Close(); err != nil {
			t.Fatal(err)
		}
	}
	if err = op.Close(); err != nil {
		t.Fatal(err)
	}
	if stats := vulkan.VulkanMemoryStats(); stats.Bytes != 0 || stats.Allocations != 0 || stats.InFlight {
		t.Fatal("fixed diagnostic closure", stats)
	}
}

// Opt-in independent fixed-input diagnostic: observe one FC2 dot without
// feeding any observed value back into AD/optimizer or publishing a model.
func TestPocketReleasedJointReductionDiagnostic(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_JOINT_REDUCTION_DIAGNOSTIC") != "1" {
		t.Skip("explicit joint reduction diagnostic")
	}
	root := os.Getenv("GO_PHERENCE_POCKET_RELEASED_INTEGRATED_OUTPUT")
	if root == "" {
		t.Fatal("exclusive diagnostic output required")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	var corpus struct{ Rows []PreparedTrainingRow }
	if err := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9", 1<<20, &corpus); err != nil || len(corpus.Rows) != 1 {
		t.Fatal(err)
	}
	var saved struct{ Identity TrainingRunIdentity }
	if err := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CPU_RUN"), "e5c1bd21309fb893336028fe363cc192b88030727a2b310f3b0be869c1e8fdbd", 1<<16, &saved); err != nil {
		t.Fatal(err)
	}
	row := integratedLongSyntheticRow(corpus.Rows[0])
	if row.Batch.Frames != 375 || row.Batch.VoiceFrames != 62 || len(row.Batch.TextTokens) != 512 {
		t.Fatal("joint diagnostic geometry")
	}
	name, index := "flow_lm.transformer.layers.2.linear2.weight", 120489
	observer := &sharedReductionSensitivityBackend{points: []sharedSensitivityPoint{{Name: name, Index: index}}, capture: true, captureLayer: 2}
	lm.TransformerBackend = observer
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	run, err := NewTrainingRunState(saved.Identity, 1, [6]uint64{42, 43, 44, 45, 46, 47})
	if err != nil {
		t.Fatal(err)
	}
	proposal, err := run.PrepareRow()
	if err != nil {
		t.Fatal(err)
	}
	samples := runnerSampleBuffers(row.Batch.Frames, 32)
	recipe := TrainingRandomRecipe{Algorithm: TrainingRandomAlgorithm, OptionsSHA256: saved.Identity.OptionsSHA256, DistillProbability: 1, FlowBatchMultiplier: 1, StopGradient: "minimal", Normalize: true, PEqual: .75, EOSLossWeight: .1, LogitMean: .4, LogitStd: 1}
	if err = SampleTrainingInto(samples, proposal, row.Batch.Frames, row.ValidFrames, 32, recipe, TrainingRandomLimits{TrainingSampleLimits{375, 375 * 32}, 2*375*32 + 6*375}); err != nil {
		t.Fatal(err)
	}
	workspace, err := NewTrainingWorkspace(lm, flow, row.Batch)
	if err != nil {
		t.Fatal(err)
	}
	metrics, gradients, err := PocketTrainingStepInto(lm, flow, w, row.Batch, samples, DefaultTrainingStepConfig(), workspace)
	if err != nil {
		t.Fatal(err)
	}
	if err = (&TrainingRunner{trainer: trainer, maxNorm: 1}).clipGradients(gradients); err != nil {
		t.Fatal(err)
	}
	if err = trainer.Step(gradients); err != nil {
		t.Fatal(err)
	}
	if len(observer.observations) != 1 {
		t.Fatal("observation inventory")
	}
	report := struct {
		Observation     sharedReductionObservation
		Metrics         TrainingStepMetrics
		Parameter, M, V float32
	}{observer.observations[0], metrics, trainer.params[name][index], trainer.Moments[name].M[index], trainer.Moments[name].V[index]}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "cpu-fixed-reduction.json")
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, err = file.Write(data)
	closeErr := file.Close()
	if err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	for _, capture := range []struct {
		name   string
		values []float32
	}{{"cpu-layer2-input.f32", observer.sequence}, {"cpu-layer2-seed.f32", observer.seed}} {
		if len(capture.values) != 950*lm.Hidden {
			t.Fatal("captured layer boundary extent")
		}
		f, err := os.OpenFile(filepath.Join(root, capture.name), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			t.Fatal(err)
		}
		err = binary.Write(f, binary.LittleEndian, capture.values)
		closeErr := f.Close()
		if err != nil || closeErr != nil {
			t.Fatal(err, closeErr)
		}
	}
	t.Logf("fixed CPU point parameter=%g M=%g V=%g CPUG=%g sequential=%g fused=%g F64=%g", report.Parameter, report.M, report.V, report.Observation.CPUG, report.Observation.SequentialG, report.Observation.FusedG, report.Observation.ReferenceG)
}

func TestPocketReleasedLongIndependentCPU(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_LONG_CPU_REFERENCE") != "1" {
		t.Skip("explicit long synthetic independent SIMD reference")
	}
	root := os.Getenv("GO_PHERENCE_POCKET_RELEASED_INTEGRATED_OUTPUT")
	if root == "" {
		t.Fatal("exclusive output required")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	var corpus struct{ Rows []PreparedTrainingRow }
	if err := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9", 1<<20, &corpus); err != nil || len(corpus.Rows) != 1 {
		t.Fatal(err)
	}
	var saved struct{ Identity TrainingRunIdentity }
	if err := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CPU_RUN"), "e5c1bd21309fb893336028fe363cc192b88030727a2b310f3b0be869c1e8fdbd", 1<<16, &saved); err != nil {
		t.Fatal(err)
	}
	row := integratedLongSyntheticRow(corpus.Rows[0])
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := NewTrainingRunner(trainer, saved.Identity, []PreparedTrainingRow{row}, [6]uint64{42, 43, 44, 45, 46, 47}, TrainingRandomLimits{TrainingSampleLimits{375, 375 * 32}, 2*375*32 + 6*375}, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	step := 1
	if restore := os.Getenv("GO_PHERENCE_POCKET_LONG_CPU_RESTORE"); restore != "" {
		if err = runner.Restore(restore, TrainingBundleLimits{MaxModelBytes: 2 << 30, MaxSidecarBytes: MaxTrainingRunStateBytes}); err != nil {
			t.Fatal(err)
		}
		step = 2
	}
	metrics, err := runner.Step(ctx)
	if err != nil {
		t.Fatal(err)
	}
	binding, err := runner.Checkpoint(root, fmt.Sprintf("step-%d", step), TrainingBundleLimits{MaxModelBytes: 2 << 30, MaxSidecarBytes: MaxTrainingRunStateBytes})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("labelled synthetic CPU frames%d valid%d voice%d tokens%d checkpoint=%+v metrics=%+v", row.Batch.Frames, row.ValidFrames, row.Batch.VoiceFrames, len(row.Batch.TextTokens), binding, metrics)
}
