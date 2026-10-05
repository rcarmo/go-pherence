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
	"sync/atomic"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/backends/vulkan"
)

func pocketEfficiencyCompareP(t *testing.T, r *sharedCheckpointPReader, trainer *FullTrainer) error {
	t.Helper()
	if binary.LittleEndian.Uint64(r.header[4:12]) != 7 || r.count != len(trainer.params) {
		return fmt.Errorf("step-seven oracle association")
	}
	var raw [32 << 10]byte
	var maximum float64
	var count, above int64
	for i := 0; i < r.count; i++ {
		name, n, err := r.next()
		if err != nil {
			return err
		}
		got := trainer.params[name]
		if len(got) != n {
			return fmt.Errorf("parameter extent %s", name)
		}
		for off := 0; off < n; {
			c := min(n-off, len(raw)/4)
			if _, err = io.ReadFull(r.file, raw[:4*c]); err != nil {
				return err
			}
			for j := 0; j < c; j++ {
				want := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*j:]))
				x := got[off+j]
				if !isFinite(x) || !isFinite(want) {
					return fmt.Errorf("nonfinite parameter oracle")
				}
				d := math.Abs(float64(x) - float64(want))
				maximum = max(maximum, d)
				if d > 4e-6 {
					above++
				}
				count++
			}
			off += c
		}
	}
	t.Logf("matched step7 parameters values=%d max=%g above4e-6=%d", count, maximum, above)
	if count != 89449730 || above != 0 {
		return fmt.Errorf("step-seven parameter gate %d/%g", above, maximum)
	}
	return nil
}

type pocketEfficiencyPhase struct {
	Name           string
	Nanoseconds    int64
	AllocatedBytes uint64
	Allocations    uint64
}

// The whole invocation still gets CPU/heap profiles; phase labels identify
// restoration, cold step and serialisation without moving their cost elsewhere.
func pocketEfficiencyMeasure(ctx context.Context, name string, fn func() error) (pocketEfficiencyPhase, error) {
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	started := time.Now()
	var err error
	pprof.Do(ctx, pprof.Labels("phase", name), func(context.Context) { err = fn() })
	elapsed := time.Since(started)
	runtime.ReadMemStats(&after)
	return pocketEfficiencyPhase{name, elapsed.Nanoseconds(), after.TotalAlloc - before.TotalAlloc, after.Mallocs - before.Mallocs}, err
}

func TestPocketEfficiencyPhaseAccounting(t *testing.T) {
	called := false
	phase, err := pocketEfficiencyMeasure(context.Background(), "fixture", func() error { called = true; return io.ErrUnexpectedEOF })
	if !called || err != io.ErrUnexpectedEOF || phase.Name != "fixture" || phase.Nanoseconds < 0 {
		t.Fatal(phase, err)
	}
}

// Each fresh process restores the accepted step-six boundary, then performs
// one NEW step-seven update. Five processes per backend use the same source,
// corpus, sampling and options. Shared GPU wall timings are diagnostic only.
// Serialisation excludes publication/fsync; no repeated full checkpoint copies.
func TestPocketReleasedEfficiencySample(t *testing.T) {
	mode := os.Getenv("GO_PHERENCE_POCKET_EFFICIENCY_MODE")
	if mode == "" {
		t.Skip("explicit matched released efficiency sample")
	}
	if mode != "cpu" && mode != "native" {
		t.Fatal("efficiency mode")
	}
	output := os.Getenv("GO_PHERENCE_POCKET_EFFICIENCY_OUTPUT")
	if output == "" {
		t.Fatal("fresh retained sample directory required")
	}
	if _, err := os.Stat(filepath.Join(output, "sample.json")); !os.IsNotExist(err) {
		t.Fatal("sample already exists or unavailable", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()
	var peak atomic.Uint64
	var captureFailed atomic.Bool
	stop, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(10 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				rss, err := flowCaptureRSS()
				if err != nil || rss > 6<<30 {
					captureFailed.Store(true)
					cancel()
					return
				}
				for old := peak.Load(); rss > old; old = peak.Load() {
					if peak.CompareAndSwap(old, rss) {
						break
					}
				}
			}
		}
	}()
	stopped := false
	defer func() {
		if !stopped {
			close(stop)
			<-done
		}
	}()
	var phases []pocketEfficiencyPhase
	var trainer *FullTrainer
	var runner *TrainingRunner
	var backend *ResidentTrainingBackend
	var saved struct{ Identity TrainingRunIdentity }
	var corpus struct{ Rows []PreparedTrainingRow }
	var err error
	phase, err := pocketEfficiencyMeasure(ctx, "prepare-restore", func() error {
		lm, flow, w, _ := releasedTrainingModels(t)
		if err := flowCapturePinned(os.Getenv("GO_PHERENCE_POCKET_SHARED_CORPUS"), "004605224b39ee9637ceb30719c76d4d3c8650142d8df8746938bc5b96e80bb9", 1<<20, &corpus); err != nil {
			return err
		}
		if len(corpus.Rows) != 1 || corpus.Rows[0].Batch.Frames != 14 || corpus.Rows[0].Batch.VoiceFrames != 13 || len(corpus.Rows[0].Batch.TextTokens) != 1 {
			return fmt.Errorf("matched short-row geometry")
		}
		restore := os.Getenv("GO_PHERENCE_POCKET_EFFICIENCY_RESTORE")
		raw, err := os.ReadFile(filepath.Join(restore, "current.json"))
		if err != nil {
			return err
		}
		var pointer struct {
			Generation  string `json:"generation"`
			ModelSHA256 string `json:"model_sha256"`
			Step        uint64 `json:"step"`
		}
		if err = json.Unmarshal(raw, &pointer); err != nil {
			return err
		}
		pin := "ce712f632bf84048eb45b3fa8d5a130dc0895940a30173045745cabfd5bdac1c"
		if mode == "native" {
			pin = "8d2b653e87b7b0be14e0d04b236e321ec3dce2c275dc2ea9a9bbd410a6a67b97"
		}
		// Pins below are fixture provenance; numerical comparison governs outputs.
		if pointer.Step != 6 || !bundleGeneration(pointer.Generation) || pointer.ModelSHA256 != pin {
			return fmt.Errorf("accepted step-six pin mismatch")
		}
		raw, err = os.ReadFile(filepath.Join(restore, pointer.Generation+".run.json"))
		if err != nil {
			return err
		}
		if err = json.Unmarshal(raw, &saved); err != nil {
			return err
		}
		adam := DefaultAdamWConfig()
		adam.LearningRate = 2e-4
		trainer, err = NewFullTrainer(lm, flow, w, adam, .999)
		if err != nil {
			return err
		}
		if mode == "native" {
			if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
				return fmt.Errorf("pinned native runtime unavailable; no fallback")
			}
			stats := vulkan.VulkanMemoryStats()
			if stats.Bytes != 0 || stats.Allocations != 0 || stats.InFlight || stats.Uncertain || stats.DeviceLost {
				return fmt.Errorf("native owner not fresh")
			}
			if err = vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 2 << 30, MaxAllocations: 4096}); err != nil {
				return err
			}
			op, err := vulkan.NewVkTrainingF32(ctx)
			if err != nil {
				return err
			}
			backend = &ResidentTrainingBackend{Driver: vulkan.NewVulkanSharedWorkspaceDriver(op), Scope: "vulkan-resident-sharedworkspace-hybrid-v1", MaxResidentBytes: 2 << 30, MaxTapeBytes: 2 << 30}
			trainer.OptimizerBackend = backend
			lm.TransformerBackend = backend
			lm.ProjectionBackend = backend
			lm.SequenceBackend = backend
			lm.EOSBackend = backend
			flow.TrainingBackend = backend
			flow.LossBackend = backend
			flow.BranchBackend = backend
			flow.StepReductionBackend = backend
			w.TrainingBackend = backend
		}
		runner, err = NewTrainingRunner(trainer, saved.Identity, corpus.Rows, [6]uint64{42, 43, 44, 45, 46, 47}, TrainingRandomLimits{TrainingSampleLimits{375, 375 * 32}, 2*375*32 + 6*375}, 1, true)
		if err != nil {
			return err
		}
		return runner.Restore(restore, TrainingBundleLimits{MaxModelBytes: 2 << 30, MaxSidecarBytes: MaxTrainingRunStateBytes})
	})
	phases = append(phases, phase)
	if err != nil {
		t.Fatal("preparation failed; no retry", err)
	}
	var metrics TrainingStepMetrics
	phase, err = pocketEfficiencyMeasure(ctx, "cold-update", func() error { var e error; metrics, e = runner.Step(ctx); return e })
	phases = append(phases, phase)
	if err != nil {
		t.Fatal("update failed; owner retained; no retry", err)
	}
	if trainer.StepCount != 7 {
		t.Fatal("expected new step seven")
	}
	serialise := func(writer io.Writer) error {
		var state FullTrainingState
		var err error
		if backend != nil {
			state, err = backend.sharedCheckpointState(trainer)
		} else {
			state, err = trainer.State()
		}
		if err != nil {
			return err
		}
		return writeFullTrainingBinary(writer, state)
	}
	phase, err = pocketEfficiencyMeasure(ctx, "snapshot-serialise-discard", func() error { return serialise(io.Discard) })
	phases = append(phases, phase)
	if err != nil {
		t.Fatal("serialisation failure", err)
	}
	phase, err = pocketEfficiencyMeasure(ctx, "close", func() error {
		if backend != nil {
			return backend.Close()
		}
		return nil
	})
	phases = append(phases, phase)
	if err != nil {
		t.Fatal("closure failure; retained resources", err)
	}
	close(stop)
	<-done
	stopped = true
	if captureFailed.Load() || peak.Load() == 0 || peak.Load() > 6<<30 {
		t.Fatal("RSS capture/gate", peak.Load())
	}
	stats := vulkan.VulkanMemoryStats()
	if mode == "native" && (stats.Bytes != 0 || stats.Allocations != 0 || stats.InFlight || stats.Uncertain || stats.DeviceLost || stats.PeakBytes > 2<<30 || stats.PeakAllocations > 4096) {
		t.Fatal("native closure/resources", stats)
	}
	// Correctness/oracle storage is AFTER measured owner closure. It is still
	// retained in whole-process profiles, never counted as command timing/RSS.
	oracle := os.Getenv("GO_PHERENCE_POCKET_EFFICIENCY_ORACLE")
	if mode == "cpu" && os.Getenv("GO_PHERENCE_POCKET_EFFICIENCY_WRITE_ORACLE") == "1" {
		f, e := os.OpenFile(oracle, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if e != nil {
			t.Fatal(e)
		}
		e = serialise(f)
		syncErr, closeErr := f.Sync(), f.Close()
		if e != nil || syncErr != nil || closeErr != nil {
			t.Fatal(e, syncErr, closeErr)
		}
	}
	if oracle == "" {
		t.Fatal("independent CPU step-seven oracle required")
	}
	reader, e := sharedOpenCheckpointP(oracle)
	if e != nil {
		t.Fatal(e)
	}
	if e = pocketEfficiencyCompareP(t, reader, trainer); e != nil {
		reader.file.Close()
		t.Fatal(e)
	}
	if e = reader.file.Close(); e != nil {
		t.Fatal(e)
	}
	compareIntegratedEMA(t, oracle, trainer.EMA)
	report := struct {
		Mode                  string
		Step                  int
		Frames, Voice, Tokens int
		Phases                []pocketEfficiencyPhase
		Metrics               TrainingStepMetrics
		PeakRSS               uint64
		Native                vulkan.VulkanMemoryUsage
		Scope                 string
	}{mode, 7, 14, 13, 1, phases, metrics, peak.Load(), stats, "cold step after accepted step6; snapshot+serialise to discard; shared GPU diagnostic; excludes fsync/publication and post-close comparison"}
	if e = flowCaptureWrite(filepath.Join(output, "sample.json"), report); e != nil {
		t.Fatal(e)
	}
	t.Logf("matched mode=%s phases=%+v peakRSS=%d native=%+v", mode, phases, peak.Load(), stats)
}
