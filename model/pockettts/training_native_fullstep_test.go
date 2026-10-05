//go:build linux

package pockettts

// Explicit opt-in native qualification. No runtime fallback, downloads/assets
// fetch or default device execution. Captured G is test-only diagnostic I/O.
import (
	"context"
	"os"
	"testing"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

type pocketNativeCapture struct{ gradients [][]float32 }

func (c *pocketNativeCapture) capture(ctx context.Context, g train.ResidentBuffer, n int) error {
	v := make([]float32, n)
	if e := g.Download(ctx, v); e != nil {
		return e
	}
	c.gradients = append(c.gradients, v)
	return nil
}

type pocketNativePTX struct {
	*nvidia.PTXResidentDriver
	capture *pocketNativeCapture
}

func (d *pocketNativePTX) UpdateScaled(ctx context.Context, p, g, m, v, ema, clip train.ResidentBuffer, n int, s train.AdamWStep) error {
	if e := d.capture.capture(ctx, g, n); e != nil {
		return e
	}
	return d.PTXResidentDriver.UpdateScaled(ctx, p, g, m, v, ema, clip, n, s)
}

type pocketNativeVulkan struct {
	*vulkan.VulkanResidentDriver
	capture *pocketNativeCapture
}

func (d *pocketNativeVulkan) UpdateScaled(ctx context.Context, p, g, m, v, ema, clip train.ResidentBuffer, n int, s train.AdamWStep) error {
	if e := d.capture.capture(ctx, g, n); e != nil {
		return e
	}
	return d.VulkanResidentDriver.UpdateScaled(ctx, p, g, m, v, ema, clip, n, s)
}

func TestPocketNativeSharedFullStepPyTorch(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_NATIVE_FULLSTEP") != "vulkan-shared" {
		t.Skip("set GO_PHERENCE_POCKET_NATIVE_FULLSTEP=vulkan-shared")
	}
	testPocketNativeFullStep(t, "vulkan-shared")
}
func TestPocketNativeFullStepPyTorch(t *testing.T) {
	backend := os.Getenv("GO_PHERENCE_POCKET_NATIVE_FULLSTEP")
	if backend == "" {
		t.Skip("set GO_PHERENCE_POCKET_NATIVE_FULLSTEP=ptx or vulkan for explicit tiny native full-step qualification")
	}
	if backend != "ptx" && backend != "vulkan" {
		t.Fatal("unknown native backend", backend)
	}
	testPocketNativeFullStep(t, backend)
}
func testPocketNativeFullStep(t *testing.T, backend string) {
	x, o := loadPocketJoinFixture(t) // admit fixture before runtime initialisation
	ctx := context.Background()
	capture := &pocketNativeCapture{}
	var driver train.ResidentDriver
	var closeKernels func() error
	switch backend {
	case "ptx":
		if os.Getenv("GO_PHERENCE_DISABLE_NVIDIA") != "" {
			t.Fatal("selected PTX backend disabled")
		}
		if !nvidia.Init() {
			t.Fatal("PTX runtime unavailable; no fallback")
		}
		op, e := nvidia.NewTrainingF32()
		if e != nil {
			t.Fatal(e)
		}
		driver = &pocketNativePTX{PTXResidentDriver: &nvidia.PTXResidentDriver{Operator: op}, capture: capture}
		closeKernels = op.Close
	case "vulkan", "vulkan-shared":
		if os.Getenv("GO_PHERENCE_DISABLE_VULKAN") != "" {
			t.Fatal("selected Vulkan backend disabled")
		}
		if !vulkan.VulkanInit() {
			t.Fatal("Vulkan runtime unavailable; no fallback")
		}
		op, e := vulkan.NewVkTrainingF32(ctx)
		if e != nil {
			t.Fatal(e)
		}
		if backend == "vulkan-shared" {
			// Preserve exact driver identity for guarded access tokens. No wrapper.
			driver = vulkan.NewVulkanSharedWorkspaceDriver(op)
		} else {
			driver = &pocketNativeVulkan{VulkanResidentDriver: &vulkan.VulkanResidentDriver{Operator: op}, capture: capture}
		}
		closeKernels = op.Close
	}
	lm, flow, w := trainingStepModelsFromOracle(t, trainingStepOracle{Frames: 3, VoiceFrames: 2, Hidden: 4, LatentDim: 2, Vocabulary: 6, Parameters: x.parameters})
	a := o.AdamW
	adam := DefaultAdamWConfig()
	adam.LearningRate, adam.Beta1, adam.Beta2, adam.Epsilon, adam.WeightDecay = a.LearningRate, a.Beta1, a.Beta2, a.Epsilon, a.WeightDecay
	trainer, e := NewFullTrainer(lm, flow, w, adam, a.EMADecay)
	if e != nil {
		t.Fatal(e)
	}
	scope := backend + "-resident-devicehandoff-hybrid-v1"
	if backend == "vulkan-shared" {
		scope = "vulkan-resident-sharedworkspace-hybrid-v1"
	}
	b := &ResidentTrainingBackend{Driver: driver, Scope: scope, MaxResidentBytes: 1 << 20, MaxTapeBytes: 1 << 20}
	trainer.OptimizerBackend = b
	flow.TrainingBackend = b
	lm.TransformerBackend = b
	w.TrainingBackend = b
	lm.ProjectionBackend = b
	lm.SequenceBackend = b
	lm.EOSBackend = b
	flow.LossBackend = b
	flow.BranchBackend = b
	flow.StepReductionBackend = b
	start := time.Now()
	if e = b.PrepareTrainingStep(ctx, trainer); e != nil {
		t.Fatal("prepare", e)
	}
	metrics, g, e := PocketTrainingStep(lm, flow, w, x.batch, TrainingStepSamples{Mask: x.mask, Noise: x.noise, DiagonalTime: x.diagonal, DistillS: x.s, DistillT: x.end}, x.cfg)
	if e != nil {
		t.Fatal("native fullstep; owner retained on uncertain failure:", e)
	}
	var derivatives, gradients, params, ema, moments pocketAdapterErrors
	for n, v := range map[string]float64{"flow_diag": metrics.FlowDiagonal, "flow_distill": metrics.FlowDistill, "flow_loss": metrics.FlowLoss, "eos_loss": metrics.EOS, "loss": metrics.Loss} {
		if !finite64(v) {
			t.Fatal("nonfinite metric", n)
		}
		assertClose64(t, "native "+n, v, o.Metrics[n], 3e-5)
	}
	derivatives.compare(t, "audio", g.Inputs.NormalizedLatents, o.DNormalizedLatents, 5e-5)
	derivatives.compare(t, "voice", g.Inputs.VoiceLatents, o.DVoiceLatents, 5e-5)
	derivatives.compare(t, "diagonal logvar", g.DiagonalLogVariance[:2], o.DDiagonalLogVariance, 5e-5)
	derivatives.compare(t, "distill logvar", g.DistillLogVariance[:2], o.DDistillLogVariance, 5e-5)
	if g.DiagonalLogVariance[2] != 0 || g.DistillLogVariance[2] != 0 {
		t.Fatal("padded derivative")
	}
	// The device-owned host roots must stay empty-valued placeholders.
	for n, v := range trainingStepGradientMap(g) {
		for _, z := range v {
			if z != 0 {
				t.Fatal("native gradient unexpectedly on host", n)
			}
		}
	}
	if e = b.UpdateTrainer(ctx, trainer, g, 1e20); e != nil {
		t.Fatal("native update; owner retained:", e)
	}
	if backend != "vulkan-shared" && len(capture.gradients) != len(trainer.names) {
		t.Fatal("native update gradient inventory")
	}
	for i, n := range trainer.names {
		if backend != "vulkan-shared" {
			gradients.compare(t, n, capture.gradients[i], o.Gradients[n], 8e-5)
		}
		params.compare(t, n, trainer.params[n], a.ParametersAfterStep[n], 4e-6)
		ema.compare(t, n, trainer.EMA[n], a.EMAAfterStep[n], 4e-6)
		wm, wv := make([]float32, len(o.Gradients[n])), make([]float32, len(o.Gradients[n]))
		for j, z := range o.Gradients[n] {
			wm[j] = (1 - a.Beta1) * z
			wv[j] = (1 - a.Beta2) * z * z
		}
		moments.compare(t, n+" M", trainer.Moments[n].M, wm, 4e-6)
		moments.compare(t, n+" V", trainer.Moments[n].V, wv, 4e-6)
	}
	if trainer.StepCount != 1 || b.ValidateBoundary(trainer) != nil {
		t.Fatal("native completed boundary")
	}
	derivatives.log(t, "native derivatives")
	if backend != "vulkan-shared" {
		gradients.log(t, "native device G")
	} else {
		t.Log("shared G stays device-side; independently pinned M/V and P/EMA qualify its update contribution")
	}
	params.log(t, "native PyTorch P")
	ema.log(t, "native PyTorch EMA")
	moments.log(t, "native oracle-derived M/V")
	t.Logf("backend=%s complete-update elapsed=%s; correctness timing only (legacy G capture; shared G remains device-side)", backend, time.Since(start))
	if e = b.Close(); e != nil {
		t.Fatal("native close", e)
	}
	if e = closeKernels(); e != nil {
		t.Fatal("native kernel close", e)
	}
}
