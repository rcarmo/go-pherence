// Package zimage implements weight-free reference slices for Z-Image-Turbo.
// Transformer, text encoder and VAE execution are not implemented here.
package zimage

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/rcarmo/go-pherence/internal/checked"
	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

// FlowSchedule contains the shifted model timesteps and sigma transitions.
// Sigmas has one additional terminal zero. Step i uses Sigmas[i:i+2].
type FlowSchedule struct {
	Timesteps []float32
	Sigmas    []float32
}

const MaxFlowSteps = 1000

// DefaultFlowSchedule follows the pinned ZImagePipeline default sigma list
// and the pinned non-dynamic FlowMatchEulerDiscreteScheduler shift of 3.
// This does not model custom timesteps/sigmas or other scheduler modes.
func DefaultFlowSchedule(cfg loaderconfig.ZImageSchedulerConfig, steps int) (FlowSchedule, error) {
	if cfg.ClassName != "FlowMatchEulerDiscreteScheduler" || cfg.NumTrainTimesteps != 1000 || cfg.Shift != 3 || cfg.UseDynamicShifting || steps < 1 || steps > MaxFlowSteps {
		return FlowSchedule{}, fmt.Errorf("zimage: unsupported FlowMatch schedule")
	}
	sigmas := make([]float32, steps+1)
	timesteps := make([]float32, steps)
	for i := 0; i < steps; i++ {
		// Z-Image uses torch.linspace(1, 1/steps, steps). Evaluate
		// its mathematical progression and round the supplied sigma to F32;
		// high-step bitwise torch.linspace parity is not established.
		base := float32(1)
		if steps > 1 {
			base = float32(1 - float64(i)/float64(steps))
		}
		// The pinned scheduler converts supplied sigmas to NumPy float32
		// before its non-dynamic shift; keep intermediate arithmetic in F32.
		shift := float32(cfg.Shift)
		sigmas[i] = shift * base / (1 + (shift-1)*base)
		timesteps[i] = sigmas[i] * float32(cfg.NumTrainTimesteps)
		if !finite(sigmas[i]) || !finite(timesteps[i]) || sigmas[i] <= 0 || sigmas[i] > 1 {
			return FlowSchedule{}, fmt.Errorf("zimage: non-finite or invalid shifted timestep")
		}
	}
	return FlowSchedule{Timesteps: timesteps, Sigmas: sigmas}, nil
}

// EulerStepInto writes sample + (sigmaNext-sigma)*velocity. It rejects
// malformed/non-finite inputs before touching dst and permits exact in-place
// dst==sample or dst==velocity operation; partial overlap is rejected.
func EulerStepInto(dst, sample, velocity []float32, sigma, sigmaNext float32) error {
	if len(dst) == 0 || len(dst) != len(sample) || len(dst) != len(velocity) || !finite(sigma) || !finite(sigmaNext) || sigma < 0 || sigma > 1 || sigmaNext < 0 || sigmaNext > sigma {
		return fmt.Errorf("zimage: invalid Euler step shape or sigma")
	}
	if overlap(dst, sample) && &dst[0] != &sample[0] || overlap(dst, velocity) && &dst[0] != &velocity[0] {
		return fmt.Errorf("zimage: partial Euler buffer overlap")
	}
	// Reading one input while writing another overlapping input would change
	// later rows even if dst exactly aliases one of them.
	if overlap(sample, velocity) && &sample[0] != &velocity[0] && (overlap(dst, sample) || overlap(dst, velocity)) {
		return fmt.Errorf("zimage: partial Euler buffer overlap")
	}
	delta := sigmaNext - sigma
	for i := range sample {
		if !finite(sample[i]) || !finite(velocity[i]) || !finite(sample[i]+delta*velocity[i]) {
			return fmt.Errorf("zimage: non-finite Euler input or result")
		}
	}
	for i := range sample {
		dst[i] = sample[i] + delta*velocity[i]
	}
	return nil
}

func finite(x float32) bool { return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) }

func overlap(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	// Checked byte lengths before converting slice endpoints to uintptr.
	nA, okA := checked.MulInt(len(a), 4)
	nB, okB := checked.MulInt(len(b), 4)
	if !okA || !okB {
		return true
	}
	startA := uintptr(unsafe.Pointer(&a[0]))
	startB := uintptr(unsafe.Pointer(&b[0]))
	return startA <= startB && uintptr(nA) > startB-startA || startB < startA && uintptr(nB) > startA-startB
}
