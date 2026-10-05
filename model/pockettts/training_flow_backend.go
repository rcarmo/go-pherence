package pockettts

import (
	"context"
	"errors"
	"fmt"
	"runtime"

	train "github.com/rcarmo/go-pherence/backends/training"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

// TrainingFlowBackend evaluates a complete single-row flow forward/time-JVP and
// reverse-over-forward tape. Time embeddings and scalar loss orchestration stay
// on CPU. This does not imply GPU transformer/optimiser or full GPU training.
// A supplied tape executor must keep source parameters immutable during calls.
type TrainingFlowBackend interface {
	EvaluateFlow(model *FlowHeadCPU, condition, times, input []float32, timeIndex int, gv, gt []float32) (FlowDualTrainingResult, error)
}

// TrainingFlowAccumulator keeps parameter gradients resident. Accumulation is
// explicit for every call; endpoint/forward passes must set accumulate=false.
type TrainingFlowAccumulator interface {
	EvaluateFlowAccumulating(*FlowHeadCPU, []float32, []float32, []float32, int, []float32, []float32, bool, float32) (FlowDualTrainingResult, error)
}

func evaluateTrainingFlow(backend TrainingFlowBackend, m *FlowHeadCPU, c, t, x []float32, index int, gv, gt []float32, accumulate bool, scale float32) (FlowDualTrainingResult, error) {
	if resident, ok := backend.(TrainingFlowAccumulator); ok {
		return resident.EvaluateFlowAccumulating(m, c, t, x, index, gv, gt, accumulate, scale)
	}
	return backend.EvaluateFlow(m, c, t, x, index, gv, gt)
}

// FlowTapeExecutor retains intermediates on device. This caller-owned interface
// never initialises GPU runtimes, chooses devices or retries uncertain work.
type FlowTapeExecutor interface {
	ExecuteFlowTape(FlowDualTrainingTape) (FlowDualTrainingResult, error)
}

type TrainingFlowBackendIdentity interface{ TrainingFlowBackendName() string }

func (b *FlowTapeTrainingBackend) TrainingFlowBackendName() string {
	if b == nil {
		return "unknown"
	}
	if named, ok := b.Executor.(TrainingFlowBackendIdentity); ok {
		return named.TrainingFlowBackendName()
	}
	return "unknown"
}
func (e *PTXFlowTapeExecutor) TrainingFlowBackendName() string {
	if e == nil {
		return "unknown"
	}
	return "ptx-flow-hybrid-v1"
}
func (e *VulkanFlowTapeExecutor) TrainingFlowBackendName() string {
	if e == nil {
		return "unknown"
	}
	return "vulkan-flow-hybrid-v1"
}
func (b *FlowTapeTrainingBackend) Close() error {
	if b == nil {
		return nil
	}
	if owner, ok := b.Executor.(interface{ Close() error }); ok {
		return owner.Close()
	}
	return nil
}
func (e *PTXFlowTapeExecutor) Close() error {
	if e == nil {
		return nil
	}
	if e.terminal != nil {
		return fmt.Errorf("terminal PTX flow resources retained: %w", e.terminal)
	}
	return e.Operator.Close()
}
func (e *VulkanFlowTapeExecutor) Close() error {
	if e == nil {
		return nil
	}
	if e.terminal != nil {
		return fmt.Errorf("terminal Vulkan flow resources retained: %w", e.terminal)
	}
	return e.Operator.Close()
}

// trainingExecutionBackendName forbids combining unbound device scopes.
func trainingExecutionBackendName(trainer *FullTrainer) string {
	if resident, ok := trainer.OptimizerBackend.(*ResidentTrainingBackend); ok {
		if resident == nil || trainer.Flow.TrainingBackend != resident || trainer.FlowLM.TransformerBackend != resident || trainer.FlowLM.AttentionBackwardBackend != nil || !resident.weightOwnerMatches(trainer) || !resident.projectionOwnerMatches(trainer) || !resident.sequenceOwnerMatches(trainer) || !resident.eosOwnerMatches(trainer) || !resident.lossOwnerMatches(trainer) || !resident.branchOwnerMatches(trainer) || !resident.stepReductionOwnerMatches(trainer) {
			return "unknown-combined"
		}
		return resident.TrainingOptimizerBackendName()
	}
	if trainer.Flow.StepReductionBackend != nil {
		return "unknown-step"
	}
	if trainer.Flow.BranchBackend != nil {
		return "unknown-branch"
	}
	if trainer.Flow.LossBackend != nil {
		return "unknown-loss"
	}
	if trainer.FlowLM.EOSBackend != nil {
		return "unknown-eos"
	}
	if trainer.FlowLM.SequenceBackend != nil {
		return "unknown-sequence"
	}
	if trainer.FlowLM.ProjectionBackend != nil {
		return "unknown-projection"
	}
	if trainer.Weighting.TrainingBackend != nil {
		return "unknown-weighting"
	}
	if backend := trainer.OptimizerBackend; backend != nil {
		if trainer.FlowLM.AttentionBackwardBackend != nil || trainer.FlowLM.TransformerBackend != nil || trainer.Flow.TrainingBackend != nil {
			return "unknown-combined"
		}
		return backend.TrainingOptimizerBackendName()
	}
	if backend := trainer.FlowLM.TransformerBackend; backend != nil {
		if trainer.FlowLM.AttentionBackwardBackend != nil || trainer.Flow.TrainingBackend != nil {
			return "unknown-combined"
		}
		if named, ok := backend.(TrainingTransformerBackendIdentity); ok {
			return named.TrainingTransformerBackendName()
		}
		return "unknown-transformer"
	}
	attention := trainingAttentionBackendName(trainer.FlowLM.AttentionBackwardBackend)
	if trainer.Flow.TrainingBackend == nil {
		return attention
	}
	if attention != "simd" {
		return "unknown-combined"
	}
	if named, ok := trainer.Flow.TrainingBackend.(TrainingFlowBackendIdentity); ok {
		return named.TrainingFlowBackendName()
	}
	return "unknown"
}

type FlowTapeTrainingBackend struct {
	Executor FlowTapeExecutor
	MaxBytes int64
}

func (b *FlowTapeTrainingBackend) EvaluateFlow(model *FlowHeadCPU, condition, times, input []float32, timeIndex int, gv, gt []float32) (FlowDualTrainingResult, error) {
	if b == nil || b.Executor == nil {
		return FlowDualTrainingResult{}, fmt.Errorf("explicit flow executor required")
	}
	tape, err := model.BuildFlowDualTrainingTape(condition, times, input, timeIndex, gv, gt, b.MaxBytes)
	if err != nil {
		return FlowDualTrainingResult{}, err
	}
	return b.Executor.ExecuteFlowTape(tape)
}

// PTXFlowTapeExecutor is single-owner and explicit; it requires already loaded
// kernels. On device/cleanup errors it retains tape ownership and becomes
// terminal. Parameter/seed transfers are per invocation, not a resident model.
type PTXFlowTapeExecutor struct {
	Context  context.Context
	Operator *nvidia.TrainingF32
	MaxBytes int64
	terminal error
	retained []*nvidia.TrainingTape
}

func (e *PTXFlowTapeExecutor) ExecuteFlowTape(t FlowDualTrainingTape) (FlowDualTrainingResult, error) {
	values, err := e.executeTape(t.Tape)
	if err != nil {
		return FlowDualTrainingResult{}, err
	}
	return t.Decode(values)
}
func (e *PTXFlowTapeExecutor) executeTape(t train.Tape) (result [][]float32, err error) {
	if e == nil || e.Context == nil || e.Operator == nil || e.terminal != nil {
		return result, fmt.Errorf("nil/terminal PTX flow executor")
	}
	if err = e.Context.Err(); err != nil {
		return result, err
	}
	tape, err := nvidia.NewTrainingTape(e.Operator, t, e.MaxBytes)
	if err != nil {
		if tape != nil {
			e.retained = append(e.retained, tape)
			e.terminal = err
		}
		return result, err
	}
	defer func() {
		if closeErr := tape.Close(); closeErr != nil {
			e.retained = append(e.retained, tape)
			e.terminal = errors.Join(e.terminal, closeErr)
			err = errors.Join(err, closeErr)
			result = nil
		}
		runtime.KeepAlive(e.Operator)
	}()
	values, err := tape.Run()
	if err != nil {
		e.terminal = err
		return result, err
	}
	return values, nil
}

type VulkanFlowTapeExecutor struct {
	Context  context.Context
	Operator *vulkan.VkTrainingF32
	MaxBytes int64
	terminal error
	retained []*vulkan.VkTrainingTape
}

func (e *VulkanFlowTapeExecutor) ExecuteFlowTape(t FlowDualTrainingTape) (FlowDualTrainingResult, error) {
	values, err := e.executeTape(t.Tape)
	if err != nil {
		return FlowDualTrainingResult{}, err
	}
	return t.Decode(values)
}
func (e *VulkanFlowTapeExecutor) executeTape(t train.Tape) (result [][]float32, err error) {
	if e == nil || e.Context == nil || e.Operator == nil || e.terminal != nil {
		return result, fmt.Errorf("nil/terminal Vulkan flow executor")
	}
	if err = e.Context.Err(); err != nil {
		return result, err
	}
	tape, err := vulkan.NewVkTrainingTape(e.Context, e.Operator, t, e.MaxBytes)
	if err != nil {
		if tape != nil {
			e.retained = append(e.retained, tape)
			e.terminal = err
		}
		return result, err
	}
	defer func() {
		if closeErr := tape.Close(); closeErr != nil {
			e.retained = append(e.retained, tape)
			e.terminal = errors.Join(e.terminal, closeErr)
			err = errors.Join(err, closeErr)
			result = nil
		}
		runtime.KeepAlive(e.Operator)
	}()
	values, err := tape.Run(e.Context)
	if err != nil {
		e.terminal = err
		return result, err
	}
	return values, nil
}
