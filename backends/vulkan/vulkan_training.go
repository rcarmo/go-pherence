package vulkan

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"math"

	train "github.com/rcarmo/go-pherence/backends/training"
)

//go:embed shaders/training_adamw_f32.spv
var spirvTrainingAdamW []byte

//go:embed shaders/training_linear_backward_f32.spv
var spirvTrainingLinearBackward []byte

//go:embed shaders/training_silu_backward_f32.spv
var spirvTrainingSiLUBackward []byte

//go:embed shaders/training_layernorm_backward_f32.spv
var spirvTrainingLayerNormBackward []byte

//go:embed shaders/training_attention_backward_f32.spv
var spirvTrainingAttentionBackward []byte

//go:embed shaders/training_silu_dual_backward_f32.spv
var spirvTrainingSiLUDualBackward []byte

//go:embed shaders/training_linear_dual_backward_f32.spv
var spirvTrainingLinearDualBackward []byte

//go:embed shaders/training_norm_dual_backward_f32.spv
var spirvTrainingNormDualBackward []byte

//go:embed shaders/training_mod_dual_backward_f32.spv
var spirvTrainingModDualBackward []byte

//go:embed shaders/training_norm_dual_forward_f32.spv
var spirvTrainingNormDualForward []byte

//go:embed shaders/training_linear_dual_forward_f32.spv
var spirvTrainingLinearDualForward []byte

//go:embed shaders/training_silu_dual_forward_f32.spv
var spirvTrainingSiLUDualForward []byte

//go:embed shaders/training_mod_dual_forward_f32.spv
var spirvTrainingModDualForward []byte

//go:embed shaders/training_vector_f32.spv
var spirvTrainingVector []byte

//go:embed shaders/training_silu_dual_adjoint_f32.spv
var spirvTrainingSiLUDualAdjoint []byte

//go:embed shaders/training_rope_f32.spv
var spirvTrainingRoPE []byte

//go:embed shaders/training_attention_forward_f32.spv
var spirvTrainingAttentionForward []byte

//go:embed shaders/training_gelu_f32.spv
var spirvTrainingGELU []byte

//go:embed shaders/training_row_f32.spv
var spirvTrainingRow []byte

//go:embed shaders/training_norm_stats_f32.spv
var spirvTrainingNormStats []byte

//go:embed shaders/training_relu_f32.spv
var spirvTrainingReLU []byte

//go:embed shaders/training_embedding_f32.spv
var spirvTrainingEmbedding []byte

//go:embed shaders/training_embedding_accumulate_f32.spv
var spirvTrainingEmbeddingAccumulate []byte

//go:embed shaders/training_time_embedding_f32.spv
var spirvTrainingTimeEmbedding []byte

//go:embed shaders/training_eos_f32.spv
var spirvTrainingEOS []byte

//go:embed shaders/training_eos_admission_f32.spv
var spirvTrainingEOSAdmission []byte

//go:embed shaders/training_lsd_loss_f32.spv
var spirvTrainingLSDLoss []byte

//go:embed shaders/training_global_norm_f32.spv
var spirvTrainingGlobalNorm []byte

//go:embed shaders/training_adamw_scaled_f32.spv
var spirvTrainingAdamWScaled []byte

// VkTrainingF32 owns experimental F32 training pipelines. It never initialises
// Vulkan itself. Callers must keep finite gradients, nonnegative second moments
// and all state resident until dispatch completion. Device numerical/performance
// qualification is separate from SPIR-V compilation and offline admission tests.
// A failed submitted update requires coherent checkpoint recovery, never retry.
type VkTrainingF32 struct{ adam, linear, silu, norm, attention, siluDual, linearDual, normDual, modDual, normDualForward, linearDualForward, siluDualForward, modDualForward, vector, siluDualAdjoint, rope, attentionForward, gelu, row, normStats, relu, embedding, embeddingAccumulate, timeEmbedding, eos, eosAdmission, lsdLoss, globalNorm, adamScaled *VkComputeKernel }

func NewVkTrainingF32(ctx context.Context) (*VkTrainingF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	return newVkTrainingF32Locked(vkKernelCreateLocked)
}

// Creation seam is a local function argument, never a mutable global. If native
// cleanup fails, return a nonnil partial owner with the error; callers must
// retain that owner for explicit disposition, not discard remaining handles.
func newVkTrainingF32Locked(create func([]byte, int, int) (*VkComputeKernel, error)) (*VkTrainingF32, error) {
	op := &VkTrainingF32{}
	for _, spec := range []struct {
		slot          **VkComputeKernel
		code          []byte
		buffers, push int
	}{
		{&op.adam, spirvTrainingAdamW, 5, 40}, {&op.linear, spirvTrainingLinearBackward, 4, 16}, {&op.silu, spirvTrainingSiLUBackward, 3, 4},
		{&op.norm, spirvTrainingLayerNormBackward, 5, 12}, {&op.attention, spirvTrainingAttentionBackward, 7, 24}, {&op.siluDual, spirvTrainingSiLUDualBackward, 6, 4},
		{&op.linearDual, spirvTrainingLinearDualBackward, 6, 16}, {&op.normDual, spirvTrainingNormDualBackward, 7, 16}, {&op.modDual, spirvTrainingModDualBackward, 7, 12}, {&op.normDualForward, spirvTrainingNormDualForward, 6, 20}, {&op.linearDualForward, spirvTrainingLinearDualForward, 5, 16}, {&op.siluDualForward, spirvTrainingSiLUDualForward, 3, 8}, {&op.modDualForward, spirvTrainingModDualForward, 5, 12}, {&op.vector, spirvTrainingVector, 3, 20}, {&op.siluDualAdjoint, spirvTrainingSiLUDualAdjoint, 5, 8}, {&op.rope, spirvTrainingRoPE, 3, 16}, {&op.attentionForward, spirvTrainingAttentionForward, 5, 24}, {&op.gelu, spirvTrainingGELU, 3, 8}, {&op.row, spirvTrainingRow, 3, 20}, {&op.normStats, spirvTrainingNormStats, 2, 8}, {&op.relu, spirvTrainingReLU, 3, 8}, {&op.embedding, spirvTrainingEmbedding, 3, 16}, {&op.timeEmbedding, spirvTrainingTimeEmbedding, 5, 8}, {&op.eos, spirvTrainingEOS, 3, 16}, {&op.lsdLoss, spirvTrainingLSDLoss, 4, 8}, {&op.globalNorm, spirvTrainingGlobalNorm, 3, 16}, {&op.adamScaled, spirvTrainingAdamWScaled, 6, 40}, {&op.embeddingAccumulate, spirvTrainingEmbeddingAccumulate, 4, 16}, {&op.eosAdmission, spirvTrainingEOSAdmission, 4, 12},
	} {
		k, err := create(spec.code, spec.buffers, spec.push)
		*spec.slot = k
		if err != nil {
			var closeErr error
			for _, owned := range []*VkComputeKernel{op.adam, op.linear, op.silu, op.norm, op.attention, op.siluDual, op.linearDual, op.normDual, op.modDual, op.normDualForward, op.linearDualForward, op.siluDualForward, op.modDualForward, op.vector, op.siluDualAdjoint, op.rope, op.attentionForward, op.gelu, op.row, op.normStats, op.relu, op.embedding, op.timeEmbedding, op.eos, op.lsdLoss, op.globalNorm, op.adamScaled, op.embeddingAccumulate, op.eosAdmission} {
				closeErr = errors.Join(closeErr, owned.closeLocked())
			}
			if closeErr != nil {
				return op, errors.Join(err, closeErr)
			}
			return nil, err
		}
	}
	return op, nil
}

// AttentionReady checks owned host metadata only. Caller still owns native
// initialisation/device health and must not race Close or replace Vulkan state.
func (op *VkTrainingF32) AttentionReady() bool {
	return op != nil && op.attention != nil && !op.attention.closed
}
func (op *VkTrainingF32) Close() error {
	if op == nil {
		return nil
	}
	// Failed closes retain native ownership, matching kernel lifetime semantics.
	return errors.Join(op.adam.Close(), op.linear.Close(), op.silu.Close(), op.norm.Close(), op.attention.Close(), op.siluDual.Close(), op.linearDual.Close(), op.normDual.Close(), op.modDual.Close(), op.normDualForward.Close(), op.linearDualForward.Close(), op.siluDualForward.Close(), op.modDualForward.Close(), op.vector.Close(), op.siluDualAdjoint.Close(), op.rope.Close(), op.attentionForward.Close(), op.gelu.Close(), op.row.Close(), op.normStats.Close(), op.relu.Close(), op.embedding.Close(), op.timeEmbedding.Close(), op.eos.Close(), op.lsdLoss.Close(), op.globalNorm.Close(), op.adamScaled.Close(), op.embeddingAccumulate.Close(), op.eosAdmission.Close())
}

// AdamWStage mutates parameter/M/V and, when enabled, EMA. All five exact
// vector ranges must be disjoint, including the read-only gradient. Disabled
// EMA still binds distinct owned scratch (it is neither read nor written).
func (op *VkTrainingF32) AdamWStage(ctx context.Context, parameter, gradient, m, v, ema *VkTensorF32, p train.AdamWStep) (VkF32Stage, error) {
	if err := p.Validate(); err != nil {
		return VkF32Stage{}, err
	}
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.adam == nil {
		return VkF32Stage{}, fmt.Errorf("uninitialised Vulkan training operator")
	}
	tensors := []*VkTensorF32{parameter, gradient, m, v, ema}
	if parameter == nil || parameter.rank != 1 || parameter.shape[0] <= 0 || parameter.shape[0] > 1<<26 {
		return VkF32Stage{}, fmt.Errorf("AdamW vector ceiling")
	}
	n := parameter.shape[0]
	for _, t := range tensors {
		if t == nil || t.rank != 1 || t.shape[0] != n {
			return VkF32Stage{}, fmt.Errorf("AdamW vector shape mismatch")
		}
	}
	words := []uint32{uint32(n), math.Float32bits(p.LearningRate), math.Float32bits(p.Beta1), math.Float32bits(p.Beta2), math.Float32bits(p.Epsilon), math.Float32bits(p.WeightDecay), math.Float32bits(p.Bias1), math.Float32bits(p.Bias2), math.Float32bits(p.EMADecay), math.Float32bits(p.GradientScale)}
	return trainingStageLocked(op.adam, tensors, []uint64{uint64(n) * 4, uint64(n) * 4, uint64(n) * 4, uint64(n) * 4, uint64(n) * 4}, []int{0, 2, 3, 4}, uint32(n), words)
}

// LinearBackwardStage returns one overwrite stage: mode0=dX,1=dW,2=dBias.
// Inputs may alias each other; destination must be disjoint from all inputs.
// Use separate stages and explicit branch-gradient accumulation as needed.
func (op *VkTrainingF32) LinearBackwardStage(ctx context.Context, dst, x, w, dy *VkTensorF32, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.linear == nil || mode > 2 || x == nil || w == nil || dy == nil || dst == nil || x.rank != 2 || w.rank != 2 || dy.rank != 2 {
		return VkF32Stage{}, fmt.Errorf("invalid Vulkan linear backward operator/rank/mode")
	}
	m, k, n := x.shape[0], x.shape[1], w.shape[0]
	if err := train.LinearShape(m, k, n); err != nil {
		return VkF32Stage{}, err
	}
	if w.shape[1] != k || dy.shape[0] != m || dy.shape[1] != n {
		return VkF32Stage{}, fmt.Errorf("linear backward input mismatch")
	}
	count := n
	switch mode {
	case 0:
		count = m * k
		if dst.rank != 2 || dst.shape[0] != m || dst.shape[1] != k {
			return VkF32Stage{}, fmt.Errorf("linear dX shape")
		}
	case 1:
		count = n * k
		if dst.rank != 2 || dst.shape[0] != n || dst.shape[1] != k {
			return VkF32Stage{}, fmt.Errorf("linear dW shape")
		}
	case 2:
		if dst.rank != 1 || dst.shape[0] != n {
			return VkF32Stage{}, fmt.Errorf("linear dBias shape")
		}
	}
	return trainingStageLocked(op.linear, []*VkTensorF32{x, w, dy, dst}, []uint64{uint64(m) * uint64(k) * 4, uint64(n) * uint64(k) * 4, uint64(m) * uint64(n) * 4, uint64(count) * 4}, []int{3}, uint32(count), []uint32{uint32(m), uint32(k), uint32(n), mode})
}

// SiLUBackwardStage consumes pre-activation X and upstream dY, writing dX.
// No output/input alias is admitted, including exact in-place use.
func (op *VkTrainingF32) SiLUBackwardStage(ctx context.Context, dst, x, dy *VkTensorF32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.silu == nil || dst == nil || x == nil || dy == nil || dst.rank != 1 || x.rank != 1 || dy.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("invalid SiLU backward rank/operator")
	}
	n := x.shape[0]
	if n <= 0 || n > 1<<26 || dy.shape[0] != n || dst.shape[0] != n {
		return VkF32Stage{}, fmt.Errorf("SiLU backward vector shape")
	}
	return trainingStageLocked(op.silu, []*VkTensorF32{x, dy, dst}, []uint64{uint64(n) * 4, uint64(n) * 4, uint64(n) * 4}, []int{2}, uint32(n), []uint32{uint32(n)})
}
func trainingStageLocked(kernel *VkComputeKernel, tensors []*VkTensorF32, sizes []uint64, writable []int, count uint32, words []uint32) (VkF32Stage, error) {
	bindings := make([]vkBufferBinding, len(tensors))
	for i, t := range tensors {
		b, err := t.bindingLocked()
		if err != nil {
			return VkF32Stage{}, err
		}
		if b.size != sizes[i] {
			return VkF32Stage{}, fmt.Errorf("training tensor shape/storage mismatch")
		}
		bindings[i] = b
	}
	for _, i := range writable {
		for j := range bindings {
			if i != j && vkBindingsOverlap(bindings[i], bindings[j]) {
				return VkF32Stage{}, fmt.Errorf("training writable storage overlaps")
			}
		}
	}
	groups := [3]uint32{(count-1)/256 + 1, 1, 1}
	if err := kernel.validateBindingsLocked(groups[0], 1, 1, bindings, unsafePushWords(words)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: kernel, Groups: groups, Tensors: tensors, PushWords: words}, nil
}
