package nvidia

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/nvidia/ptx"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// TrainingF32 owns experimental training kernels, not model parameters. All
// buffers and this module must outlive launches. Methods synchronise before
// returning: an error may follow partial device mutation; recover a coherent
// checkpoint, never blindly replay the update. No CPU fallback is performed.
// Close retains a failed module for explicit recovery, matching PTXModule.
type TrainingF32 struct {
	mu         sync.Mutex
	module     *PTXModule
	dualModule *PTXModule
	poison     error // Any attempted AdamW launch failure makes state uncertain.
}

func NewTrainingF32() (*TrainingF32, error) {
	// Each source has its own header, so retain separate modules via one combined
	// source with exactly one version/target/address declaration.
	const header = ".version 7.0\n.target sm_80\n.address_size 64\n"
	source := ptx.TrainingAdamWPTX + ptx.TrainingLinearBackwardPTX[len(header):] + ptx.TrainingSiLUBackwardPTX[len(header):] + ptx.TrainingLayerNormBackwardPTX[len(header):] + ptx.TrainingAttentionBackwardPTX[len(header):] + ptx.TrainingSiLUDualBackwardPTX[len(header):]
	m, err := LoadPTXFunctions(source, []string{"training_adamw", "training_linear_backward", "training_silu_backward", "training_layernorm_backward", "training_attention_backward", "training_silu_dual_backward"})
	if err != nil {
		return nil, err
	}
	dual, err := LoadPTXFunctions(ptx.TrainingDualPTX, []string{"training_linear_dual_backward", "training_norm_dual_backward", "training_mod_dual_backward", "training_norm_dual_forward", "training_linear_dual_forward", "training_silu_dual_forward", "training_mod_dual_forward", "training_vector", "training_silu_dual_adjoint", "training_rope", "training_attention_forward", "training_gelu", "training_row", "training_norm_stats", "training_relu", "training_embedding", "training_time_embedding", "training_eos", "training_eos_admission", "training_lsd_loss", "training_global_norm", "training_adamw_scaled", "training_embedding_accumulate"})
	if err != nil {
		// Retain a failed-cleanup owner in the returned object. Caller must
		// retain nonnil owner even with error, never discard native handles.
		owner := &TrainingF32{module: m, poison: err}
		if closeErr := owner.Close(); closeErr != nil {
			return owner, fmt.Errorf("dual module load: %w; prior module cleanup: %v", err, closeErr)
		}
		return nil, err
	}
	return &TrainingF32{module: m, dualModule: dual}, nil
}

// AttentionReady performs a host-only ownership check. It never Initialises
// CUDA or probes driver state. A live handle is not proof of device health.
func (op *TrainingF32) AttentionReady() bool {
	if op == nil {
		return false
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	return op.module != nil && op.poison == nil && op.module.Function("training_attention_backward") != 0
}
func (op *TrainingF32) Close() error {
	if op == nil {
		return nil
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	var result error
	if op.module != nil {
		if err := op.module.Close(); err != nil {
			result = errors.Join(result, err)
		} else {
			op.module = nil
		}
	}
	if op.dualModule != nil {
		if err := op.dualModule.Close(); err != nil {
			result = errors.Join(result, err)
		} else {
			op.dualModule = nil
		}
	}
	return result
}
func trainingExtent(b *Buffer, n int) bool {
	if n <= 0 || !whisperF32Extent(b, n) {
		return false
	}
	bytes := uint64(n) * 4
	ptr := uint64(b.Ptr)
	return ptr%4 == 0 && bytes <= ^uint64(0)-ptr
}
func trainingOverlap(a *Buffer, an int, b *Buffer, bn int) bool {
	ap, bp := uint64(a.Ptr), uint64(b.Ptr)
	return ap < bp+uint64(bn)*4 && bp < ap+uint64(an)*4
}
func (op *TrainingF32) AdamW(parameter, gradient, m, v, ema *Buffer, n int, p train.AdamWStep) error {
	if err := p.Validate(); err != nil {
		return err
	}
	if n <= 0 || n > 1<<26 {
		return fmt.Errorf("PTX AdamW vector ceiling")
	}
	buffers := []*Buffer{parameter, gradient, m, v, ema}
	for _, b := range buffers {
		if !trainingExtent(b, n) {
			return fmt.Errorf("PTX AdamW extent")
		}
	}
	for i, a := range buffers {
		for _, b := range buffers[i+1:] {
			if trainingOverlap(a, n, b, n) {
				return fmt.Errorf("PTX AdamW storage overlaps")
			}
		}
	}
	if op == nil {
		return fmt.Errorf("nil PTX training operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.module == nil {
		return fmt.Errorf("closed PTX training operator")
	}
	if op.poison != nil {
		return fmt.Errorf("PTX training update requires coherent recovery: %w", op.poison)
	}
	nn := uint32(n)
	err := LaunchKernel(op.module.Function("training_adamw"), uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&parameter.Ptr), unsafe.Pointer(&gradient.Ptr), unsafe.Pointer(&m.Ptr), unsafe.Pointer(&v.Ptr), unsafe.Pointer(&ema.Ptr), unsafe.Pointer(&nn),
		unsafe.Pointer(&p.LearningRate), unsafe.Pointer(&p.Beta1), unsafe.Pointer(&p.Beta2), unsafe.Pointer(&p.Epsilon), unsafe.Pointer(&p.WeightDecay), unsafe.Pointer(&p.Bias1), unsafe.Pointer(&p.Bias2), unsafe.Pointer(&p.EMADecay), unsafe.Pointer(&p.GradientScale))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}

// SiLUBackward writes the pre-activation derivative to distinct destination.
// Device exponential approximation is experimental, with no CPU fallback.
func (op *TrainingF32) SiLUBackward(dst, x, dy *Buffer, n int) error {
	if n <= 0 || n > 1<<26 || !trainingExtent(dst, n) || !trainingExtent(x, n) || !trainingExtent(dy, n) {
		return fmt.Errorf("PTX SiLU backward extent")
	}
	if trainingOverlap(dst, n, x, n) || trainingOverlap(dst, n, dy, n) {
		return fmt.Errorf("PTX SiLU backward output overlaps")
	}
	if op == nil {
		return fmt.Errorf("nil PTX training operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.module == nil {
		return fmt.Errorf("closed PTX training operator")
	}
	nn := uint32(n)
	if err := LaunchKernel(op.module.Function("training_silu_backward"), uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&dy.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&nn)); err != nil {
		return err
	}
	return SyncErr()
}

// LinearBackward overwrites one destination (mode0=dX,1=dW,2=dBias).
// Read-only inputs may overlap; output may not overlap any input.
func (op *TrainingF32) LinearBackward(dst, x, w, dy *Buffer, m, k, n int, mode uint32) error {
	if err := train.LinearShape(m, k, n); err != nil {
		return err
	}
	if mode > 2 {
		return fmt.Errorf("PTX linear backward mode")
	}
	count := n
	if mode == 0 {
		count = m * k
	} else if mode == 1 {
		count = n * k
	}
	if !trainingExtent(dst, count) || !trainingExtent(x, m*k) || !trainingExtent(w, n*k) || !trainingExtent(dy, m*n) {
		return fmt.Errorf("PTX linear backward extents")
	}
	for _, in := range []struct {
		b *Buffer
		n int
	}{{x, m * k}, {w, n * k}, {dy, m * n}} {
		if trainingOverlap(dst, count, in.b, in.n) {
			return fmt.Errorf("PTX linear backward destination overlaps input")
		}
	}
	if op == nil {
		return fmt.Errorf("nil PTX training operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.module == nil {
		return fmt.Errorf("closed PTX training operator")
	}
	mm, kk, nn := uint32(m), uint32(k), uint32(n)
	err := LaunchKernel(op.module.Function("training_linear_backward"), uint32((count-1)/256+1), 1, 1, 256, 1, 1, 0,
		unsafe.Pointer(&x.Ptr), unsafe.Pointer(&w.Ptr), unsafe.Pointer(&dy.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&mm), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&mode))
	if err != nil {
		return err
	}
	return SyncErr()
}
