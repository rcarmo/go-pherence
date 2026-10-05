package vulkan

import (
	"context"
	"errors"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// VkTrainingTape owns a bounded tensor allocation and ordered plans, while the
// caller owns kernels. Commands/tensor shapes are validated before allocation.
// Forward/backward tensor intermediates stay resident between plan chunks.
// A submitted failure marks terminal and retains all native resources.
type VkTrainingTape struct {
	op       *VkTrainingF32
	tensors  []*VkTensorF32
	arenas   []*VkTensorArena
	plans    []*VkF32Plan
	outputs  []int
	terminal error
	closed   bool
	reuse    bool
}

func NewVkTrainingTape(ctx context.Context, op *VkTrainingF32, tape train.Tape, maxBytes int64) (*VkTrainingTape, error) {
	return newVkTrainingTapeBound(ctx, op, tape, nil, maxBytes)
}
func newVkTrainingTapeBound(ctx context.Context, op *VkTrainingF32, tape train.Tape, borrowed map[int]*VkTensorF32, maxBytes int64, reuse ...bool) (owner *VkTrainingTape, err error) {
	if err = tape.Validate(maxBytes); err != nil {
		return nil, err
	}
	if err = train.ValidateExternalTapeBindings(tape, borrowed); err != nil {
		return nil, err
	}
	var storage train.TapeStoragePlan
	if len(reuse) > 0 && reuse[0] {
		if len(reuse) > 1 && reuse[1] {
			protected := map[int]bool{}
			for id := range borrowed {
				protected[id] = true
			}
			storage, err = train.PlanZeroTapeStorage(tape, maxBytes, protected)
		} else {
			storage, err = train.PlanTapeStorage(tape, maxBytes)
		}
		if err != nil {
			return nil, err
		}
	}
	if err = op.admitTapeKernels(tape); err != nil {
		return nil, err
	}
	if err = vkAcquire(ctx); err != nil {
		return nil, err
	}
	for id, t := range borrowed {
		if id < 0 || id >= len(tape.Tensors) || (tape.Tensors[id].Values == nil && !tape.Tensors[id].External) || t == nil {
			vkRelease()
			return nil, fmt.Errorf("Vulkan borrowed input id")
		}
		n, _ := tape.Tensors[id].Elements()
		if t.Elements() != n {
			vkRelease()
			return nil, fmt.Errorf("Vulkan borrowed input extent")
		}
		if _, e := t.bindingLocked(); e != nil {
			vkRelease()
			return nil, e
		}
	}
	for id, t := range borrowed {
		a, _ := t.bindingLocked()
		for other, u := range borrowed {
			if other < id {
				b, _ := u.bindingLocked()
				if vkBindingsOverlap(a, b) {
					vkRelease()
					return nil, fmt.Errorf("Vulkan borrowed inputs overlap")
				}
			}
		}
	}
	vkRelease()
	owner = &VkTrainingTape{reuse: storage.Slots != nil, op: op, tensors: make([]*VkTensorF32, len(tape.Tensors)), outputs: append([]int(nil), tape.Outputs...)}
	defer func() {
		if err != nil {
			if closeErr := owner.Close(); closeErr != nil {
				err = errors.Join(err, closeErr)
			} else {
				owner = nil
			}
		}
	}()
	aliases := map[int]int{}
	for _, c := range tape.Commands {
		if c.Op == "reshape" {
			aliases[c.Output] = c.Inputs[0]
		}
	}
	slots := map[int]*VkTensorF32{}
	for i, s := range tape.Tensors {
		if t, ok := borrowed[i]; ok {
			view, e := t.ReshapeF32(ctx, s.Shape...)
			if e != nil {
				return owner, e
			}
			owner.tensors[i] = view
			continue
		}
		if _, alias := aliases[i]; alias {
			continue
		}
		n, _ := s.Elements()
		if storage.Slots != nil {
			if base, ok := slots[storage.Slots[i]]; ok {
				view, e := base.ReshapeF32(ctx, s.Shape...)
				if e != nil {
					return owner, e
				}
				owner.tensors[i] = view
				continue
			}
		}
		a, e := NewVkTensorArena(ctx, n*4)
		if e != nil {
			return owner, e
		}
		owner.arenas = append(owner.arenas, a)
		tensor, e := a.AllocF32(ctx, s.Shape...)
		if e != nil {
			return owner, e
		}
		owner.tensors[i] = tensor
		if storage.Slots != nil {
			slots[storage.Slots[i]] = tensor
		}
		if s.Values != nil {
			if e = tensor.Upload(ctx, s.Values); e != nil {
				return owner, e
			}
		}
	}
	var stages []VkF32Stage
	flush := func() error {
		if len(stages) == 0 {
			return nil
		}
		p, e := NewVkF32Plan(ctx, stages)
		if e != nil {
			return e
		}
		owner.plans = append(owner.plans, p)
		stages = nil
		return nil
	}
	for _, c := range tape.Commands {
		if c.Op == "reshape" {
			view, e := owner.tensors[c.Inputs[0]].ReshapeF32(ctx, tape.Tensors[c.Output].Shape...)
			if e != nil {
				return owner, e
			}
			owner.tensors[c.Output] = view
			continue
		}
		s, e := op.tapeStage(ctx, owner.tensors, c)
		if e != nil {
			return owner, e
		}
		stages = append(stages, s)
		if len(stages) == 64 {
			if e = flush(); e != nil {
				return owner, e
			}
		}
	}
	if err = flush(); err != nil {
		return owner, err
	}
	return owner, nil
}

// Check all host kernel owners before allocating any tape tensors/plans.
func (op *VkTrainingF32) admitTapeKernels(tape train.Tape) error {
	if op == nil {
		return fmt.Errorf("nil training kernels")
	}
	for _, c := range tape.Commands {
		if c.Op == "reshape" {
			continue
		}
		var k *VkComputeKernel
		switch c.Op {
		case "linear-forward":
			k = op.linearDualForward
		case "linear-backward":
			k = op.linearDual
		case "silu-forward":
			k = op.siluDualForward
		case "silu-backward":
			k = op.siluDualAdjoint
		case "norm-forward":
			k = op.normDualForward
		case "norm-backward":
			k = op.normDual
		case "mod-forward":
			k = op.modDualForward
		case "mod-backward":
			k = op.modDual
		case "vector":
			k = op.vector
		case "rope":
			k = op.rope
		case "attention-forward":
			k = op.attentionForward
		case "lsd-loss":
			k = op.lsdLoss
		case "eos-admission":
			k = op.eosAdmission
		case "eos":
			k = op.eos
		case "time-embedding":
			k = op.timeEmbedding
		case "embedding":
			k = op.embedding
		case "relu":
			k = op.relu
		case "gelu":
			k = op.gelu
		case "row":
			k = op.row
		case "attention-backward":
			k = op.attention
		default:
			return fmt.Errorf("unsupported Vulkan tape operation")
		}
		if k == nil || k.closed {
			return fmt.Errorf("Vulkan tape kernel %s unavailable", c.Op)
		}
	}
	return nil
}

func (op *VkTrainingF32) tapeStage(ctx context.Context, t []*VkTensorF32, c train.Command) (VkF32Stage, error) {
	in := func(i int) *VkTensorF32 { return t[c.Inputs[i]] }
	out := t[c.Output]
	switch c.Op {
	case "linear-forward":
		return op.LinearDualForwardStage(ctx, out, in(0), in(1), in(2), in(3), c.Mode)
	case "linear-backward":
		return op.LinearDualBackwardStage(ctx, out, in(0), in(1), in(2), in(3), in(4), c.Mode)
	case "silu-forward":
		return op.SiLUDualForwardStage(ctx, out, in(0), in(1), c.Mode)
	case "silu-backward":
		return op.siluAdjointStage(ctx, out, in(0), in(1), in(2), in(3), c.Mode)
	case "norm-forward":
		return op.NormDualForwardStage(ctx, out, in(0), in(1), in(2), in(3), in(4), c.Kind, c.Mode, c.Epsilon)
	case "norm-backward":
		return op.NormDualBackwardStage(ctx, out, in(0), in(1), in(2), in(3), in(4), in(5), c.Kind, c.Mode)
	case "mod-forward":
		return op.ModDualForwardStage(ctx, out, in(0), in(1), in(2), in(3), c.Kind, c.Mode)
	case "mod-backward":
		return op.ModDualBackwardStage(ctx, out, in(0), in(1), in(2), in(3), in(4), in(5), c.Kind, c.Mode)
	case "lsd-loss":
		return op.LSDLossStage(ctx, out, in(0), in(1), in(2), c.Mode)
	case "eos-admission":
		return op.EOSAdmissionStage(ctx, in(0), in(1), in(2), out, c.Rows, c.Width)
	case "eos":
		return op.EOSStage(ctx, out, in(0), in(1), c.Rows, c.Mode, c.Scale)
	case "time-embedding":
		return op.TimeEmbeddingStage(ctx, out, in(0), in(1), in(2), in(3), c.Mode)
	case "embedding":
		return op.EmbeddingStage(ctx, out, in(0), in(1), c.Rows, c.Width, c.Parts, c.Mode)
	case "relu":
		return op.ReLUStage(ctx, out, in(0), in(1), c.Mode)
	case "gelu":
		return op.GELUStage(ctx, out, in(0), in(1), c.Mode)
	case "row":
		return op.RowStage(ctx, out, in(0), in(1), c.Rows, c.Width, c.Parts, c.Slot, c.Mode)
	case "attention-backward":
		return op.AttentionBackwardStage(ctx, out, in(0), in(1), in(2), in(3), in(4), in(5), c.Offset, c.Mode, c.Scale)
	case "attention-forward":
		return op.AttentionForwardStage(ctx, out, in(0), in(1), in(2), in(3), c.Offset, c.Mode, c.Scale)
	case "rope":
		return op.RoPEStage(ctx, out, in(0), in(1), c.Mode)
	case "vector":
		return op.VectorStage(ctx, out, in(0), in(1), c.Offset, c.Mode, c.Scale)
	default:
		return VkF32Stage{}, fmt.Errorf("unsupported tape operation")
	}
}
func (op *VkTrainingF32) siluAdjointStage(ctx context.Context, out, x, xt, gv, gt *VkTensorF32, mode uint32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	if op == nil || op.siluDualAdjoint == nil || mode > 1 || x == nil || x.rank != 1 {
		return VkF32Stage{}, fmt.Errorf("invalid dual SiLU adjoint")
	}
	n := x.shape[0]
	tensors := []*VkTensorF32{x, xt, gv, gt, out}
	sizes := make([]uint64, 5)
	for i, t := range tensors {
		if t == nil || t.rank != 1 || t.shape[0] != n {
			return VkF32Stage{}, fmt.Errorf("dual SiLU adjoint shape")
		}
		sizes[i] = uint64(n) * 4
	}
	return trainingStageLocked(op.siluDualAdjoint, tensors, sizes, []int{4}, uint32(n), []uint32{uint32(n), mode})
}
func (t *VkTrainingTape) Run(ctx context.Context) ([][]float32, error) {
	if err := t.runDevice(ctx); err != nil {
		return nil, err
	}
	return t.downloadOutputs(ctx, t.outputs)
}
func (t *VkTrainingTape) runDevice(ctx context.Context) error {
	if t == nil || t.closed || t.terminal != nil {
		return fmt.Errorf("nil/closed/terminal Vulkan training tape")
	}
	for _, p := range t.plans {
		if err := p.Run(ctx); err != nil {
			t.terminal = err
			return err
		}
	}
	return nil
}
func (t *VkTrainingTape) downloadOutputs(ctx context.Context, ids []int) ([][]float32, error) {
	if t == nil || t.closed || t.terminal != nil {
		return nil, fmt.Errorf("nil/closed/terminal Vulkan training tape")
	}
	for _, id := range ids {
		if t.reuse && !train.DeclaredTapeOutput(t.outputs, id) {
			return nil, fmt.Errorf("lifetime tape download must be declared output")
		}
		if id < 0 || id >= len(t.tensors) {
			return nil, fmt.Errorf("Vulkan output id")
		}
	}
	values := make([][]float32, len(ids))
	for i, id := range ids {
		values[i] = make([]float32, t.tensors[id].Elements())
		if err := t.tensors[id].Download(ctx, values[i]); err != nil {
			t.terminal = err
			return nil, err
		}
	}
	return values, nil
}
func (t *VkTrainingTape) Close() error {
	if t == nil || t.closed {
		return nil
	}
	if t.terminal != nil {
		return fmt.Errorf("Vulkan training tape retains uncertain resources: %w", t.terminal)
	}
	var err error
	for _, p := range t.plans {
		err = errors.Join(err, p.Close())
	}
	if err != nil {
		return err
	}
	for _, a := range t.arenas {
		err = errors.Join(err, a.Close())
	}
	if err != nil {
		t.terminal = err
	} else {
		t.closed = true
	}
	return err
}
