package nvidia

import (
	"errors"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

// TrainingTape owns device tensors for a validated ordered flow tape. Transfers
// occur only at construction and requested outputs; each operation currently
// synchronises. Caller owns kernel modules and must retain them until Close.
// An uncertain device error retains all allocations, never blind free/retry.
type TrainingTape struct {
	op       *TrainingF32
	buffers  []*Buffer
	owned    []*Buffer
	commands []train.Command
	outputs  []int
	shapes   [][]int
	terminal error
	closed   bool
	reuse    bool
}

func NewTrainingTape(op *TrainingF32, t train.Tape, maxBytes int64) (*TrainingTape, error) {
	return newTrainingTapeBound(op, t, nil, maxBytes)
}
func newTrainingTapeBound(op *TrainingF32, t train.Tape, borrowed map[int]*Buffer, maxBytes int64, reuse ...bool) (owner *TrainingTape, err error) {
	if err = t.Validate(maxBytes); err != nil {
		return nil, err
	}
	if err = train.ValidateExternalTapeBindings(t, borrowed); err != nil {
		return nil, err
	}
	var storage train.TapeStoragePlan
	if len(reuse) > 0 && reuse[0] {
		if len(reuse) > 1 && reuse[1] {
			protected := map[int]bool{}
			for id := range borrowed {
				protected[id] = true
			}
			storage, err = train.PlanZeroTapeStorage(t, maxBytes, protected)
		} else {
			storage, err = train.PlanTapeStorage(t, maxBytes)
		}
		if err != nil {
			return nil, err
		}
	}
	if err = op.admitTapeKernels(t); err != nil {
		return nil, err
	}
	for id, b := range borrowed {
		if id < 0 || id >= len(t.Tensors) || (t.Tensors[id].Values == nil && !t.Tensors[id].External) {
			return nil, fmt.Errorf("PTX tape borrowed input id")
		}
		n, _ := t.Tensors[id].Elements()
		if !trainingExtent(b, n) || b.Size != n*4 {
			return nil, fmt.Errorf("PTX tape borrowed input extent")
		}
	}
	for id, b := range borrowed {
		n, _ := t.Tensors[id].Elements()
		for other, c := range borrowed {
			if other < id {
				cn, _ := t.Tensors[other].Elements()
				if trainingOverlap(b, n, c, cn) {
					return nil, fmt.Errorf("PTX tape borrowed inputs overlap")
				}
			}
		}
	}
	owner = &TrainingTape{reuse: storage.Slots != nil, op: op, buffers: make([]*Buffer, len(t.Tensors)), outputs: append([]int(nil), t.Outputs...), shapes: make([][]int, len(t.Tensors))}
	for i, s := range t.Tensors {
		owner.shapes[i] = append([]int(nil), s.Shape...)
	}
	defer func() {
		if err != nil {
			if owner.terminal != nil {
				return
			}
			if closeErr := owner.Close(); closeErr != nil {
				err = errors.Join(err, closeErr)
			} else {
				owner = nil
			}
		}
	}()
	aliases := map[int]int{}
	for _, c := range t.Commands {
		if c.Op == "reshape" {
			aliases[c.Output] = c.Inputs[0]
		}
	}
	slots := map[int]*Buffer{}
	for i, s := range t.Tensors {
		if b, ok := borrowed[i]; ok {
			owner.buffers[i] = b
			continue
		}
		if _, alias := aliases[i]; alias {
			continue
		}
		n, _ := s.Elements()
		if storage.Slots != nil {
			if b, ok := slots[storage.Slots[i]]; ok {
				owner.buffers[i] = b
				continue
			}
		}
		b, e := Malloc(n)
		if e != nil {
			owner.terminal = e
			return owner, e
		}
		owner.owned = append(owner.owned, b)
		for id, input := range borrowed {
			bn, _ := t.Tensors[id].Elements()
			if trainingOverlap(b, n, input, bn) {
				owner.terminal = fmt.Errorf("PTX allocator overlaps borrowed input")
				return owner, owner.terminal
			}
		}
		owner.buffers[i] = b
		if storage.Slots != nil {
			slots[storage.Slots[i]] = b
		}
		if s.Values != nil {
			if e = b.Upload(s.Values); e != nil {
				owner.terminal = e
				return owner, e
			}
		}
	}
	for _, c := range t.Commands {
		if c.Op == "reshape" {
			owner.buffers[c.Output] = owner.buffers[c.Inputs[0]]
			continue
		}
		clone := c
		clone.Inputs = append([]int(nil), c.Inputs...)
		owner.commands = append(owner.commands, clone)
	}
	return owner, nil
}

// Admit every entry before any allocation or launch; mixed RoPE/attention
// tapes require both modules. A missing later entry cannot follow partial work.
func (op *TrainingF32) admitTapeKernels(t train.Tape) error {
	if op == nil {
		return fmt.Errorf("explicit training kernels required")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.poison != nil {
		return fmt.Errorf("terminal training kernels")
	}
	for _, c := range t.Commands {
		if c.Op == "reshape" {
			continue
		}
		module, name := op.dualModule, ""
		switch c.Op {
		case "linear-forward":
			name = "training_linear_dual_forward"
		case "linear-backward":
			name = "training_linear_dual_backward"
		case "silu-forward":
			name = "training_silu_dual_forward"
		case "silu-backward":
			name = "training_silu_dual_adjoint"
		case "norm-forward":
			name = "training_norm_dual_forward"
		case "norm-backward":
			name = "training_norm_dual_backward"
		case "mod-forward":
			name = "training_mod_dual_forward"
		case "mod-backward":
			name = "training_mod_dual_backward"
		case "vector":
			name = "training_vector"
		case "rope":
			name = "training_rope"
		case "attention-forward":
			name = "training_attention_forward"
		case "lsd-loss":
			name = "training_lsd_loss"
		case "eos-admission":
			name = "training_eos_admission"
		case "eos":
			name = "training_eos"
		case "time-embedding":
			name = "training_time_embedding"
		case "embedding":
			name = "training_embedding"
		case "relu":
			name = "training_relu"
		case "gelu":
			name = "training_gelu"
		case "row":
			name = "training_row"
		case "attention-backward":
			module, name = op.module, "training_attention_backward"
		default:
			return fmt.Errorf("unsupported PTX tape operation")
		}
		if module == nil || module.Function(name) == 0 {
			return fmt.Errorf("PTX tape kernel %s unavailable", name)
		}
	}
	return nil
}

func (t *TrainingTape) Run() ([][]float32, error) {
	if err := t.runDevice(); err != nil {
		return nil, err
	}
	return t.downloadOutputs(t.outputs)
}
func (t *TrainingTape) runDevice() error {
	if t == nil || t.closed || t.terminal != nil {
		return fmt.Errorf("nil/closed/terminal PTX training tape")
	}
	for _, c := range t.commands {
		if err := t.command(c); err != nil {
			t.terminal = err
			return err
		}
	}
	return nil
}
func (t *TrainingTape) downloadOutputs(ids []int) ([][]float32, error) {
	if t == nil || t.closed || t.terminal != nil {
		return nil, fmt.Errorf("nil/closed/terminal PTX training tape")
	}
	for _, id := range ids {
		if t.reuse && !train.DeclaredTapeOutput(t.outputs, id) {
			return nil, fmt.Errorf("lifetime tape download must be declared output")
		}
		if id < 0 || id >= len(t.buffers) {
			return nil, fmt.Errorf("PTX output id")
		}
	}
	values := make([][]float32, len(ids))
	for i, id := range ids {
		n := 1
		for _, d := range t.shapes[id] {
			n *= d
		}
		values[i] = make([]float32, n)
		if err := t.buffers[id].Download(values[i]); err != nil {
			t.terminal = err
			return nil, err
		}
	}
	return values, nil
}
func (t *TrainingTape) command(c train.Command) error {
	in := func(i int) *Buffer { return t.buffers[c.Inputs[i]] }
	out := t.buffers[c.Output]
	op := t.op
	// Shapes are owned by validated creation. Derive M/K/N from embedded command
	// dimensions captured below, not pointer extents which lose rank metadata.
	dims := t.shapes[c.Inputs[0]]
	switch c.Op {
	case "linear-forward":
		return op.LinearDualForward(out, in(0), in(1), in(2), in(3), dims[0], dims[1], t.shapes[c.Inputs[2]][0], c.Mode)
	case "linear-backward":
		return op.LinearDualBackward(out, in(0), in(1), in(2), in(3), in(4), dims[0], dims[1], t.shapes[c.Inputs[2]][0], c.Mode)
	case "silu-forward":
		return op.SiLUDualForward(out, in(0), in(1), dims[0], c.Mode)
	case "silu-backward":
		return op.siluAdjoint(out, in(0), in(1), in(2), in(3), dims[0], c.Mode)
	case "norm-forward":
		return op.NormDualForward(out, in(0), in(1), in(2), in(3), in(4), dims[0], dims[1], c.Kind, c.Mode, c.Epsilon)
	case "norm-backward":
		return op.NormDualBackward(out, in(0), in(1), in(2), in(3), in(4), in(5), dims[0], dims[1], c.Kind, c.Mode)
	case "mod-forward":
		return op.ModDualForward(out, in(0), in(1), in(2), in(3), dims[0], c.Kind, c.Mode)
	case "mod-backward":
		return op.ModDualBackward(out, in(0), in(1), in(2), in(3), in(4), in(5), dims[0], c.Kind, c.Mode)
	case "lsd-loss":
		return op.LSDLoss(out, in(0), in(1), in(2), dims[0], c.Mode)
	case "eos-admission":
		return op.EOSAdmission(out, in(0), in(1), in(2), c.Rows, c.Width)
	case "eos":
		return op.EOS(out, in(0), in(1), dims[0], c.Rows, c.Mode, c.Scale)
	case "time-embedding":
		return op.TimeEmbedding(out, in(0), in(1), in(2), in(3), dims[0], c.Mode)
	case "embedding":
		return op.Embedding(out, in(0), in(1), c.Rows, c.Width, c.Parts, c.Mode)
	case "relu":
		return op.ReLU(out, in(0), in(1), dims[0], c.Mode)
	case "gelu":
		return op.GELU(out, in(0), in(1), dims[0], c.Mode)
	case "row":
		return op.Row(out, in(0), in(1), c.Rows, c.Width, c.Parts, c.Slot, c.Mode)
	case "attention-backward":
		return op.AttentionBackward(out, in(0), in(1), in(2), in(3), in(4), in(5), dims[0], dims[1], dims[2], c.Offset, c.Mode, c.Scale)
	case "attention-forward":
		return op.AttentionForward(out, in(0), in(1), in(2), in(3), dims[0], dims[1], dims[2], c.Offset, c.Mode, c.Scale)
	case "rope":
		return op.RoPE(out, in(0), in(1), dims[0], dims[1], dims[2], c.Mode)
	case "vector":
		return op.Vector(out, in(0), in(1), t.shapes[c.Output][0], dims[0], c.Offset, c.Mode, c.Scale)
	default:
		return fmt.Errorf("unknown PTX tape operation")
	}
}
func (op *TrainingF32) siluAdjoint(dst, x, xt, gv, gt *Buffer, n int, mode uint32) error {
	if n < 1 || n > 1<<26 || mode > 1 {
		return fmt.Errorf("dual SiLU adjoint bound")
	}
	if err := dualExtents([]*Buffer{x, xt, gv, gt, dst}, []int{n, n, n, n, n}); err != nil {
		return err
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.dualModule == nil {
		return fmt.Errorf("dual module unavailable")
	}
	nn := uint32(n)
	if err := LaunchKernel(op.dualModule.Function("training_silu_dual_adjoint"), uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&xt.Ptr), unsafe.Pointer(&gv.Ptr), unsafe.Pointer(&gt.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&mode)); err != nil {
		return err
	}
	return SyncErr()
}
func (t *TrainingTape) Close() error {
	if t == nil || t.closed {
		return nil
	}
	if t.terminal != nil {
		return fmt.Errorf("PTX training tape retains uncertain storage: %w", t.terminal)
	}
	var err error
	for _, b := range t.owned {
		err = errors.Join(err, b.FreeChecked())
	}
	if err != nil {
		t.terminal = err
	} else {
		t.closed = true
	}
	return err
}
