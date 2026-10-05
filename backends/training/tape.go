package training

import (
	"fmt"
	"math"
)

// Tape is a bounded, explicit F32 execution list for flow dual primitives.
// Tensor IDs index owned contiguous storage. Commands overwrite fresh/disjoint
// outputs; branch sums use Vector rather than device float atomics. No device
// is initialised by admission. Constructors must validate before allocation.
type TensorSpec struct {
	Shape  []int
	Values []float32
	// External declares a shape-only read-only device input with no host values.
	// Execution requires an exact native binding; it cannot be written by commands.
	External bool
}
type Command struct {
	Op                       string
	Inputs                   []int
	Output                   int
	Mode, Kind               uint32
	Offset                   int
	Rows, Width, Parts, Slot int // explicit flat row-operation geometry
	Scale, Epsilon           float32
}
type Tape struct {
	Tensors  []TensorSpec
	Commands []Command
	Outputs  []int
}

func (s TensorSpec) Elements() (int, error) {
	if len(s.Shape) < 1 || len(s.Shape) > 3 {
		return 0, fmt.Errorf("flow tape rank")
	}
	n := 1
	for _, d := range s.Shape {
		if d <= 0 || d > (1<<26)/n {
			return 0, fmt.Errorf("flow tape tensor bound")
		}
		n *= d
	}
	return n, nil
}
func (t Tape) Validate(maxBytes int64) error {
	if maxBytes <= 0 || len(t.Tensors) == 0 || len(t.Tensors) > 4096 || len(t.Commands) > 4096 || len(t.Outputs) == 0 {
		return fmt.Errorf("flow tape ceilings")
	}
	total := int64(0)
	initialized := make([]bool, len(t.Tensors))
	written := make([]bool, len(t.Tensors))
	for i, s := range t.Tensors {
		n, err := s.Elements()
		if err != nil {
			return err
		}
		total += int64(n) * 4
		if total > maxBytes {
			return fmt.Errorf("flow tape byte ceiling")
		}
		if s.External {
			if s.Values != nil {
				return fmt.Errorf("external tensor%d has host values", i)
			}
			initialized[i] = true
		}
		if s.Values != nil {
			if len(s.Values) != n {
				return fmt.Errorf("flow tape input length")
			}
			for _, x := range s.Values {
				if math.Float32bits(x)&0x7f800000 == 0x7f800000 {
					return fmt.Errorf("flow tape nonfinite input")
				}
			}
			initialized[i] = true
		}
	}
	for index, c := range t.Commands {
		if c.Output < 0 || c.Output >= len(t.Tensors) || initialized[c.Output] || written[c.Output] {
			return fmt.Errorf("flow tape command%d output owner", index)
		}
		for _, id := range c.Inputs {
			if id < 0 || id >= len(t.Tensors) || !initialized[id] || id == c.Output {
				return fmt.Errorf("flow tape command%d input dependency", index)
			}
		}
		if err := t.validateCommand(c); err != nil {
			return fmt.Errorf("flow tape command%d: %w", index, err)
		}
		initialized[c.Output] = true
		written[c.Output] = true
	}
	for _, id := range t.Outputs {
		if id < 0 || id >= len(t.Tensors) || !initialized[id] {
			return fmt.Errorf("flow tape requested output")
		}
	}
	return nil
}
func (t Tape) validateCommand(c Command) error {
	get := func(i int) TensorSpec { return t.Tensors[c.Inputs[i]] }
	out := t.Tensors[c.Output]
	same := func(a, b []int) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	}
	need := func(count int) bool { return len(c.Inputs) == count }
	switch c.Op {
	case "lsd-loss":
		if !need(3) || len(get(0).Shape) != 1 {
			return fmt.Errorf("LSD loss input count/rank")
		}
		n := get(0).Shape[0]
		outN, err := LSDLossShape(n, c.Mode)
		if err != nil {
			return err
		}
		if !same(get(1).Shape, []int{n}) || !same(get(2).Shape, []int{1}) || !same(out.Shape, []int{outN}) {
			return fmt.Errorf("LSD loss tensor shape")
		}
		if c.Mode == 2 {
			cutoff, err := LSDPredictionCutoff(n, c.Scale)
			if err != nil || get(2).Values == nil || get(2).Values[0] != cutoff {
				return fmt.Errorf("LSD admission cutoff constant/provenance")
			}
		} else if get(2).Values == nil || get(2).Values[0] < -80 || get(2).Values[0] > 80 {
			return fmt.Errorf("LSD logvar constant/range")
		}
	case "eos-admission":
		if len(c.Inputs) != 3 {
			return fmt.Errorf("EOS admission input count")
		}
		n, _, err := EOSAdmissionShape(c.Rows, c.Width)
		if err != nil {
			return err
		}
		if !same(get(0).Shape, []int{n}) || !same(get(1).Shape, []int{c.Width}) || !same(get(2).Shape, []int{1}) || !same(out.Shape, []int{1}) {
			return fmt.Errorf("EOS admission shape")
		}
		return nil
	case "eos":
		if !need(2) || len(get(0).Shape) != 1 {
			return fmt.Errorf("EOS input count/rank")
		}
		n := get(0).Shape[0]
		outN, err := EOSShape(n, c.Rows, c.Mode, c.Scale)
		if err != nil {
			return err
		}
		if !same(get(1).Shape, []int{n}) || !same(out.Shape, []int{outN}) || get(1).Values == nil {
			return fmt.Errorf("EOS labels/output shape")
		}
		active := 0
		for _, label := range get(1).Values {
			if label != -1 && label != 0 && label != 1 {
				return fmt.Errorf("EOS label value")
			}
			if label >= 0 {
				active++
			}
		}
		if active != c.Rows {
			return fmt.Errorf("EOS active count mismatch")
		}
	case "time-embedding":
		if !need(4) || len(get(0).Shape) != 1 {
			return fmt.Errorf("time embedding input count/rank")
		}
		half := get(0).Shape[0]
		n, err := TimeEmbeddingShape(half, c.Mode)
		if err != nil {
			return err
		}
		if !same(get(1).Shape, []int{2}) || !same(get(2).Shape, []int{2 * half}) || !same(get(3).Shape, []int{2 * half}) || !same(out.Shape, []int{n}) {
			return fmt.Errorf("time embedding tensor shape")
		}
		f, time := get(0).Values, get(1).Values
		if f == nil || time == nil {
			return fmt.Errorf("time embedding frequency/time constants required")
		}
		for _, v := range f {
			angle := time[0] * v
			if math.IsInf(float64(angle), 0) || math.Abs(float64(angle)) > 8192 || math.IsInf(float64(v*v), 0) || math.IsInf(float64(v*v*time[1]), 0) {
				return fmt.Errorf("time embedding angle/derivative ceiling")
			}
		}
	case "embedding":
		if !need(2) {
			return fmt.Errorf("embedding input count")
		}
		source, n, err := EmbeddingShape(c.Rows, c.Width, c.Parts, c.Mode)
		if err != nil {
			return err
		}
		indexSize := c.Rows
		if c.Mode == 2 {
			indexSize += c.Parts + 1
		}
		if c.Mode == 3 {
			indexSize = 3*c.Rows + 1
		}
		if !same(get(0).Shape, []int{source}) || !same(get(1).Shape, []int{indexSize}) || !same(out.Shape, []int{n}) {
			return fmt.Errorf("embedding tensor shape")
		}
		if get(1).Values == nil {
			return fmt.Errorf("embedding IDs must be explicit constants")
		}
		if c.Mode == 2 {
			return ValidateEmbeddingIndex(get(1).Values, c.Rows, c.Parts)
		}
		if c.Mode == 3 {
			return ValidateCompactEmbeddingIndex(get(1).Values, c.Rows, c.Parts)
		}
		for _, id := range get(1).Values {
			if id < 0 || id >= float32(c.Parts) || id != float32(int(id)) {
				return fmt.Errorf("embedding ID out of range/noninteger")
			}
		}
	case "reshape":
		if !need(1) {
			return fmt.Errorf("reshape input count")
		}
		a, err := get(0).Elements()
		if err != nil {
			return err
		}
		b, err := out.Elements()
		if err != nil || a != b {
			return fmt.Errorf("reshape element mismatch")
		}
	case "linear-forward", "linear-backward":
		count := 4
		if c.Op == "linear-backward" {
			count = 5
		}
		if !need(count) {
			return fmt.Errorf("linear input count")
		}
		x, xt, w := get(0), get(1), get(2)
		if len(x.Shape) != 2 || len(w.Shape) != 2 || !same(x.Shape, xt.Shape) {
			return fmt.Errorf("linear ranks")
		}
		m, k, n := x.Shape[0], x.Shape[1], w.Shape[0]
		if err := LinearShape(m, k, n); err != nil {
			return err
		}
		if w.Shape[1] != k {
			return fmt.Errorf("linear weights")
		}
		if c.Op == "linear-forward" {
			if c.Mode > 1 || !same(get(3).Shape, []int{n}) || !same(out.Shape, []int{m, n}) {
				return fmt.Errorf("linear forward shape")
			}
		}
		if c.Op == "linear-backward" {
			if c.Mode > 3 || !same(get(3).Shape, []int{m, n}) || !same(get(4).Shape, []int{m, n}) {
				return fmt.Errorf("linear seed shape")
			}
			shape := []int{n}
			if c.Mode < 2 {
				shape = []int{m, k}
			} else if c.Mode == 2 {
				shape = []int{n, k}
			}
			if !same(out.Shape, shape) {
				return fmt.Errorf("linear adjoint shape")
			}
		}
	case "silu-forward", "silu-backward":
		count := 2
		if c.Op == "silu-backward" {
			count = 4
		}
		if !need(count) || c.Mode > 1 || len(out.Shape) != 1 {
			return fmt.Errorf("SiLU shape/mode")
		}
		for i := range c.Inputs {
			if !same(get(i).Shape, out.Shape) {
				return fmt.Errorf("SiLU input shape")
			}
		}
	case "norm-forward", "norm-backward":
		count := 5
		if c.Op == "norm-backward" {
			count = 6
		}
		if !need(count) {
			return fmt.Errorf("norm input count")
		}
		x := get(0)
		if len(x.Shape) != 2 || !same(x.Shape, get(1).Shape) {
			return fmt.Errorf("norm input rank")
		}
		m, n := x.Shape[0], x.Shape[1]
		if !same(get(2).Shape, []int{n}) {
			return fmt.Errorf("norm weight shape")
		}
		if c.Op == "norm-forward" {
			if err := DualNormForwardShape(m, n, c.Kind, c.Mode, c.Epsilon); err != nil {
				return err
			}
			if !same(get(3).Shape, []int{n}) || !same(get(4).Shape, []int{m, 5}) {
				return fmt.Errorf("norm forward input")
			}
			shape := []int{m, n}
			if c.Mode == 0 {
				shape = []int{m, 5}
			}
			if !same(out.Shape, shape) {
				return fmt.Errorf("norm forward output")
			}
		}
		if c.Op == "norm-backward" {
			if err := DualNormShape(m, n, c.Kind, c.Mode); err != nil {
				return err
			}
			if !same(get(3).Shape, x.Shape) || !same(get(4).Shape, x.Shape) || !same(get(5).Shape, []int{m, 5}) {
				return fmt.Errorf("norm backward input")
			}
			shape := []int{n}
			if c.Mode < 2 {
				shape = x.Shape
			}
			if !same(out.Shape, shape) {
				return fmt.Errorf("norm adjoint output")
			}
		}
	case "mod-forward", "mod-backward":
		count := 4
		if c.Op == "mod-backward" {
			count = 6
		}
		if !need(count) || len(get(0).Shape) != 1 {
			return fmt.Errorf("mod input count/rank")
		}
		n := get(0).Shape[0]
		if err := DualModShape(n, c.Kind, c.Mode); err != nil {
			return err
		}
		if c.Op == "mod-forward" && c.Mode > 1 {
			return fmt.Errorf("mod forward mode")
		}
		bn := n
		if c.Kind == 0 {
			bn *= 2
		}
		if !same(get(1).Shape, []int{n}) || !same(get(2).Shape, []int{bn}) || !same(get(3).Shape, []int{bn}) {
			return fmt.Errorf("mod input shape")
		}
		size := n
		if c.Op == "mod-backward" {
			if !same(get(4).Shape, []int{n}) || !same(get(5).Shape, []int{n}) {
				return fmt.Errorf("mod seeds")
			}
			if c.Kind == 0 && c.Mode >= 2 {
				size = bn
			}
		}
		if !same(out.Shape, []int{size}) {
			return fmt.Errorf("mod output shape")
		}
	case "relu":
		if !need(2) || len(out.Shape) != 1 || !same(get(0).Shape, out.Shape) || !same(get(1).Shape, out.Shape) {
			return fmt.Errorf("ReLU tensor shape")
		}
		if err := ReLUShape(out.Shape[0], c.Mode); err != nil {
			return err
		}
	case "gelu":
		if !need(2) || len(out.Shape) != 1 || !same(get(0).Shape, out.Shape) || !same(get(1).Shape, out.Shape) {
			return fmt.Errorf("GELU tensor shape")
		}
		if err := GELUShape(out.Shape[0], c.Mode); err != nil {
			return err
		}
	case "row":
		if !need(2) {
			return fmt.Errorf("row input count")
		}
		an, bn, n, err := RowShape(c.Rows, c.Width, c.Parts, c.Slot, c.Mode)
		if err != nil {
			return err
		}
		if !same(get(0).Shape, []int{an}) || !same(get(1).Shape, []int{bn}) || !same(out.Shape, []int{n}) {
			return fmt.Errorf("row tensor shape")
		}
	case "attention-backward":
		if !need(6) || len(get(0).Shape) != 3 {
			return fmt.Errorf("attention backward input count/rank")
		}
		x := get(0)
		rows, heads, dim := x.Shape[0], x.Shape[1], x.Shape[2]
		if err := AttentionShape(rows, heads, dim, c.Offset, c.Mode, c.Scale); err != nil {
			return err
		}
		for _, i := range []int{1, 2, 4} {
			if !same(get(i).Shape, x.Shape) {
				return fmt.Errorf("attention backward tensor input shape")
			}
		}
		dense := []int{heads, rows, rows}
		if !same(get(3).Shape, dense) || !same(get(5).Shape, dense) {
			return fmt.Errorf("attention backward probability/aux shape")
		}
		shape := x.Shape
		if c.Mode < 2 {
			shape = dense
		}
		if !same(out.Shape, shape) {
			return fmt.Errorf("attention backward output shape")
		}
	case "attention-forward":
		if !need(4) || len(get(0).Shape) != 3 {
			return fmt.Errorf("attention forward input count/rank")
		}
		x := get(0)
		rows, heads, dim := x.Shape[0], x.Shape[1], x.Shape[2]
		if err := AttentionForwardShape(rows, heads, dim, c.Offset, c.Mode, c.Scale); err != nil {
			return err
		}
		if !same(get(1).Shape, x.Shape) || !same(get(2).Shape, x.Shape) || !same(get(3).Shape, []int{heads, rows, rows}) {
			return fmt.Errorf("attention forward input shape")
		}
		shape := []int{heads, rows, rows}
		if c.Mode == 2 {
			shape = x.Shape
		}
		if !same(out.Shape, shape) {
			return fmt.Errorf("attention forward output shape")
		}
	case "rope":
		if !need(2) || len(get(0).Shape) != 3 || len(get(1).Shape) != 3 {
			return fmt.Errorf("RoPE input count/rank")
		}
		x := get(0)
		rows, heads, dim := x.Shape[0], x.Shape[1], x.Shape[2]
		if err := RoPEShape(rows, heads, dim, c.Mode); err != nil {
			return err
		}
		if !same(out.Shape, x.Shape) || !same(get(1).Shape, []int{rows, dim / 2, 2}) {
			return fmt.Errorf("RoPE tensor shape")
		}
	case "vector":
		if !need(2) || len(get(0).Shape) != 1 || len(get(1).Shape) != 1 || len(out.Shape) != 1 {
			return fmt.Errorf("vector rank")
		}
		if err := VectorShape(out.Shape[0], get(0).Shape[0], c.Offset, c.Mode, c.Scale); err != nil {
			return err
		}
		if c.Mode == 0 && get(1).Shape[0] != out.Shape[0] {
			return fmt.Errorf("vector input B")
		}
	default:
		return fmt.Errorf("unsupported flow tape operation %q", c.Op)
	}
	return nil
}
