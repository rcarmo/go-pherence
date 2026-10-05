package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

// AttentionTrainingTape evaluates exact-length causal attention and its input
// adjoints, including forward and transpose adjacent-pair RoPE. Inputs Q/K/V
// are unrotated [rows,heads,dim]; outputs and gradients have the same layout.
// Model projections, norm/MLP, parameter gradients and optimisers are outside
// this graph. It performs no native initialisation or automatic fallback.
type AttentionTrainingTape struct {
	Tape             train.Tape
	rows, heads, dim int
}
type AttentionTrainingResult struct{ Output, Probability, DQ, DK, DV []float32 }

func BuildAttentionTrainingTape(q, k, v, dy []float32, rows, heads, dim, contextRows, origin int, maxPeriod float64, maxBytes int64) (AttentionTrainingTape, error) {
	var result AttentionTrainingTape
	if err := train.RoPEShape(rows, heads, dim, 0); err != nil {
		return result, err
	}
	scale := float32(1 / math.Sqrt(float64(dim)))
	if err := train.AttentionShape(rows, heads, dim, contextRows, 0, scale); err != nil {
		return result, err
	}
	if maxBytes <= 0 {
		return result, fmt.Errorf("attention tape byte ceiling")
	}
	n, dense := rows*heads*dim, heads*rows*rows
	// Graph has 12 tensor-sized tensors and five dense tensors plus one shared
	// RoPE table. Preflight the complete logical footprint before copying inputs.
	bytes := int64(12*n+5*dense+rows*dim) * 4
	if bytes > maxBytes {
		return result, fmt.Errorf("attention tape byte ceiling")
	}
	for _, values := range [][]float32{q, k, v, dy} {
		if len(values) != n || !finiteF32(values) {
			return result, fmt.Errorf("attention tape input shape/nonfinite")
		}
	}
	table := make([]float32, rows*dim)
	if err := train.RoPETableInto(table, rows, dim, origin, maxPeriod); err != nil {
		return result, err
	}
	tensorShape, denseShape := []int{rows, heads, dim}, []int{heads, rows, rows}
	t := train.Tape{}
	add := func(shape []int, values []float32) int {
		id := len(t.Tensors)
		t.Tensors = append(t.Tensors, train.TensorSpec{Shape: append([]int(nil), shape...), Values: values})
		return id
	}
	input := func(values []float32) int { return add(tensorShape, append([]float32(nil), values...)) }
	qi, ki, vi, di := input(q), input(k), input(v), input(dy)
	ti := add([]int{rows, dim / 2, 2}, table)
	scratch := add(denseShape, make([]float32, dense))
	command := func(op string, inputs []int, shape []int, mode uint32) int {
		id := add(shape, nil)
		t.Commands = append(t.Commands, train.Command{Op: op, Inputs: inputs, Output: id, Mode: mode, Offset: contextRows, Scale: scale})
		return id
	}
	qr, kr := command("rope", []int{qi, ti}, tensorShape, 0), command("rope", []int{ki, ti}, tensorShape, 0)
	scores := command("attention-forward", []int{qr, kr, vi, scratch}, denseShape, 0)
	prob := command("attention-forward", []int{qr, kr, vi, scores}, denseShape, 1)
	out := command("attention-forward", []int{qr, kr, vi, prob}, tensorShape, 2)
	backward := func(aux int, shape []int, mode uint32) int {
		return command("attention-backward", []int{qr, kr, vi, prob, di, aux}, shape, mode)
	}
	dp := backward(scratch, denseShape, 0)
	ds := backward(dp, denseShape, 1)
	dq, dk, dv := backward(ds, tensorShape, 2), backward(ds, tensorShape, 3), backward(scratch, tensorShape, 4)
	dqi, dki := command("rope", []int{dq, ti}, tensorShape, 1), command("rope", []int{dk, ti}, tensorShape, 1)
	t.Outputs = []int{out, prob, dqi, dki, dv}
	if err := t.Validate(maxBytes); err != nil {
		return result, err
	}
	return AttentionTrainingTape{Tape: t, rows: rows, heads: heads, dim: dim}, nil
}
func (t AttentionTrainingTape) Decode(values [][]float32) (AttentionTrainingResult, error) {
	var result AttentionTrainingResult
	if len(values) != 5 || t.rows <= 0 || t.heads <= 0 || t.dim <= 0 {
		return result, fmt.Errorf("attention tape output count/metadata")
	}
	n, dense := t.rows*t.heads*t.dim, t.heads*t.rows*t.rows
	for i, x := range values {
		want := n
		if i == 1 {
			want = dense
		}
		if len(x) != want || !finiteF32(x) {
			return result, fmt.Errorf("attention tape output%d shape/nonfinite", i)
		}
	}
	own := func(i int) []float32 { return append([]float32(nil), values[i]...) }
	return AttentionTrainingResult{Output: own(0), Probability: own(1), DQ: own(2), DK: own(3), DV: own(4)}, nil
}
