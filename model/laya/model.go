package laya

import (
	"fmt"
	"math"
	"strings"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/model/modernbert"
)

type Tensor struct {
	Shape []int
	Data  []float32
}

type QuestionType int

const (
	Choice QuestionType = iota
	Score
	Noul
)

type Output struct {
	Logits        []float32
	ActionLogits  []float32
	Probabilities []float32
	Confidence    float32
}

type layer struct {
	n1w, n1b, qkv, qkvb, outw, outb, n2w, n2b, ff1, ff1b, ff2, ff2b []float32
}

type Model struct {
	Encoder                                                                                                 *modernbert.Model
	typeEmb, temp, scoreNormW, scoreNormB, scoreW, scoreB, scoreOut, scoreOutB, actW, actB, actOut, actOutB []float32
	layers                                                                                                  []layer
	tempByOptions                                                                                           map[string]float32
	hidden, actions                                                                                         int
}

// New validates and copies all caller-owned tensors.
func New(enc *modernbert.Model, tensors map[string]Tensor) (*Model, error) {
	return newModel(enc, tensors, true)
}

func newModel(enc *modernbert.Model, tensors map[string]Tensor, copyData bool) (*Model, error) {
	if enc == nil {
		return nil, fmt.Errorf("laya: nil encoder")
	}
	h := enc.Config.HiddenSize
	used := make(map[string]bool, len(tensors))
	get := func(name string, shape ...int) ([]float32, error) {
		v, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("laya: missing %s", name)
		}
		if len(v.Shape) != len(shape) {
			return nil, fmt.Errorf("laya: %s rank", name)
		}
		size := 1
		for i, d := range shape {
			if d < 1 || v.Shape[i] != d {
				return nil, fmt.Errorf("laya: %s shape", name)
			}
			if size > int(^uint(0)>>1)/d {
				return nil, fmt.Errorf("laya: %s size overflow", name)
			}
			size *= d
		}
		if len(v.Data) != size {
			return nil, fmt.Errorf("laya: %s length", name)
		}
		used[name] = true
		if copyData {
			return append([]float32(nil), v.Data...), nil
		}
		return v.Data, nil
	}
	m := &Model{Encoder: enc, hidden: h}
	var err error
	assign := func(dst *[]float32, name string, shape ...int) bool {
		*dst, err = get(name, shape...)
		return err == nil
	}
	if !assign(&m.typeEmb, "type_emb.weight", 3, h) || !assign(&m.temp, "temperature", 3) ||
		!assign(&m.scoreNormW, "scorer.0.weight", h) || !assign(&m.scoreNormB, "scorer.0.bias", h) ||
		!assign(&m.scoreW, "scorer.1.weight", h, h) || !assign(&m.scoreB, "scorer.1.bias", h) ||
		!assign(&m.scoreOut, "scorer.3.weight", 1, h) || !assign(&m.scoreOutB, "scorer.3.bias", 1) {
		return nil, err
	}
	for i := 0; ; i++ {
		prefix := fmt.Sprintf("head.layers.%d.", i)
		if _, ok := tensors[prefix+"self_attn.in_proj_weight"]; !ok {
			break
		}
		var w layer
		if !assign(&w.n1w, prefix+"norm1.weight", h) || !assign(&w.n1b, prefix+"norm1.bias", h) ||
			!assign(&w.qkv, prefix+"self_attn.in_proj_weight", 3*h, h) || !assign(&w.qkvb, prefix+"self_attn.in_proj_bias", 3*h) ||
			!assign(&w.outw, prefix+"self_attn.out_proj.weight", h, h) || !assign(&w.outb, prefix+"self_attn.out_proj.bias", h) ||
			!assign(&w.n2w, prefix+"norm2.weight", h) || !assign(&w.n2b, prefix+"norm2.bias", h) ||
			!assign(&w.ff1, prefix+"linear1.weight", 4*h, h) || !assign(&w.ff1b, prefix+"linear1.bias", 4*h) ||
			!assign(&w.ff2, prefix+"linear2.weight", h, 4*h) || !assign(&w.ff2b, prefix+"linear2.bias", h) {
			return nil, err
		}
		m.layers = append(m.layers, w)
	}
	if len(m.layers) == 0 {
		return nil, fmt.Errorf("laya: no head layers")
	}
	a, ok := tensors["act_head.2.weight"]
	if !ok || len(a.Shape) != 2 || a.Shape[0] < 1 || a.Shape[1] != 256 {
		return nil, fmt.Errorf("laya: action head shape")
	}
	m.actions = a.Shape[0]
	if !assign(&m.actW, "act_head.0.weight", 256, h+4) || !assign(&m.actB, "act_head.0.bias", 256) ||
		!assign(&m.actOut, "act_head.2.weight", m.actions, 256) || !assign(&m.actOutB, "act_head.2.bias", m.actions) {
		return nil, err
	}
	for name := range tensors {
		if !used[name] && !strings.HasSuffix(name, "num_batches_tracked") {
			return nil, fmt.Errorf("laya: unexpected tensor %s", name)
		}
	}
	return m, nil
}

func ln(dst, src, w, b []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		x := src[r*cols : (r+1)*cols]
		var mean float32
		for _, v := range x {
			mean += v
		}
		mean /= float32(cols)
		var ss float32
		for _, v := range x {
			d := v - mean
			ss += d * d
		}
		inv := float32(1 / math.Sqrt(float64(ss/float32(cols)+1e-5)))
		for j, v := range x {
			dst[r*cols+j] = (v-mean)*inv*w[j] + b[j]
		}
	}
}

func dense(dst, x, w, b []float32, rows, out, in int) {
	clear(dst)
	simd.DenseNTTo(dst, x, w, rows, out, in, 1, in, in, out)
	if b != nil {
		simd.AddBiasRowsTo(dst, b, rows, out)
	}
}

// Forward is the convenient allocating API. Use Session.ForwardInto for a warm zero-allocation path.
func (m *Model) Forward(ids []int, mask []bool, markers []int, qtype QuestionType) (Output, error) {
	if m == nil {
		return Output{}, fmt.Errorf("laya: nil model")
	}
	s, err := m.NewSession(len(ids), len(markers))
	if err != nil {
		return Output{}, err
	}
	o := Output{Logits: make([]float32, len(markers)), ActionLogits: make([]float32, m.actions), Probabilities: make([]float32, len(markers))}
	o.Confidence, err = s.ForwardInto(o.Logits, o.ActionLogits, o.Probabilities, ids, mask, markers, qtype)
	return o, err
}
