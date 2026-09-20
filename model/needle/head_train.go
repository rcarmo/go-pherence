package needle

import (
	"fmt"
	"math"
	"strings"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

// HeadLossGrad freezes the language trunk as upstream _head_cells does. The
// caller selects a supervised objective, not a confidence-calibration claim:
// confidence BCE with target[0] in [0,1]; router CE with one integer class;
// embedding/contrastive MSE against a supplied normalized target vector (not
// paired contrastive/InfoNCE training).
// This is not the upstream unpublished end-to-end head dataset/training recipe.
func (m *Model) HeadLossGrad(ids []int, kind HeadKind, target []float32, opts Options) (loss float64, grads map[string]checkpoint.Tensor, err error) {
	defer recoverWork(&err)
	if err = m.validateTokens(ids); err != nil {
		return 0, nil, err
	}
	if m.deployed || opts.Packed {
		return 0, nil, fmt.Errorf("needle: head training requires source checkpoint")
	}
	if err = opts.Quant.validate(m.config.Generation); err != nil {
		return 0, nil, err
	}
	_, _, dim, err := m.headGeometry(kind)
	if err != nil {
		return 0, nil, err
	}
	for _, v := range target {
		if !finite(v) {
			return 0, nil, fmt.Errorf("needle: nonfinite head target")
		}
	}
	switch kind {
	case Confidence:
		if len(target) != 1 || target[0] < 0 || target[0] > 1 {
			return 0, nil, fmt.Errorf("needle: confidence target must be one probability")
		}
	case Router:
		if len(target) != 1 || target[0] < 0 || target[0] >= float32(dim) || target[0] != float32(int(target[0])) {
			return 0, nil, fmt.Errorf("needle: router target must be one class index")
		}
	case Embedding, Contrastive:
		if len(target) != dim {
			return 0, nil, fmt.Errorf("needle: embedding target dimension mismatch")
		}
		var norm float64
		for _, v := range target {
			norm += float64(v) * float64(v)
		}
		if math.Abs(norm-1) > 1e-3 {
			return 0, nil, fmt.Errorf("needle: embedding target must have unit norm")
		}
	}
	e := m.execution(false, opts)
	e.keep = make([]bool, len(ids))
	e.t.reserve(int64(len(ids)) + int64(m.config.Layers+1)*8)
	count := 0
	for i, id := range ids {
		e.keep[i] = id != m.config.PadID
		if e.keep[i] {
			count++
		}
	}
	if count == 0 {
		return 0, nil, fmt.Errorf("needle: head input has only padding")
	}
	e.trunk(ids, true)
	// A fresh parameter set prevents gradients through the frozen trunk. Reuse
	// read-only cells but own gradient buffers so no nil cell gradient is written.
	e.p = map[string]*value{}
	e.qp = map[string]*value{}
	e.t.train = true
	for i, cell := range e.cells {
		e.cells[i] = e.t.leaf(cell.x)
		e.cells[i].r, e.cells[i].c = cell.r, cell.c
	}
	result := e.probeHead(kind)
	for _, v := range result.x {
		if !finite(v) {
			return 0, nil, fmt.Errorf("needle: nonfinite head result")
		}
	}
	switch kind {
	case Confidence:
		z := float64(result.x[0])
		y := float64(target[0])
		loss = math.Max(z, 0) - z*y + math.Log1p(math.Exp(-math.Abs(z)))
		result.g[0] = sigmoid(result.x[0]) - target[0]
	case Router:
		mx := result.x[0]
		for _, v := range result.x {
			mx = max(mx, v)
		}
		var sum float64
		for _, v := range result.x {
			sum += math.Exp(float64(v - mx))
		}
		label := int(target[0])
		loss = math.Log(sum) + float64(mx-result.x[label])
		for i, v := range result.x {
			result.g[i] = float32(math.Exp(float64(v-mx)) / sum)
		}
		result.g[label] -= 1
	case Embedding, Contrastive:
		for i, v := range result.x {
			delta := float64(v - target[i])
			loss += delta * delta / float64(dim)
			result.g[i] = float32(2 * delta / float64(dim))
		}
	}
	e.t.back()
	grads = map[string]checkpoint.Tensor{}
	for name, p := range e.p {
		if !strings.HasPrefix(name, string(kind)+"_head/") {
			continue
		}
		for _, v := range p.g {
			if !finite(v) {
				return 0, nil, fmt.Errorf("needle: nonfinite head gradient")
			}
		}
		grads[name] = checkpoint.Tensor{Shape: append([]int{}, m.tensors[name].Shape...), Data: p.g}
	}
	return loss, grads, nil
}

// TrainHeadStep updates only the selected head; other weights/calibration data
// are preserved. Use a separate optimizer for each objective/parameter set.
func (m *Model) TrainHeadStep(opt *AdamW, ids []int, kind HeadKind, target []float32, lr float64, opts Options) (*Model, float64, error) {
	loss, g, err := m.HeadLossGrad(ids, kind, target, opts)
	if err != nil {
		return nil, 0, err
	}
	weights := map[string]checkpoint.Tensor{}
	for key := range g {
		weights[key] = m.tensors[key]
	}
	updated, err := opt.update(weights, g, lr)
	if err != nil {
		return nil, 0, err
	}
	cp := m.checkpointView()
	for key, w := range updated {
		cp.Tensors[key] = w
	}
	next, err := newModel(cp, true)
	return next, loss, err
}
