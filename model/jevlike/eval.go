package jevlike

import (
	"fmt"
	"math"
	"sort"
)

// Metrics reports accuracy and ten-bin expected calibration error.
type Metrics struct {
	Top1     float64 `json:"top1"`
	Top3     float64 `json:"top3"`
	ECE      float64 `json:"ece"`
	Examples int     `json:"examples"`
}

type Evaluation struct {
	Model           Metrics `json:"model"`
	ShuffledContext Metrics `json:"shuffled_context"`
}

// ForwardShuffled rolls contexts and their masks by one example within the
// batch, retaining each row's options. Singleton batches are unchanged.
func (m *TinyScorer) ForwardShuffled(batch ByteBatch) ([][]float32, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	rows, ct, no, nt, err := validateBatchForScorer(batch, m.Config)
	if err != nil {
		return nil, err
	}
	ctx, opts, _ := encodeTinyBatch(m, batch, rows, ct, no, nt)
	rolled := make([][][]float32, rows)
	masks := make([][]bool, rows)
	for i := 0; i < rows; i++ {
		j := (i + rows - 1) % rows
		rolled[i] = ctx[j]
		masks[i] = batch.ContextMask[j]
	}
	return m.Head.Forward(rolled, masks, opts, batch.OptionMask)
}

// EvaluateTiny evaluates both ordinary and rolled-context predictions. Batching
// is significant to the control, exactly as in upstream's batch-local roll.
func EvaluateTiny(m *TinyScorer, examples []ChoiceExample, batchSize int) (Evaluation, error) {
	if len(examples) == 0 || batchSize <= 0 {
		return Evaluation{}, fmt.Errorf("evaluation requires examples and positive batch size")
	}
	var all, shuffled [][]float32
	labels := make([]int, 0, len(examples))
	for start := 0; start < len(examples); start += batchSize {
		end := min(start+batchSize, len(examples))
		b, err := BuildByteBatch(examples[start:end], m.Config.ContextTokens, m.Config.OptionTokens)
		if err != nil {
			return Evaluation{}, err
		}
		normal, err := m.Forward(b)
		if err != nil {
			return Evaluation{}, err
		}
		control, err := m.ForwardShuffled(b)
		if err != nil {
			return Evaluation{}, err
		}
		for i, ex := range examples[start:end] {
			all = append(all, normal[i][:len(ex.Options)])
			shuffled = append(shuffled, control[i][:len(ex.Options)])
			labels = append(labels, ex.Label)
		}
	}
	a, err := ComputeMetrics(all, labels)
	if err != nil {
		return Evaluation{}, err
	}
	b, err := ComputeMetrics(shuffled, labels)
	return Evaluation{Model: a, ShuffledContext: b}, err
}

// ComputeMetrics ignores no implicit padding: each row must contain only its
// valid option logits. Ties use the first option, deterministically.
func ComputeMetrics(logits [][]float32, labels []int) (Metrics, error) {
	if len(logits) == 0 || len(labels) != len(logits) {
		return Metrics{}, fmt.Errorf("invalid metric row count")
	}
	var bins [10]struct {
		count, correct int
		confidence     float64
	}
	out := Metrics{Examples: len(labels)}
	for row, x := range logits {
		if len(x) < 2 || labels[row] < 0 || labels[row] >= len(x) {
			return Metrics{}, fmt.Errorf("invalid metric row %d", row)
		}
		for _, v := range x {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return Metrics{}, fmt.Errorf("nonfinite logit in row %d", row)
			}
		}
		ids := make([]int, len(x))
		for i := range ids {
			ids[i] = i
		}
		sort.SliceStable(ids, func(i, j int) bool { return x[ids[i]] > x[ids[j]] })
		correct := ids[0] == labels[row]
		if correct {
			out.Top1++
		}
		for _, id := range ids[:min(3, len(ids))] {
			if id == labels[row] {
				out.Top3++
				break
			}
		}
		p := softmaxFloat32(x)
		confidence := float64(p[ids[0]])
		// Include confidence==1 in the last bin rather than losing perfectly
		// confident predictions at the upstream half-open upper boundary.
		bin := min(9, int(confidence*10))
		bins[bin].count++
		bins[bin].confidence += confidence
		if correct {
			bins[bin].correct++
		}
	}
	n := float64(out.Examples)
	out.Top1 /= n
	out.Top3 /= n
	for _, b := range bins {
		if b.count > 0 {
			out.ECE += math.Abs(float64(b.correct)-b.confidence) / n
		}
	}
	return out, nil
}
