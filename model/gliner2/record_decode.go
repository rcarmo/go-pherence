package gliner2

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
)

// DecodedRecord supports optional scalar/list and required scalar fields.
// Exclusive cardinalities are not represented by RecordField yet.
type DecodedRecord struct {
	Confidence float64             `json:"confidence"`
	Fields     map[string][]Entity `json:"fields"`
}

func DecodeRecords(text string, s RecordScores, c BoundaryHeadConfig) ([]DecodedRecord, error) {
	g := s.Group
	if err := g.Spec.Validate(); err != nil {
		return nil, err
	}
	if err := validateWords(text, s.Input.Words); err != nil {
		return nil, err
	}
	if c.RecordTemperature <= 0 || math.IsNaN(c.RecordTemperature) || math.IsInf(c.RecordTemperature, 0) {
		return nil, fmt.Errorf("invalid record temperature")
	}
	for _, v := range []float64{c.RecordAnchorThreshold, c.RecordFieldThreshold} {
		if v < 0 || v > 1 || math.IsNaN(v) {
			return nil, fmt.Errorf("invalid record threshold")
		}
	}
	n, f, p := len(g.ObjectLogits), len(g.Fields), len(g.PoolSpans)
	if len(g.InstanceMask) != n || len(g.AssignLogits) != n || len(g.InstancePoolIndex) != n || len(g.FieldMembership) != f || len(s.CandidateLogits) != p {
		return nil, fmt.Errorf("record group dimensions differ")
	}
	for i := range g.FieldMembership {
		if len(g.FieldMembership[i]) != p {
			return nil, fmt.Errorf("membership dimensions differ")
		}
	}
	for i := range g.AssignLogits {
		if len(g.AssignLogits[i]) != f {
			return nil, fmt.Errorf("assignment field count")
		}
		for _, row := range g.AssignLogits[i] {
			if len(row) != p+1 {
				return nil, fmt.Errorf("assignment candidate count")
			}
			for _, v := range row {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return nil, fmt.Errorf("nonfinite assignment")
				}
			}
		}
	}
	order := make([]int, 0, n)
	for i, v := range g.ObjectLogits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("nonfinite object")
		}
		if g.InstanceMask[i] && sigmoid(float64(v)/c.RecordTemperature) >= c.RecordAnchorThreshold {
			order = append(order, i)
		}
	}
	sort.SliceStable(order, func(i, j int) bool { return g.ObjectLogits[order[i]] > g.ObjectLogits[order[j]] })
	records := []DecodedRecord{}
	seen := map[string]bool{}
	for _, inst := range order {
		rec := DecodedRecord{sigmoid(float64(g.ObjectLogits[inst]) / c.RecordTemperature), map[string][]Entity{}}
		for fi, field := range g.Fields {
			selected := map[int]float64{}
			if g.Spec.Mode == RecordModeNatural && field.QueryID == g.Spec.AnchorQueryID {
				idx := g.InstancePoolIndex[inst]
				if idx >= 0 && idx < p && g.FieldMembership[fi][idx] {
					selected[idx] = rec.Confidence
				}
			} else if field.Scalar {
				row := g.AssignLogits[inst][fi]
				best := 0
				mx := float64(row[0]) / c.RecordTemperature
				if field.Required {
					best = -1
					mx = math.Inf(-1)
				}
				for i := 1; i < len(row); i++ {
					if g.FieldMembership[fi][i-1] && float64(row[i])/c.RecordTemperature > mx {
						mx = float64(row[i]) / c.RecordTemperature
						best = i
					}
				}
				if best > 0 {
					// A required field can skip a much larger null logit. Use
					// a separate softmax maximum to avoid exp overflow.
					normMax := math.Max(mx, float64(row[0])/c.RecordTemperature)
					sum := math.Exp(float64(row[0])/c.RecordTemperature - normMax)
					for i := 1; i < len(row); i++ {
						if g.FieldMembership[fi][i-1] {
							sum += math.Exp(float64(row[i])/c.RecordTemperature - normMax)
						}
					}
					prob := math.Exp(mx-normMax) / sum
					if field.Required || prob >= c.RecordFieldThreshold {
						selected[best-1] = prob
					}
				}
			} else {
				for i, valid := range g.FieldMembership[fi] {
					if valid {
						prob := sigmoid(float64(g.AssignLogits[inst][fi][i+1]) / c.RecordTemperature)
						if prob >= c.RecordFieldThreshold {
							selected[i] = prob
						}
					}
				}
			}
			// Reuse checked span formatting and overlap resolution without changing
			// the assignment confidence or losing original character offsets.
			scores := EntityScores{Input: s.Input, Candidates: PooledCandidates{Indices: g.PoolSpans, ValidMask: make([]bool, p)}, Logits: make([][]float32, p)}
			scores.Input.Labels = []string{field.Name}
			for i := 0; i < p; i++ {
				scores.Logits[i] = []float32{0}
			}
			for i, prob := range selected {
				if field.QueryID < 0 || field.QueryID >= len(s.CandidateLogits[i]) {
					return nil, fmt.Errorf("field query outside candidate logits")
				}
				candidateProb := sigmoid(float64(s.CandidateLogits[i][field.QueryID]))
				if !field.Required && candidateProb < c.RecordAnchorThreshold {
					continue
				}
				prob = math.Min(prob, candidateProb)
				scores.Candidates.ValidMask[i] = true
				prob = math.Max(1e-15, math.Min(1-1e-15, prob))
				scores.Logits[i][0] = float32(math.Log(prob / (1 - prob)))
			}
			entities, err := DecodeEntities(text, scores, 0, c.OverlapPolicy)
			if err != nil {
				return nil, err
			}
			if field.Scalar && len(entities) > 1 {
				entities = entities[:1]
			}
			if len(entities) > 0 {
				rec.Fields[field.Name] = entities
			}
		}
		if len(rec.Fields) == 0 {
			continue
		}
		if g.Spec.Mode != RecordModeNatural {
			keyFields := map[string][][2]int{}
			for name, es := range rec.Fields {
				for _, e := range es {
					keyFields[name] = append(keyFields[name], [2]int{e.TokenStart, e.TokenEnd})
				}
			}
			key, _ := json.Marshal(keyFields)
			if seen[string(key)] {
				continue
			}
			seen[string(key)] = true
		}
		records = append(records, rec)
	}
	return records, nil
}
