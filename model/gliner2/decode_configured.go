package gliner2

import (
	"fmt"
	"math"
	"sort"
)

// DecodeConfiguredEntities applies checkpoint temperature, optional Poisson
// count guidance and null-head abstention before the shared overlap resolver.
func DecodeConfiguredEntities(text string, s EntityScores, threshold float64, policy string, c BoundaryHeadConfig) ([]Entity, error) {
	if err := validateDecodeInput(text, s, threshold); err != nil {
		return nil, err
	}
	if _, err := normalizeOverlapPolicy(policy); err != nil {
		return nil, err
	}
	if c.PairTemperature <= 0 || math.IsNaN(c.PairTemperature) || math.IsInf(c.PairTemperature, 0) {
		return nil, fmt.Errorf("invalid pair temperature")
	}
	qn := len(s.Input.Labels)
	if c.EnableAbstention && len(s.NullLogits) != qn {
		return nil, fmt.Errorf("missing null logits")
	}
	if c.AdaptiveThreshold && len(s.CountLogits) != qn {
		return nil, fmt.Errorf("missing count log rates")
	}
	if c.AbstentionThreshold < 0 || c.AbstentionThreshold > 1 || math.IsNaN(c.AbstentionThreshold) {
		return nil, fmt.Errorf("invalid abstention threshold")
	}
	var result []Entity
	for q := 0; q < qn; q++ {
		if c.EnableAbstention {
			v := float64(s.NullLogits[q])
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("nonfinite null logit")
			}
			if sigmoid(v) > c.AbstentionThreshold {
				continue
			}
		}
		one := s
		one.Input.Labels = []string{s.Input.Labels[q]}
		one.Logits = make([][]float32, len(s.Logits))
		one.Candidates.ValidMask = append([]bool(nil), s.Candidates.ValidMask...)
		ids := make([]int, len(s.Logits))
		for i, row := range s.Logits {
			one.Logits[i] = []float32{row[q] / float32(c.PairTemperature)}
			ids[i] = i
		}
		sort.SliceStable(ids, func(i, j int) bool {
			a, b := ids[i], ids[j]
			if one.Candidates.ValidMask[a] != one.Candidates.ValidMask[b] {
				return one.Candidates.ValidMask[a]
			}
			return one.Logits[a][0] > one.Logits[b][0]
		})
		count := 0
		if c.AdaptiveThreshold {
			rate := float64(s.CountLogits[q])
			if math.IsNaN(rate) || math.IsInf(rate, 0) {
				return nil, fmt.Errorf("nonfinite count log rate")
			}
			predicted := math.RoundToEven(math.Exp(rate))
			if predicted >= float64(len(ids)) {
				count = len(ids)
			} else {
				count = int(predicted)
			}
		}
		keep := make([]bool, len(ids))
		for rank, id := range ids {
			keep[id] = one.Candidates.ValidMask[id] && (sigmoid(float64(one.Logits[id][0])) >= threshold || rank < count)
		}
		one.Candidates.ValidMask = keep
		// Selection was already performed; threshold zero preserves probabilities
		// while allowing count-guided candidates through the common resolver.
		entities, err := DecodeEntities(text, one, 0, policy)
		if err != nil {
			return nil, err
		}
		result = append(result, entities...)
	}
	sort.SliceStable(result, func(i, j int) bool {
		a, b := result[i], result[j]
		if a.Confidence != b.Confidence {
			return a.Confidence > b.Confidence
		}
		if a.TokenStart != b.TokenStart {
			return a.TokenStart < b.TokenStart
		}
		if a.TokenEnd != b.TokenEnd {
			return a.TokenEnd < b.TokenEnd
		}
		return a.Label < b.Label
	})
	return result, nil
}
