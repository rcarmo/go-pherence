package mojev

import (
	"fmt"
	"math"
	"sort"
	"strconv"
)

// AssembleAnswer maps injected logits from text-sorted candidate order back to
// the caller's Choice, Noul or Score order. It never executes the encoder.
// keys and options are in caller order; logits are in sorted option order.
// The returned maps are owned. Floating-point results use F64 as in the
// upstream public softmax, not the F32 scorer head's masked logit type.
func AssembleAnswer(kind string, keys, options []string, sortedLogits []float64) (map[string]any, error) {
	n := len(options)
	if len(sortedLogits) != n {
		return nil, fmt.Errorf("mojev: invalid answer cardinality")
	}
	if err := validateAnswerLabels(kind, keys, options); err != nil {
		return nil, err
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return options[order[i]] < options[order[j]] })
	peak := math.Inf(-1)
	for _, logit := range sortedLogits {
		if math.IsInf(logit, 0) || math.IsNaN(logit) {
			return nil, fmt.Errorf("mojev: non-finite logit")
		}
		if logit > peak {
			peak = logit
		}
	}
	prob := make([]float64, n)
	var sum float64
	for i, v := range sortedLogits {
		prob[i] = math.Exp(v - peak)
		sum += prob[i]
	}
	if sum == 0 || math.IsInf(sum, 0) || math.IsNaN(sum) {
		return nil, fmt.Errorf("mojev: invalid logit sum")
	}
	caller := make([]float64, n)
	for sorted, original := range order {
		caller[original] = prob[sorted] / sum
	}
	if kind == "noul" {
		return map[string]any{"type": "noul", "noul": caller[1]}, nil
	}
	mapped := make(map[string]float64, n)
	for i, key := range keys {
		mapped[key] = caller[i]
	}
	if kind == "choice" {
		best := 0
		for i := 1; i < n; i++ {
			if caller[i] > caller[best] {
				best = i
			}
		}
		return map[string]any{"type": "choice", "choice": keys[best], "confidence": caller[best], "probabilities": mapped}, nil
	}
	legend := make(map[string]string, n)
	var expected, confidence float64
	for i, key := range keys {
		legend[key] = options[i]
		expected += float64(i) * caller[i]
		if i == 0 || caller[i] > confidence {
			confidence = caller[i]
		}
	}
	return map[string]any{"type": "score", "score": expected, "confidence": confidence, "legend": legend, "probabilities": mapped}, nil
}

func validateAnswerLabels(kind string, keys, options []string) error {
	n := len(options)
	if n < 2 || n > 64 || len(keys) != n {
		return fmt.Errorf("mojev: invalid answer cardinality")
	}
	if kind != "choice" && kind != "noul" && kind != "score" {
		return fmt.Errorf("mojev: unsupported answer kind")
	}
	if kind == "noul" && (n != 2 || keys[0] != "false" || keys[1] != "true") {
		return fmt.Errorf("mojev: invalid noul candidates")
	}
	seenKeys, seenOptions := make(map[string]bool, n), make(map[string]bool, n)
	for i, key := range keys {
		if key == "" || options[i] == "" || seenKeys[key] || seenOptions[options[i]] {
			return fmt.Errorf("mojev: empty or duplicate answer label")
		}
		if kind == "score" && key != strconv.Itoa(i) {
			return fmt.Errorf("mojev: unordered score levels")
		}
		seenKeys[key], seenOptions[options[i]] = true, true
	}
	return nil
}
