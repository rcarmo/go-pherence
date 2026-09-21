package jevlike

import (
	"fmt"
	"math"
	"sort"
)

type ReliabilityBin struct {
	Count      int     `json:"count"`
	Accuracy   float64 `json:"accuracy"`
	Confidence float64 `json:"confidence"`
}
type CoveragePoint struct {
	Count         int     `json:"count"`
	Coverage      float64 `json:"coverage"`
	ErrorRate     float64 `json:"error_rate"`
	ErrorLower95  float64 `json:"error_lower_95"`
	ErrorUpper95  float64 `json:"error_upper_95"`
	MinConfidence float64 `json:"min_confidence"`
}
type DecisionMetrics struct {
	Examples            int              `json:"examples"`
	Accuracy            float64          `json:"accuracy"`
	NLL                 float64          `json:"nll"`
	Brier               float64          `json:"brier"` // sum over candidates; no division by choice count
	ECE                 float64          `json:"ece"`
	RandomAccuracy      float64          `json:"random_accuracy"`
	FirstOptionAccuracy float64          `json:"first_option_accuracy"`
	Reliability         []ReliabilityBin `json:"reliability_10_equal_width"`
	Coverage            []CoveragePoint  `json:"risk_coverage"`
}

// DecisionMetricsAtTemperature consumes only genuine unpadded candidate logits.
// Temperature must be frozen from a separate calibration partition.
func DecisionMetricsAtTemperature(logits [][]float32, labels []int, temp float64) (DecisionMetrics, error) {
	if len(logits) == 0 || len(labels) != len(logits) || temp <= 0 || math.IsNaN(temp) || math.IsInf(temp, 0) {
		return DecisionMetrics{}, fmt.Errorf("invalid metric shape/temperature")
	}
	out := DecisionMetrics{Examples: len(logits), Reliability: make([]ReliabilityBin, 10)}
	type observation struct {
		confidence float64
		wrong      bool
		index      int
	}
	obs := make([]observation, len(logits))
	for i, row := range logits {
		if len(row) < 2 || labels[i] < 0 || labels[i] >= len(row) {
			return DecisionMetrics{}, fmt.Errorf("invalid label/candidates at %d", i)
		}
		best := 0
		maxValue := math.Inf(-1)
		for j, l := range row {
			if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
				return DecisionMetrics{}, fmt.Errorf("nonfinite logits")
			}
			if float64(l) > maxValue {
				maxValue = float64(l)
				best = j
			}
		}
		prob := make([]float64, len(row))
		var sum float64
		for j, l := range row {
			prob[j] = math.Exp((float64(l) - maxValue) / temp)
			sum += prob[j]
		}
		loss := (maxValue-float64(row[labels[i]]))/temp + math.Log(sum)
		if math.IsInf(loss, 0) {
			return DecisionMetrics{}, fmt.Errorf("temperature yields nonfinite NLL")
		}
		out.NLL += loss
		for j := range prob {
			prob[j] /= sum
			target := float64(0)
			if j == labels[i] {
				target = 1
			}
			d := prob[j] - target
			out.Brier += d * d
		}
		correct := best == labels[i]
		if correct {
			out.Accuracy++
		}
		if labels[i] == 0 {
			out.FirstOptionAccuracy++
		}
		out.RandomAccuracy += 1 / float64(len(row))
		confidence := prob[best]
		bin := min(9, int(confidence*10))
		out.Reliability[bin].Count++
		out.Reliability[bin].Confidence += confidence
		if correct {
			out.Reliability[bin].Accuracy++
		}
		obs[i] = observation{confidence, !correct, i}
	}
	n := float64(len(logits))
	out.Accuracy /= n
	out.NLL /= n
	out.Brier /= n
	out.RandomAccuracy /= n
	out.FirstOptionAccuracy /= n
	for i := range out.Reliability {
		b := &out.Reliability[i]
		if b.Count > 0 {
			b.Accuracy /= float64(b.Count)
			b.Confidence /= float64(b.Count)
			out.ECE += float64(b.Count) / n * math.Abs(b.Accuracy-b.Confidence)
		}
	}
	sort.SliceStable(obs, func(i, j int) bool { return obs[i].confidence > obs[j].confidence })
	wrong := 0
	for i, o := range obs {
		if o.wrong {
			wrong++
		}
		count := i + 1
		// Equal-confidence observations must enter coverage together.
		if count < len(obs) && obs[count].confidence == o.confidence {
			continue
		}
		lo, hi := wilson(float64(wrong), float64(count))
		out.Coverage = append(out.Coverage, CoveragePoint{count, float64(count) / n, float64(wrong) / float64(count), lo, hi, o.confidence})
	}
	return out, nil
}

// DecisionMetricsFromProbabilities evaluates genuine, unpadded candidate
// probabilities. Rows are normalized so callers may provide independently
// scored positive masses, such as entailment probabilities from a reranker.
// It does not fit or apply a calibration parameter.
func DecisionMetricsFromProbabilities(probabilities [][]float32, labels []int) (DecisionMetrics, error) {
	if len(probabilities) == 0 || len(labels) != len(probabilities) {
		return DecisionMetrics{}, fmt.Errorf("invalid metric shape")
	}
	normalized := make([][]float64, len(probabilities))
	for i, row := range probabilities {
		if len(row) < 2 || labels[i] < 0 || labels[i] >= len(row) {
			return DecisionMetrics{}, fmt.Errorf("invalid label/candidates at %d", i)
		}
		normalized[i] = make([]float64, len(row))
		var sum float64
		for j, value := range row {
			if value < 0 || math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return DecisionMetrics{}, fmt.Errorf("invalid probability at %d", i)
			}
			normalized[i][j] = float64(value)
			sum += float64(value)
		}
		if sum <= 0 {
			return DecisionMetrics{}, fmt.Errorf("zero probability mass at %d", i)
		}
		for j := range normalized[i] {
			normalized[i][j] /= sum
		}
	}
	return decisionMetricsFromNormalizedProbabilities(normalized, labels)
}

func decisionMetricsFromNormalizedProbabilities(probabilities [][]float64, labels []int) (DecisionMetrics, error) {
	out := DecisionMetrics{Examples: len(probabilities), Reliability: make([]ReliabilityBin, 10)}
	type observation struct {
		confidence float64
		wrong      bool
	}
	obs := make([]observation, len(probabilities))
	for i, probability := range probabilities {
		best := 0
		for j, value := range probability {
			if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
				return DecisionMetrics{}, fmt.Errorf("invalid normalized probability at %d", i)
			}
			if value > probability[best] {
				best = j
			}
			target := float64(0)
			if j == labels[i] {
				target = 1
			}
			delta := value - target
			out.Brier += delta * delta
		}
		gold := probability[labels[i]]
		if gold <= 0 {
			return DecisionMetrics{}, fmt.Errorf("zero gold probability at %d", i)
		}
		out.NLL -= math.Log(gold)
		correct := best == labels[i]
		if correct {
			out.Accuracy++
		}
		if labels[i] == 0 {
			out.FirstOptionAccuracy++
		}
		out.RandomAccuracy += 1 / float64(len(probability))
		confidence := probability[best]
		bin := min(9, int(confidence*10))
		out.Reliability[bin].Count++
		out.Reliability[bin].Confidence += confidence
		if correct {
			out.Reliability[bin].Accuracy++
		}
		obs[i] = observation{confidence, !correct}
	}
	n := float64(len(probabilities))
	out.Accuracy /= n
	out.NLL /= n
	out.Brier /= n
	out.RandomAccuracy /= n
	out.FirstOptionAccuracy /= n
	for i := range out.Reliability {
		bin := &out.Reliability[i]
		if bin.Count > 0 {
			bin.Accuracy /= float64(bin.Count)
			bin.Confidence /= float64(bin.Count)
			out.ECE += float64(bin.Count) / n * math.Abs(bin.Accuracy-bin.Confidence)
		}
	}
	sort.SliceStable(obs, func(i, j int) bool { return obs[i].confidence > obs[j].confidence })
	wrong := 0
	for i, o := range obs {
		if o.wrong {
			wrong++
		}
		count := i + 1
		if count < len(obs) && obs[count].confidence == o.confidence {
			continue
		}
		lo, hi := wilson(float64(wrong), float64(count))
		out.Coverage = append(out.Coverage, CoveragePoint{count, float64(count) / n, float64(wrong) / float64(count), lo, hi, o.confidence})
	}
	return out, nil
}
func wilson(successes, n float64) (float64, float64) {
	const z = 1.959963984540054
	p := successes / n
	den := 1 + z*z/n
	mid := (p + z*z/(2*n)) / den
	half := z * math.Sqrt(p*(1-p)/n+z*z/(4*n*n)) / den
	return math.Max(0, mid-half), math.Min(1, mid+half)
}

// FitChoiceTemperature minimises calibration NLL in a fixed [0.05,20] range.
// Callers must pass calibration-only rows and persist the returned parameter.
func FitChoiceTemperature(logits [][]float32, labels []int) (float64, error) {
	if _, err := DecisionMetricsAtTemperature(logits, labels, 1); err != nil {
		return 0, err
	}
	left, right := math.Log(0.05), math.Log(20.0)
	const ratio = 0.6180339887498949
	loss := func(x float64) float64 {
		m, e := DecisionMetricsAtTemperature(logits, labels, math.Exp(x))
		if e != nil {
			return math.Inf(1)
		}
		return m.NLL
	}
	a, b := right-ratio*(right-left), left+ratio*(right-left)
	fa, fb := loss(a), loss(b)
	for range 80 {
		if fa < fb {
			right = b
			b = a
			fb = fa
			a = right - ratio*(right-left)
			fa = loss(a)
		} else {
			left = a
			a = b
			fa = fb
			b = left + ratio*(right-left)
			fb = loss(b)
		}
	}
	return math.Exp((left + right) / 2), nil
}
