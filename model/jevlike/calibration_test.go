package jevlike

import (
	"math"
	"testing"
)

func TestDecisionMetricsUniformAndPerfect(t *testing.T) {
	m, err := DecisionMetricsAtTemperature([][]float32{{0, 0}, {0, 0, 0, 0}}, []int{0, 1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if m.Accuracy != 0.5 || math.Abs(m.RandomAccuracy-0.375) > 1e-12 || math.Abs(m.Brier-0.625) > 1e-12 {
		t.Fatal(m)
	}
	if math.Abs(m.NLL-(math.Log(2)+math.Log(4))/2) > 1e-12 {
		t.Fatal(m)
	}
	perfect, err := DecisionMetricsAtTemperature([][]float32{{1000, 0}, {0, 1000}}, []int{0, 1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if perfect.Accuracy != 1 || perfect.Reliability[9].Count != 2 || perfect.Brier != 0 || perfect.NLL != 0 || len(perfect.Coverage) != 1 {
		t.Fatal(perfect)
	}
}
func TestTemperatureFitOnOverconfidentErrors(t *testing.T) {
	logits := [][]float32{{8, 0}, {8, 0}, {8, 0}, {8, 0}}
	labels := []int{0, 0, 0, 1}
	temp, err := FitChoiceTemperature(logits, labels)
	if err != nil {
		t.Fatal(err)
	}
	if temp <= 1 || temp > 20 {
		t.Fatal(temp)
	}
	before, _ := DecisionMetricsAtTemperature(logits, labels, 1)
	after, _ := DecisionMetricsAtTemperature(logits, labels, temp)
	if after.NLL >= before.NLL || after.Accuracy != before.Accuracy {
		t.Fatal(before, after)
	}
}
func TestDecisionMetricsRejectsMalformedAndGroupsTies(t *testing.T) {
	for _, row := range [][]float32{{1}, {float32(math.NaN()), 0}, {float32(math.Inf(1)), 0}} {
		if _, err := DecisionMetricsAtTemperature([][]float32{row}, []int{0}, 1); err == nil {
			t.Fatal("accepted", row)
		}
	}
	m, err := DecisionMetricsAtTemperature([][]float32{{0, 0}, {0, 0}}, []int{0, 1}, 1)
	if err != nil || len(m.Coverage) != 1 || m.Coverage[0].Count != 2 {
		t.Fatal(m, err)
	}
	if m.Coverage[0].ErrorLower95 >= 0.5 || m.Coverage[0].ErrorUpper95 <= 0.5 {
		t.Fatal("interval", m)
	}
}
