package jevlike

import (
	"fmt"
	"math"
	"testing"
)

func TestAttentionHeadBackwardFiniteDifference(t *testing.T) {
	h, err := NewAttentionHead(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	copy(h.ContextNormWeight, []float32{1.1, -0.9, 0.7})
	copy(h.ContextNormBias, []float32{0.05, -0.03, 0.02})
	copy(h.OptionNormWeight, []float32{0.8, 1.2, -1.1})
	copy(h.OptionNormBias, []float32{0.04, -0.02, 0.01})
	copy(h.QueryWeight, []float32{0.3, -0.2, 0.5, -0.4, 0.6, 0.1})
	copy(h.KeyWeight, []float32{-0.5, 0.7, 0.2, 0.3, -0.1, 0.4})
	copy(h.ValueWeight, []float32{0.6, -0.3, 0.2, -0.2, 0.5, -0.4})

	context := [][][]float32{
		{{0.2, -0.4, 0.7}, {-0.1, 0.5, 0.3}, {0.4, 0.1, -0.6}},
		{{0.6, -0.7, 0.2}, {-0.3, 0.4, -0.5}, {0.8, 0.1, 0.3}},
	}
	contextMask := [][]bool{{true, false, true}, {false, false, false}}
	options := [][][]float32{
		{{0.3, -0.2, 0.5}, {-0.6, 0.2, 0.1}},
		{{0.4, 0.7, -0.3}, {0.2, -0.5, 0.6}},
	}
	optionMask := [][]bool{{true, false}, {true, true}}
	dLogits := [][]float32{{0.7, -0.9}, {-0.4, 0.6}}

	paramGrads, dContext, dOptions, err := h.Backward(context, contextMask, options, optionMask, dLogits)
	if err != nil {
		t.Fatal(err)
	}

	expectedKeys := []string{
		defaultHeadPrefix + ".context_norm.weight",
		defaultHeadPrefix + ".context_norm.bias",
		defaultHeadPrefix + ".option_norm.weight",
		defaultHeadPrefix + ".option_norm.bias",
		defaultHeadPrefix + ".query.weight",
		defaultHeadPrefix + ".key.weight",
		defaultHeadPrefix + ".value.weight",
	}
	for _, key := range expectedKeys {
		if _, ok := paramGrads[key]; !ok {
			t.Fatalf("missing parameter gradient %q", key)
		}
	}

	eval := func() (float64, error) {
		return attentionObjective(h, context, contextMask, options, optionMask, dLogits)
	}
	step := float32(1e-3)

	for _, key := range expectedKeys {
		slice := parameterSlice(h, key)
		grads := paramGrads[key]
		if len(slice) != len(grads) {
			t.Fatalf("parameter gradient %q len=%d want=%d", key, len(grads), len(slice))
		}
		for i := range slice {
			numeric := centralDifference(t, &slice[i], step, eval)
			requireGradientClose(t, key+indexLabel(i), grads[i], numeric)
		}
		requireAnyNonZero(t, key, grads)
	}

	for row := range context {
		for token := range context[row] {
			for dim := range context[row][token] {
				numeric := centralDifference(t, &context[row][token][dim], step, eval)
				requireGradientClose(t, inputLabel("context", row, token, dim), dContext[row][token][dim], numeric)
			}
		}
	}
	require3DAnyNonZero(t, "context input", dContext)

	for row := range options {
		for option := range options[row] {
			for dim := range options[row][option] {
				numeric := centralDifference(t, &options[row][option][dim], step, eval)
				requireGradientClose(t, inputLabel("option", row, option, dim), dOptions[row][option][dim], numeric)
			}
		}
	}
	require3DAnyNonZero(t, "option input", dOptions)

	for dim, value := range dOptions[0][1] {
		if value != 0 {
			t.Fatalf("masked option gradient dim=%d got=%g want 0", dim, value)
		}
	}
	for dim, value := range dContext[1][0] {
		if math.Abs(float64(value)) < 1e-5 {
			t.Fatalf("all-masked context row should still backprop through values dim=%d got=%g", dim, value)
		}
	}
}

func TestAttentionHeadBackwardRejectsGradientShape(t *testing.T) {
	h, err := NewAttentionHead(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	context := [][][]float32{{{1, 2}}}
	contextMask := [][]bool{{true}}
	options := [][][]float32{{{3, 4}}}
	optionMask := [][]bool{{true}}
	if _, _, _, err := h.Backward(context, contextMask, options, optionMask, [][]float32{}); err == nil {
		t.Fatal("expected row mismatch error")
	}
	if _, _, _, err := h.Backward(context, contextMask, options, optionMask, [][]float32{{1, 2}}); err == nil {
		t.Fatal("expected option mismatch error")
	}
}

func attentionObjective(h *AttentionHead, context [][][]float32, contextMask [][]bool, options [][][]float32, optionMask [][]bool, dLogits [][]float32) (float64, error) {
	logits, err := h.Forward(context, contextMask, options, optionMask)
	if err != nil {
		return 0, err
	}
	var total float64
	for row := range logits {
		for option := range logits[row] {
			if !optionMask[row][option] {
				continue
			}
			total += float64(logits[row][option]) * float64(dLogits[row][option])
		}
	}
	return total, nil
}

func parameterSlice(h *AttentionHead, key string) []float32 {
	switch key {
	case defaultHeadPrefix + ".context_norm.weight":
		return h.ContextNormWeight
	case defaultHeadPrefix + ".context_norm.bias":
		return h.ContextNormBias
	case defaultHeadPrefix + ".option_norm.weight":
		return h.OptionNormWeight
	case defaultHeadPrefix + ".option_norm.bias":
		return h.OptionNormBias
	case defaultHeadPrefix + ".query.weight":
		return h.QueryWeight
	case defaultHeadPrefix + ".key.weight":
		return h.KeyWeight
	case defaultHeadPrefix + ".value.weight":
		return h.ValueWeight
	default:
		panic("unknown parameter key: " + key)
	}
}

func centralDifference(t *testing.T, value *float32, step float32, eval func() (float64, error)) float32 {
	t.Helper()
	original := *value
	*value = original + step
	plus, err := eval()
	if err != nil {
		t.Fatal(err)
	}
	*value = original - step
	minus, err := eval()
	if err != nil {
		t.Fatal(err)
	}
	*value = original
	return float32((plus - minus) / (2 * float64(step)))
}

func requireGradientClose(t *testing.T, label string, got, want float32) {
	t.Helper()
	diff := math.Abs(float64(got - want))
	scale := math.Max(1, math.Max(math.Abs(float64(got)), math.Abs(float64(want))))
	if diff > 7e-3 && diff > 4e-2*scale {
		t.Fatalf("%s gradient=%g numeric=%g diff=%g", label, got, want, diff)
	}
}

func requireAnyNonZero(t *testing.T, label string, values []float32) {
	t.Helper()
	for _, value := range values {
		if math.Abs(float64(value)) > 1e-5 {
			return
		}
	}
	t.Fatalf("%s gradients are all zero", label)
}

func require3DAnyNonZero(t *testing.T, label string, values [][][]float32) {
	t.Helper()
	for i := range values {
		for j := range values[i] {
			for _, value := range values[i][j] {
				if math.Abs(float64(value)) > 1e-5 {
					return
				}
			}
		}
	}
	t.Fatalf("%s gradients are all zero", label)
}

func indexLabel(i int) string {
	return fmt.Sprintf("[%d]", i)
}

func inputLabel(kind string, a, b, c int) string {
	return kind + indexLabel(a) + indexLabel(b) + indexLabel(c)
}
