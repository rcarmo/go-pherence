package jevlike

import (
	"math"
	"reflect"
	"testing"
)

func TestObservationRoundingClippingAndTensor(t *testing.T) {
	frame := RGBFrame{Width: 2, Height: 1, Pixels: []uint8{10, 20, 30, 250, 251, 252}}
	previous := RGBFrame{Width: 2, Height: 1, Pixels: []uint8{0, 10, 20, 255, 255, 255}}
	got, err := Observation(frame, &previous)
	if err != nil {
		t.Fatal(err)
	}
	wantPixels := []uint8{10, 20, 30, 138, 250, 251, 252, 124}
	if !reflect.DeepEqual(got.Pixels, wantPixels) {
		t.Fatalf("observation pixels=%v want %v", got.Pixels, wantPixels)
	}
	first, err := Observation(frame, nil)
	if err != nil {
		t.Fatal(err)
	}
	if first.Pixels[3] != 128 || first.Pixels[7] != 128 {
		t.Fatalf("nil previous diff bytes=%v want 128s", []uint8{first.Pixels[3], first.Pixels[7]})
	}
	if got := encodeSignedDifference(0.5); got != 128 {
		t.Fatalf("round-to-even 0.5=%d want 128", got)
	}
	if got := encodeSignedDifference(1.5); got != 130 {
		t.Fatalf("round-to-even 1.5=%d want 130", got)
	}
	if got := encodeSignedDifference(2.5); got != 130 {
		t.Fatalf("round-to-even 2.5=%d want 130", got)
	}
	if got := encodeSignedDifference(-1.5); got != 126 {
		t.Fatalf("round-to-even -1.5=%d want 126", got)
	}
	if got := encodeSignedDifference(200); got != 255 {
		t.Fatalf("positive clip=%d want 255", got)
	}
	if got := encodeSignedDifference(-200); got != 0 {
		t.Fatalf("negative clip=%d want 0", got)
	}
	tensor, err := ObservationTensor([]PackedObservation{got})
	if err != nil {
		t.Fatal(err)
	}
	if tensor.Shape != [4]int{1, 4, 1, 2} {
		t.Fatalf("tensor shape=%v want [1 4 1 2]", tensor.Shape)
	}
	wantTensor := []float32{
		10.0 / 255.0, 250.0 / 255.0,
		20.0 / 255.0, 251.0 / 255.0,
		30.0 / 255.0, 252.0 / 255.0,
		138.0 / 255.0, 124.0 / 255.0,
	}
	assertClose1D(t, "tensor", tensor.Values, wantTensor, 1e-7)
}

func TestPosition2DReference(t *testing.T) {
	got, err := Position2D(2, 3, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2*3*8 {
		t.Fatalf("position len=%d want %d", len(got), 2*3*8)
	}
	firstWant := []float32{0, 0, 1, 1, 0, 0, 1, 1}
	assertClose1D(t, "position[0,0]", got[:8], firstWant, 1e-7)
	freq1 := math.Exp(-math.Log(10_000))
	index := (1*3 + 2) * 8
	want := []float32{
		float32(math.Sin(1)),
		float32(math.Sin(1 * freq1)),
		float32(math.Cos(1)),
		float32(math.Cos(1 * freq1)),
		float32(math.Sin(2)),
		float32(math.Sin(2 * freq1)),
		float32(math.Cos(2)),
		float32(math.Cos(2 * freq1)),
	}
	assertClose1D(t, "position[1,2]", got[index:index+8], want, 1e-7)
}

func TestVisionActionScorerOptionSelectionMultiReadTrace(t *testing.T) {
	cfg := VisionActionConfig{Width: 4, Rank: 4, Actions: 4, Reads: 2, PatchRows: 1, PatchColumns: 2}
	scorer, err := NewVisionActionScorer(cfg)
	if err != nil {
		t.Fatal(err)
	}
	scorer.Positions = []float32{
		0.1, 0.0, -0.1, 0.2,
		-0.2, 0.3, 0.1, 0.0,
	}
	copy(scorer.OptionEmbedding, []float32{
		1, 0, -1, 0.5,
		0.2, 1.2, -0.7, 0.3,
		-0.5, 0.5, 1.5, -1,
		1, 1, 0, 0,
	})
	setDiagonalProjection(&scorer.Heads[0], []float32{1, 1, 1, 1}, []float32{1, 1, 1, 1}, []float32{1, 1, 1, 1})
	setDiagonalProjection(&scorer.Heads[1], []float32{0.5, 1.5, -1, 1}, []float32{0, 0, 0, 0}, []float32{1.2, 0.8, 1, -0.5})
	copy(scorer.ValueNormWeight, []float32{1.2, -0.8, 0.6, 1.1})
	copy(scorer.ValueNormBias, []float32{0.05, -0.02, 0.03, -0.04})
	copy(scorer.ValueWeight, []float32{0.2, -0.1, 0.3, 0.4})
	scorer.ValueBias = -0.15

	features := [][][]float32{
		{{1, 2, 0, -1}, {0.5, -1.5, 2, 1}},
		{{-0.5, 1, 1.5, 0}, {2, -1, 0.5, -0.5}},
	}
	selection := OptionSelection{PerBatch: [][]int{{2, 0}, {1, 3}}}

	wantLogits, wantValue, wantTrace := referenceVisionForward(scorer, features, selection)
	gotLogits, gotValue, gotTrace, err := scorer.ForwardTrace(features, selection)
	if err != nil {
		t.Fatal(err)
	}
	assertClose2D(t, "logits", gotLogits, wantLogits, 1e-5)
	assertClose1D(t, "value", gotValue, wantValue, 1e-5)
	assertClose3D(t, "trace.query", gotTrace.QueryMatrix, wantTrace.QueryMatrix, 1e-5)
	assertClose3D(t, "trace.key", gotTrace.KeyMatrix, wantTrace.KeyMatrix, 1e-5)
	assertClose3D(t, "trace.value", gotTrace.ValueMatrix, wantTrace.ValueMatrix, 1e-5)
	assertClose3D(t, "trace.attention", gotTrace.AttentionMap, wantTrace.AttentionMap, 1e-5)
	assertClose2D(t, "trace.logits", gotTrace.LogitsMatrix, wantTrace.LogitsMatrix, 1e-5)
	assertClose2D(t, "trace.probabilities", gotTrace.Probabilities, wantTrace.Probabilities, 1e-5)
	assertClose2D(t, "trace.entropy", gotTrace.Entropy, wantTrace.Entropy, 1e-5)

	plainLogits, plainValue, err := scorer.Forward(features, selection)
	if err != nil {
		t.Fatal(err)
	}
	assertClose2D(t, "forward logits", plainLogits, gotLogits, 1e-6)
	assertClose1D(t, "forward value", plainValue, gotValue, 1e-6)

	sharedLogits, sharedValue, err := scorer.Forward(features, OptionSelection{Shared: []int{2, 0}})
	if err != nil {
		t.Fatal(err)
	}
	repeatedLogits, repeatedValue, err := scorer.Forward(features, OptionSelection{PerBatch: [][]int{{2, 0}, {2, 0}}})
	if err != nil {
		t.Fatal(err)
	}
	assertClose2D(t, "shared logits", sharedLogits, repeatedLogits, 1e-6)
	assertClose1D(t, "shared value", sharedValue, repeatedValue, 1e-6)

	if _, _, _, err := scorer.ForwardTrace([][][]float32{{{1, 2, 3, 4}}}, OptionSelection{}); err == nil {
		t.Fatal("expected token-count validation error")
	}
	if _, _, _, err := scorer.ForwardTrace(features, OptionSelection{Shared: []int{4}}); err == nil {
		t.Fatal("expected option id validation error")
	}
	if _, _, _, err := scorer.ForwardTrace(features, OptionSelection{Shared: []int{0}, PerBatch: [][]int{{0}}}); err == nil {
		t.Fatal("expected mixed option selection error")
	}
}

func setDiagonalProjection(head *AttentionHead, queryDiag, keyDiag, valueDiag []float32) {
	fillDiagonalMatrix(head.QueryWeight, head.Width, queryDiag)
	fillDiagonalMatrix(head.KeyWeight, head.Width, keyDiag)
	fillDiagonalMatrix(head.ValueWeight, head.Width, valueDiag)
}

func fillDiagonalMatrix(weight []float32, width int, diagonal []float32) {
	for i := range weight {
		weight[i] = 0
	}
	for i, value := range diagonal {
		weight[i*width+i] = value
	}
}

func referenceVisionForward(scorer *VisionActionScorer, features [][][]float32, selection OptionSelection) ([][]float32, []float32, VisionActionTrace) {
	optionIDs := selection.PerBatch
	rows := len(features)
	tokens := len(features[0])
	optionsPerRow := len(optionIDs[0])
	trace := VisionActionTrace{
		QueryMatrix:   make3DFloat32(rows, optionsPerRow, scorer.Config.Rank),
		KeyMatrix:     make3DFloat32(rows, tokens, scorer.Config.Rank),
		ValueMatrix:   make3DFloat32(rows, tokens, scorer.Config.Rank),
		AttentionMap:  make3DFloat32(rows, optionsPerRow, tokens),
		LogitsMatrix:  make2DFloat32(rows, optionsPerRow),
		Probabilities: make2DFloat32(rows, optionsPerRow),
		Entropy:       make2DFloat32(rows, optionsPerRow),
	}
	readScale := float32(1.0 / float64(len(scorer.Heads)))
	rankScale := float32(math.Sqrt(float64(scorer.Config.Rank)))
	for read := range scorer.Heads {
		head := &scorer.Heads[read]
		for row := 0; row < rows; row++ {
			keys := make([][]float32, tokens)
			values := make([][]float32, tokens)
			for token := 0; token < tokens; token++ {
				normalized := referenceLayerNorm(features[row][token], head.ContextNormWeight, head.ContextNormBias)
				key := referenceLinearNoBias(head.KeyWeight, head.Rank, head.Width, normalized)
				position := scorer.positionRow(token)
				for dim := 0; dim < head.Rank; dim++ {
					key[dim] += position[dim]
				}
				value := referenceLinearNoBias(head.ValueWeight, head.Rank, head.Width, normalized)
				keys[token] = key
				values[token] = value
				for dim := 0; dim < head.Rank; dim++ {
					trace.KeyMatrix[row][token][dim] += key[dim] * readScale
					trace.ValueMatrix[row][token][dim] += value[dim] * readScale
				}
			}
			for option := 0; option < optionsPerRow; option++ {
				embedding := scorer.optionRow(optionIDs[row][option])
				normalized := referenceLayerNorm(embedding, head.OptionNormWeight, head.OptionNormBias)
				query := referenceLinearNoBias(head.QueryWeight, head.Rank, head.Width, normalized)
				scores := make([]float32, tokens)
				for token := 0; token < tokens; token++ {
					scores[token] = referenceDot(query, keys[token]) / rankScale
				}
				attention := referenceSoftmax(scores)
				attended := make([]float32, head.Rank)
				for token, weight := range attention {
					for dim := 0; dim < head.Rank; dim++ {
						attended[dim] += weight * values[token][dim]
					}
				}
				trace.LogitsMatrix[row][option] += referenceDot(query, attended) / rankScale * readScale
				for dim := 0; dim < head.Rank; dim++ {
					trace.QueryMatrix[row][option][dim] += query[dim] * readScale
				}
				for token := 0; token < tokens; token++ {
					trace.AttentionMap[row][option][token] += attention[token] * readScale
				}
			}
		}
	}
	value := make([]float32, rows)
	for row := 0; row < rows; row++ {
		pooled := make([]float32, scorer.Config.Width)
		for token := 0; token < tokens; token++ {
			normalized := referenceLayerNorm(features[row][token], scorer.Heads[0].ContextNormWeight, scorer.Heads[0].ContextNormBias)
			for dim := 0; dim < scorer.Config.Width; dim++ {
				pooled[dim] += normalized[dim] / float32(tokens)
			}
		}
		pooled = referenceLayerNorm(pooled, scorer.ValueNormWeight, scorer.ValueNormBias)
		value[row] = referenceDot(pooled, scorer.ValueWeight) + scorer.ValueBias
		trace.Probabilities[row] = referenceSoftmax(trace.LogitsMatrix[row])
		for option := 0; option < optionsPerRow; option++ {
			trace.Entropy[row][option] = normalizedAttentionEntropy(trace.AttentionMap[row][option])
		}
	}
	return trace.LogitsMatrix, value, trace
}

func referenceLayerNorm(input, gamma, beta []float32) []float32 {
	out := make([]float32, len(input))
	var mean float64
	for _, value := range input {
		mean += float64(value)
	}
	mean /= float64(len(input))
	var variance float64
	for _, value := range input {
		delta := float64(value) - mean
		variance += delta * delta
	}
	variance /= float64(len(input))
	invStd := 1 / math.Sqrt(variance+float64(layerNormEpsilon))
	for i, value := range input {
		normalized := (float64(value) - mean) * invStd
		normalized *= float64(gamma[i])
		normalized += float64(beta[i])
		out[i] = float32(normalized)
	}
	return out
}

func referenceLinearNoBias(weight []float32, outDim, inDim int, input []float32) []float32 {
	out := make([]float32, outDim)
	for row := 0; row < outDim; row++ {
		base := row * inDim
		for col, value := range input {
			out[row] += weight[base+col] * value
		}
	}
	return out
}

func referenceDot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func referenceSoftmax(values []float32) []float32 {
	maxValue := values[0]
	for _, value := range values[1:] {
		if value > maxValue {
			maxValue = value
		}
	}
	out := make([]float32, len(values))
	var sum float64
	for i, value := range values {
		expValue := math.Exp(float64(value - maxValue))
		out[i] = float32(expValue)
		sum += expValue
	}
	for i := range out {
		out[i] /= float32(sum)
	}
	return out
}

func assertClose1D(t *testing.T, label string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d want %d", label, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("%s[%d]=%g want %g", label, i, got[i], want[i])
		}
	}
}

func assertClose2D(t *testing.T, label string, got, want [][]float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s rows=%d want %d", label, len(got), len(want))
	}
	for i := range got {
		assertClose1D(t, label+" row", got[i], want[i], tol)
	}
}

func assertClose3D(t *testing.T, label string, got, want [][][]float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s rows=%d want %d", label, len(got), len(want))
	}
	for i := range got {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("%s[%d] cols=%d want %d", label, i, len(got[i]), len(want[i]))
		}
		for j := range got[i] {
			assertClose1D(t, label+" row", got[i][j], want[i][j], tol)
		}
	}
}
