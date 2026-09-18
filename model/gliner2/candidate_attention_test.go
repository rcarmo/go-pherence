package gliner2

import (
	"math"
	"testing"
)

func TestClassifyOverlapBucketsExhaustivePairs(t *testing.T) {
	var spans [][2]int
	for start := 0; start < 4; start++ {
		for end := start + 1; end <= 4; end++ {
			spans = append(spans, [2]int{start, end})
		}
	}
	got := ClassifyOverlapBuckets(spans)
	counts := make([]int, NumOverlapBuckets)
	for i, left := range spans {
		if len(got[i]) != len(spans) {
			t.Fatalf("row %d len=%d want=%d", i, len(got[i]), len(spans))
		}
		for j, right := range spans {
			want := oracleClassifyOverlapBucket(left, right)
			if single := ClassifyOverlapBucket(left, right); single != want {
				t.Fatalf("single bucket %v vs %v got=%d want=%d", left, right, single, want)
			}
			if got[i][j] != want {
				t.Fatalf("bucket[%d][%d] %v vs %v got=%d want=%d", i, j, left, right, got[i][j], want)
			}
			counts[want]++
		}
	}
	for bucket, count := range counts {
		if count == 0 {
			t.Fatalf("bucket %d never observed", bucket)
		}
	}
}

func TestOverlapBiasedCandidateAttentionOracleNoBias(t *testing.T) {
	block := OverlapBiasedCandidateAttention{
		NumHeads: 2,
		Norm1:    LayerNorm{Weight: []float32{1, 1, 1, 1}, Bias: []float32{0, 0, 0, 0}, Epsilon: 1e-5},
		QKVProjection: Linear{
			InDim:  4,
			OutDim: 12,
			Weight: []float32{
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1,
				0.5, 0.1, 0, 0,
				0, 1.2, 0.2, 0,
				0.1, 0, 0.9, 0.3,
				0, 0.2, 0, 0.7,
				1, 0, 0, 0,
				0, 0.5, 0, 0,
				0, 0, 1.5, 0,
				0, 0, 0, 0.75,
			},
		},
		OutputProjection: Linear{
			InDim:  4,
			OutDim: 4,
			Weight: []float32{
				1, 0.1, 0, 0,
				0, 1, 0.2, 0,
				0, 0, 1, 0.3,
				0.4, 0, 0, 1,
			},
		},
		RelativeBias: make([]float32, NumOverlapBuckets*2),
		Norm2:        LayerNorm{Weight: []float32{1, 1, 1, 1}, Bias: []float32{0, 0, 0, 0}, Epsilon: 1e-5},
		FFNInputProjection: Linear{
			InDim:  4,
			OutDim: 16,
			Weight: make([]float32, 64),
			Bias:   make([]float32, 16),
		},
		FFNOutputProjection: Linear{
			InDim:  16,
			OutDim: 4,
			Weight: make([]float32, 64),
			Bias:   make([]float32, 4),
		},
	}
	states := [][]float32{{2, 0, -1, 1}, {-1, 3, 2, 0}, {0.5, -0.5, 1.5, -2}}
	indices := [][2]int{{0, 2}, {0, 3}, {1, 3}}
	mask := []bool{true, true, true}

	got, err := block.Forward(states, indices, mask)
	if err != nil {
		t.Fatal(err)
	}
	want := oracleOverlapBiasedCandidateAttention(block, states, indices, mask)
	assertCloseMatrix(t, "candidate attention", got, want, 3e-5)
}

func TestOverlapBiasedCandidateAttentionMaskingAndAllMasked(t *testing.T) {
	block := OverlapBiasedCandidateAttention{
		NumHeads: 1,
		Norm1:    LayerNorm{Weight: []float32{1, 1}, Bias: []float32{0, 0}, Epsilon: 1e-5},
		QKVProjection: Linear{
			InDim:  2,
			OutDim: 6,
			Weight: []float32{
				1, 0,
				0, 1,
				1, 0,
				0, 1,
				1, 0,
				0, 1,
			},
		},
		OutputProjection: Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}},
		RelativeBias:     []float32{2, -1, -1, 0, -0.5, -0.5, -2, -2},
		Norm2:            LayerNorm{Weight: []float32{1, 1}, Bias: []float32{0, 0}, Epsilon: 1e-5},
		FFNInputProjection: Linear{
			InDim:  2,
			OutDim: 8,
			Weight: make([]float32, 16),
			Bias:   make([]float32, 8),
		},
		FFNOutputProjection: Linear{
			InDim:  8,
			OutDim: 2,
			Weight: make([]float32, 16),
			Bias:   make([]float32, 2),
		},
	}
	states := [][]float32{{1, 2}, {-3, 4}, {5, -1}}
	indices := [][2]int{{0, 1}, {1, 3}, {3, 4}}
	mask := []bool{true, false, true}

	got, err := block.Forward(states, indices, mask)
	if err != nil {
		t.Fatal(err)
	}
	want := oracleOverlapBiasedCandidateAttention(block, states, indices, mask)
	assertCloseMatrix(t, "candidate attention masked", got, want, 3e-5)
	if got[1][0] != 0 || got[1][1] != 0 {
		t.Fatalf("masked row leaked: %v", got[1])
	}

	allMasked := []bool{false, false, false}
	gotAll, err := block.Forward(states, indices, allMasked)
	if err != nil {
		t.Fatal(err)
	}
	wantAll := oracleOverlapBiasedCandidateAttention(block, states, indices, allMasked)
	assertCloseMatrix(t, "candidate attention all masked", gotAll, wantAll, 3e-5)
	for row := range gotAll {
		for col := range gotAll[row] {
			if gotAll[row][col] != 0 {
				t.Fatalf("all-masked row %d col %d = %g want 0", row, col, gotAll[row][col])
			}
		}
	}
}

func TestLoadOverlapBiasedCandidateAttentionExactPrefix(t *testing.T) {
	shapes := shapeSource{
		"caller.provided.norm1.weight":  {4},
		"caller.provided.norm1.bias":    {4},
		"caller.provided.qkv.weight":    {12, 4},
		"caller.provided.qkv.bias":      {12},
		"caller.provided.output.weight": {4, 4},
		"caller.provided.output.bias":   {4},
		"caller.provided.relative_bias": {NumOverlapBuckets, 2},
		"caller.provided.norm2.weight":  {4},
		"caller.provided.norm2.bias":    {4},
		"caller.provided.ffn.0.weight":  {16, 4},
		"caller.provided.ffn.0.bias":    {16},
		"caller.provided.ffn.3.weight":  {4, 16},
		"caller.provided.ffn.3.bias":    {4},
	}
	block, err := LoadOverlapBiasedCandidateAttention(shapes, 4, 2, "caller.provided")
	if err != nil {
		t.Fatal(err)
	}
	if block.NumHeads != 2 || block.Norm1.Dim() != 4 || len(block.RelativeBias) != NumOverlapBuckets*2 {
		t.Fatalf("binding mismatch: %+v", block)
	}
	if _, err := LoadOverlapBiasedCandidateAttention(shapes, 4, 2, ""); err == nil {
		t.Fatal("empty prefix accepted")
	}

	missing := shapeSource{}
	for k, v := range shapes {
		missing[k] = append([]int(nil), v...)
	}
	delete(missing, "caller.provided.qkv.weight")
	if _, err := LoadOverlapBiasedCandidateAttention(missing, 4, 2, "caller.provided"); err == nil {
		t.Fatal("missing tensor accepted")
	}

	wrong := shapeSource{}
	for k, v := range shapes {
		wrong[k] = append([]int(nil), v...)
	}
	wrong["caller.provided.relative_bias"] = []int{NumOverlapBuckets, 1}
	if _, err := LoadOverlapBiasedCandidateAttention(wrong, 4, 2, "caller.provided"); err == nil {
		t.Fatal("wrong relative_bias shape accepted")
	}
}

func oracleClassifyOverlapBucket(left, right [2]int) int {
	s1, e1 := left[0], left[1]
	s2, e2 := right[0], right[1]
	switch {
	case s1 == s2 && e1 == e2:
		return OverlapIdentical
	case e1 <= s2:
		return OverlapDisjointLeft
	case e2 <= s1:
		return OverlapDisjointRight
	case s1 == s2:
		return OverlapSameStart
	case e1 == e2:
		return OverlapSameEnd
	case s1 > s2 && e1 < e2:
		return OverlapNestedInside
	case s1 < s2 && e1 > e2:
		return OverlapNestedOutside
	default:
		return OverlapCrossing
	}
}

func oracleOverlapBiasedCandidateAttention(block OverlapBiasedCandidateAttention, states [][]float32, indices [][2]int, mask []bool) [][]float32 {
	normed1 := manualLayerNormRows(states, block.Norm1)
	qkv := manualLinearRows(normed1, block.QKVProjection)
	rows := len(states)
	dim := block.Norm1.Dim()
	headDim := dim / block.NumHeads
	scale := 1 / math.Sqrt(float64(headDim))
	context := make([][]float32, rows)
	for query := 0; query < rows; query++ {
		context[query] = make([]float32, dim)
		for head := 0; head < block.NumHeads; head++ {
			scores := make([]float64, rows)
			maxScore := math.Inf(-1)
			for key := 0; key < rows; key++ {
				if !mask[key] && key != query {
					scores[key] = math.Inf(-1)
					continue
				}
				dot := 0.0
				for i := 0; i < headDim; i++ {
					dot += float64(qkv[query][head*headDim+i]) * float64(qkv[key][dim+head*headDim+i])
				}
				scores[key] = dot*scale + float64(block.RelativeBias[oracleClassifyOverlapBucket(indices[query], indices[key])*block.NumHeads+head])
				if scores[key] > maxScore {
					maxScore = scores[key]
				}
			}
			sum := 0.0
			probs := make([]float64, rows)
			for key := 0; key < rows; key++ {
				if math.IsInf(scores[key], -1) {
					continue
				}
				probs[key] = math.Exp(scores[key] - maxScore)
				sum += probs[key]
			}
			for key := 0; key < rows; key++ {
				if probs[key] == 0 {
					continue
				}
				weight := float32(probs[key] / sum)
				for i := 0; i < headDim; i++ {
					context[query][head*headDim+i] += weight * qkv[key][2*dim+head*headDim+i]
				}
			}
		}
	}
	attnUpdate := manualLinearRows(context, block.OutputProjection)
	attended := cloneMatrix(states)
	for row := range attended {
		for i := range attended[row] {
			attended[row][i] += attnUpdate[row][i]
		}
	}
	normed2 := manualLayerNormRows(attended, block.Norm2)
	ffnHidden := manualLinearRows(normed2, block.FFNInputProjection)
	for row := range ffnHidden {
		for i := range ffnHidden[row] {
			ffnHidden[row][i] = oracleGELU(ffnHidden[row][i])
		}
	}
	ffnUpdate := manualLinearRows(ffnHidden, block.FFNOutputProjection)
	out := cloneMatrix(attended)
	for row := range out {
		for i := range out[row] {
			out[row][i] += ffnUpdate[row][i]
		}
		if mask[row] {
			continue
		}
		for i := range out[row] {
			out[row][i] = 0
		}
	}
	return out
}

func oracleGELU(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
}
