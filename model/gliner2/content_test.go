package gliner2

import (
	"math"
	"testing"
)

func TestSpanContentPoolerBuildPrefixStableLargeExponentAndMasking(t *testing.T) {
	pooler := testSpanContentPooler(true)
	textStates := [][]float32{{80, 1}, {100, -3}, {88, 6}}
	textMask := []bool{true, false, true}

	meanPrefix, lsePrefix, err := pooler.BuildPrefix(textStates, textMask)
	if err != nil {
		t.Fatal(err)
	}
	wantMean, wantLSE := oracleSpanContentBuildPrefix(pooler, textStates, textMask)
	assertCloseMatrix(t, "mean_prefix", meanPrefix, wantMean, 1e-5)
	assertCloseMatrix(t, "lse_prefix", lsePrefix, wantLSE, 1e-5)

	assertCloseVec(t, "masked mean row", meanPrefix[2], meanPrefix[1], 0)
	assertCloseVec(t, "masked lse row", lsePrefix[2], lsePrefix[1], 0)
}

func TestSpanContentPoolerPoolRowsSoftMaxClampsAndLayerNorm(t *testing.T) {
	pooler := testSpanContentPooler(true)
	textStates := [][]float32{{80, 1}, {100, -3}, {88, 6}, {4, 2}}
	textMask := []bool{true, false, true, true}
	meanPrefix, lsePrefix, err := pooler.BuildPrefix(textStates, textMask)
	if err != nil {
		t.Fatal(err)
	}
	starts := []int{-5, 0, 1, 2, 3}
	ends := []int{1, 2, 4, 4, 99}

	got, err := pooler.PoolRows(meanPrefix, lsePrefix, starts, ends)
	if err != nil {
		t.Fatal(err)
	}
	want := oracleSpanContentPoolRows(pooler, meanPrefix, lsePrefix, starts, ends)
	assertCloseMatrix(t, "pooled rows", got, want, 2e-5)

	if got[0][0] == got[1][0] {
		t.Fatal("raw length clamp was ignored for clamped negative start")
	}
}

func TestSpanContentPoolerPoolSoftMaxBatched(t *testing.T) {
	pooler := testSpanContentPooler(true)
	textStates := [][]float32{{9, 2}, {7, -1}, {3, 4}}
	textMask := []bool{true, true, true}
	meanPrefix, lsePrefix, err := pooler.BuildPrefix(textStates, textMask)
	if err != nil {
		t.Fatal(err)
	}
	starts := [][]int{{0, 1}, {-4, 2, 1}}
	ends := [][]int{{1, 3}, {1, 3, 99}}

	got, err := pooler.Pool(meanPrefix, lsePrefix, starts, ends)
	if err != nil {
		t.Fatal(err)
	}
	want := oracleSpanContentPool(pooler, meanPrefix, lsePrefix, starts, ends)
	if !rowsClose(flatten3(got), flatten3(want), 2e-5) {
		t.Fatalf("batched pool mismatch got=%v want=%v", got, want)
	}
}

func TestSpanContentPoolerEmptyAndInvalidShapes(t *testing.T) {
	pooler := testSpanContentPooler(true)
	meanPrefix, lsePrefix, err := pooler.BuildPrefix(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(meanPrefix) != 1 || len(lsePrefix) != 1 {
		t.Fatalf("empty prefix rows mean=%d lse=%d", len(meanPrefix), len(lsePrefix))
	}
	assertCloseVec(t, "empty mean row", meanPrefix[0], []float32{0, 0}, 0)
	assertCloseVec(t, "empty lse row", lsePrefix[0], []float32{spanContentMaskFloor, spanContentMaskFloor}, 0)
	pooled, err := pooler.PoolRows(meanPrefix, lsePrefix, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(pooled) != 0 {
		t.Fatalf("empty pooled rows=%d want=0", len(pooled))
	}

	if _, _, err := pooler.BuildPrefix([][]float32{{1}}, []bool{true}); err == nil {
		t.Fatal("accepted short text state row")
	}
	if _, _, err := pooler.BuildPrefix([][]float32{{1, 2}}, nil); err == nil {
		t.Fatal("accepted mismatched text mask")
	}
	if _, err := pooler.PoolRows(meanPrefix, nil, []int{0}, []int{1}); err == nil {
		t.Fatal("accepted missing lse prefix for softmax pooler")
	}
	if _, err := pooler.PoolRows(meanPrefix, lsePrefix, []int{0}, nil); err == nil {
		t.Fatal("accepted mismatched pooled row shapes")
	}
	if _, err := pooler.Pool(meanPrefix, lsePrefix, [][]int{{0}, {1}}, [][]int{{1}, nil}); err == nil {
		t.Fatal("accepted mismatched batched candidate shapes")
	}

	meanOnly := testSpanContentPooler(false)
	meanPrefixOnly, _, err := meanOnly.BuildPrefix([][]float32{{1, 2}}, []bool{true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := meanOnly.PoolRows(meanPrefixOnly, [][]float32{{0, 0}}, []int{0}, []int{1}); err == nil {
		t.Fatal("accepted lse prefix when softmax pooling disabled")
	}
}

func TestLoadSpanContentPoolerExactPrefix(t *testing.T) {
	shapes := shapeSource{
		"caller.provided.value_projection.weight": {2, 3},
		"caller.provided.value_projection.bias":   {2},
		"caller.provided.layer_norm.weight":       {4},
		"caller.provided.layer_norm.bias":         {4},
	}
	pooler, err := LoadSpanContentPooler(shapes, 3, 2, true, "caller.provided")
	if err != nil {
		t.Fatal(err)
	}
	if pooler.HiddenSize != 3 || pooler.ContentDim != 2 || !pooler.UseSoftMaxPool || pooler.LayerNorm.Dim() != 4 {
		t.Fatalf("binding mismatch: %+v", pooler)
	}
	delete(shapes, "caller.provided.value_projection.weight")
	if _, err := LoadSpanContentPooler(shapes, 3, 2, true, "caller.provided"); err == nil {
		t.Fatal("missing content tensor accepted")
	}
}

func testSpanContentPooler(useSoftMax bool) SpanContentPooler {
	outputDim := 2
	lnWeight := []float32{1.25, 0.75}
	lnBias := []float32{0.1, -0.2}
	if useSoftMax {
		outputDim = 4
		lnWeight = []float32{1.25, 0.75, 0.5, 1.5}
		lnBias = []float32{0.1, -0.2, 0.3, -0.4}
	}
	return SpanContentPooler{
		HiddenSize:     2,
		ContentDim:     2,
		UseSoftMaxPool: useSoftMax,
		ValueProjection: Linear{
			InDim:  2,
			OutDim: 2,
			Weight: []float32{1, 0, 0, 1},
		},
		LayerNorm: LayerNorm{Weight: lnWeight[:outputDim], Bias: lnBias[:outputDim], Epsilon: 1e-5},
	}
}

func oracleSpanContentBuildPrefix(pooler SpanContentPooler, textStates [][]float32, textMask []bool) ([][]float32, [][]float32) {
	values := manualLinearRows(textStates, pooler.ValueProjection)
	meanPrefix := makeMatrix(len(textStates)+1, pooler.ContentDim)
	var lsePrefix [][]float32
	if pooler.UseSoftMaxPool {
		lsePrefix = makeMatrix(len(textStates)+1, pooler.ContentDim)
		for i := range lsePrefix[0] {
			lsePrefix[0][i] = spanContentMaskFloor
		}
	}
	for row := range textStates {
		copy(meanPrefix[row+1], meanPrefix[row])
		if textMask[row] {
			for i := 0; i < pooler.ContentDim; i++ {
				meanPrefix[row+1][i] += values[row][i]
			}
		}
		if lsePrefix == nil {
			continue
		}
		for i := 0; i < pooler.ContentDim; i++ {
			value := spanContentMaskFloor
			if textMask[row] {
				value = values[row][i]
			}
			lsePrefix[row+1][i] = oracleLogAddExp32(lsePrefix[row][i], value)
		}
	}
	return meanPrefix, lsePrefix
}

func oracleSpanContentPoolRows(pooler SpanContentPooler, meanPrefix, lsePrefix [][]float32, starts, ends []int) [][]float32 {
	rows := make([][]float32, len(starts))
	for i := range starts {
		row := make([]float32, pooler.OutputDim())
		start := clampIndex(starts[i], len(meanPrefix))
		end := clampIndex(ends[i], len(meanPrefix))
		length := float32(max(ends[i]-starts[i], 1))
		for j := 0; j < pooler.ContentDim; j++ {
			row[j] = (meanPrefix[end][j] - meanPrefix[start][j]) / length
		}
		if pooler.UseSoftMaxPool {
			for j := 0; j < pooler.ContentDim; j++ {
				delta := lsePrefix[start][j] - lsePrefix[end][j]
				if delta > -1e-6 {
					delta = -1e-6
				}
				v := lsePrefix[end][j] + float32(math.Log1p(-math.Exp(float64(delta))))
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					v = 0
				}
				row[pooler.ContentDim+j] = v
			}
		}
		rows[i] = row
	}
	return manualLayerNormRows(rows, pooler.LayerNorm)
}

func oracleSpanContentPool(pooler SpanContentPooler, meanPrefix, lsePrefix [][]float32, starts, ends [][]int) [][][]float32 {
	flatStarts := make([]int, 0)
	flatEnds := make([]int, 0)
	for q := range starts {
		flatStarts = append(flatStarts, starts[q]...)
		flatEnds = append(flatEnds, ends[q]...)
	}
	flat := oracleSpanContentPoolRows(pooler, meanPrefix, lsePrefix, flatStarts, flatEnds)
	out := make([][][]float32, len(starts))
	pos := 0
	for q := range starts {
		out[q] = make([][]float32, len(starts[q]))
		for c := range starts[q] {
			out[q][c] = append([]float32(nil), flat[pos]...)
			pos++
		}
	}
	return out
}

func oracleLogAddExp32(a, b float32) float32 {
	if a < b {
		a, b = b, a
	}
	return a + float32(math.Log1p(math.Exp(float64(b-a))))
}

func flatten3(rows [][][]float32) [][]float32 {
	out := make([][]float32, 0)
	for i := range rows {
		out = append(out, rows[i]...)
	}
	return out
}
