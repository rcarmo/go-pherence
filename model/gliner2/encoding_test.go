package gliner2

import (
	"math"
	"reflect"
	"testing"
)

func TestLinearForwardAndValidation(t *testing.T) {
	lin := Linear{
		InDim:  2,
		OutDim: 3,
		Weight: []float32{1, 2, 3, 4, 5, 6},
		Bias:   []float32{0.5, -0.5, 1},
	}
	one := make([]float32, 3)
	if err := lin.Apply([]float32{2, 3}, one); err != nil {
		t.Fatal(err)
	}
	wantOne := []float32{8.5, 17.5, 29}
	assertCloseVec(t, "linear single", one, wantOne, 1e-6)

	batchIn := []float32{2, 3, -1, 4}
	batchOut := make([]float32, 6)
	if err := lin.ApplyBatch(batchIn, batchOut, 2); err != nil {
		t.Fatal(err)
	}
	wantBatch := []float32{8.5, 17.5, 29, 7.5, 12.5, 20}
	assertCloseVec(t, "linear batch", batchOut, wantBatch, 1e-6)

	if err := (Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 2, 3}}).Validate(); err == nil {
		t.Fatal("accepted malformed linear weight")
	}
	if err := lin.Apply([]float32{1}, make([]float32, 3)); err == nil {
		t.Fatal("accepted short linear input")
	}
	if err := (LayerNorm{Weight: []float32{1, 1}, Bias: []float32{0}}).Validate(); err == nil {
		t.Fatal("accepted malformed layernorm bias")
	}
}

func TestResidualSwiGLUIdentity(t *testing.T) {
	block := ResidualSwiGLU{
		Norm: LayerNorm{Weight: []float32{1, 1}, Bias: []float32{0, 0}, Epsilon: 1e-5},
		InputProjection: Linear{
			InDim:  2,
			OutDim: 4,
			Weight: make([]float32, 8),
			Bias:   make([]float32, 4),
		},
		OutputProjection: Linear{
			InDim:  2,
			OutDim: 2,
			Weight: make([]float32, 4),
			Bias:   make([]float32, 2),
		},
	}
	states := [][]float32{{1, 2}, {-3, 4}, {0.5, -0.25}}
	got, err := block.Forward(states)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, states) {
		t.Fatalf("identity swiglu\ngot  %v\nwant %v", got, states)
	}
}

func TestBoundaryAttentionBlockLocalWindowAndIdentity(t *testing.T) {
	identity := BoundaryAttentionBlock{
		NumHeads: 1,
		Norm:     LayerNorm{Weight: []float32{1, 1}, Bias: []float32{0, 0}, Epsilon: 1e-5},
		QKVProjection: Linear{
			InDim:  2,
			OutDim: 6,
			Weight: make([]float32, 12),
			Bias:   make([]float32, 6),
		},
		OutputProjection: Linear{
			InDim:  2,
			OutDim: 2,
			Weight: make([]float32, 4),
			Bias:   make([]float32, 2),
		},
	}
	identityIn := [][]float32{{1, -1}, {2, 3}, {9, 9}}
	identityMask := []bool{true, true, false}
	identityOut, err := identity.Forward(identityIn, identityMask)
	if err != nil {
		t.Fatal(err)
	}
	wantIdentity := [][]float32{{1, -1}, {2, 3}, {0, 0}}
	assertCloseMatrix(t, "attention identity", identityOut, wantIdentity, 0)

	block := BoundaryAttentionBlock{
		NumHeads: 1,
		Window:   1,
		Norm:     LayerNorm{Weight: []float32{1, 1}, Bias: []float32{0, 0}, Epsilon: 1e-5},
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
	}
	states := [][]float32{{2, 0}, {0, 2}, {2, 1}, {4, 0}}
	mask := []bool{true, true, true, false}
	got, err := block.Forward(states, mask)
	if err != nil {
		t.Fatal(err)
	}
	want := manualAttentionBlock(block, states, mask)
	assertCloseMatrix(t, "attention local", got, want, 2e-5)
	if got[3][0] != 0 || got[3][1] != 0 {
		t.Fatalf("masked row leaked: %v", got[3])
	}

	global := block
	global.Window = 0
	globalWant := manualAttentionBlock(global, states, mask)
	if rowsClose(got, globalWant, 1e-6) {
		t.Fatal("local window had no effect")
	}
}

func TestBoundaryEncoderReferencePaddingAndEmptyDocument(t *testing.T) {
	enc := BoundaryEncoder{
		HiddenSize:  2,
		BoundaryDim: 2,
		LeftProjection: Linear{
			InDim:  2,
			OutDim: 2,
			Weight: []float32{1, 0, 0, 1},
		},
		RightProjection: Linear{
			InDim:  2,
			OutDim: 2,
			Weight: []float32{1, 0, 0, 1},
		},
		OutputProjection: Linear{
			InDim:  4,
			OutDim: 2,
			Weight: []float32{
				1, 0, 2, 0,
				0, 3, 0, 4,
			},
			Bias: []float32{0.5, -0.5},
		},
		LayerNorm: LayerNorm{Weight: []float32{1, 1}, Bias: []float32{0, 0}, Epsilon: 1e-5},
		BosState:  []float32{100, 1},
		EosState:  []float32{2, 200},
	}

	text := [][]float32{{1, 10}, {2, 20}, {30, 300}}
	got, err := enc.Forward(text, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := manualBoundaryEncoder(enc, text, 2)
	if !reflect.DeepEqual(got.Mask, want.Mask) {
		t.Fatalf("mask got=%v want=%v", got.Mask, want.Mask)
	}
	assertCloseMatrix(t, "encoder padded", got.States, want.States, 2e-5)
	if got.States[3][0] != 0 || got.States[3][1] != 0 {
		t.Fatalf("padding boundary leaked: %v", got.States[3])
	}

	empty, err := enc.Forward(nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	wantEmpty := manualBoundaryEncoder(enc, nil, 0)
	if !reflect.DeepEqual(empty.Mask, []bool{true}) {
		t.Fatalf("empty mask=%v", empty.Mask)
	}
	assertCloseMatrix(t, "encoder empty", empty.States, wantEmpty.States, 2e-5)
}

func TestBoundaryEncoderValidation(t *testing.T) {
	if _, err := BuildBoundaryMask(3, 2); err == nil {
		t.Fatal("accepted invalid boundary mask")
	}
	badBlock := BoundaryAttentionBlock{
		NumHeads: 2,
		Window:   -1,
		Norm:     LayerNorm{Weight: []float32{1, 1}, Bias: []float32{0, 0}},
		QKVProjection: Linear{
			InDim:  2,
			OutDim: 6,
			Weight: make([]float32, 12),
		},
		OutputProjection: Linear{InDim: 2, OutDim: 2, Weight: make([]float32, 4)},
	}
	if err := badBlock.Validate(); err == nil {
		t.Fatal("accepted negative attention window")
	}
	badEncoder := BoundaryEncoder{
		HiddenSize:       2,
		BoundaryDim:      2,
		LeftProjection:   Linear{InDim: 2, OutDim: 2, Weight: make([]float32, 4)},
		RightProjection:  Linear{InDim: 2, OutDim: 2, Weight: make([]float32, 4)},
		OutputProjection: Linear{InDim: 4, OutDim: 2, Weight: make([]float32, 7)},
		LayerNorm:        LayerNorm{Weight: []float32{1, 1}, Bias: []float32{0, 0}},
		BosState:         []float32{1, 2},
		EosState:         []float32{3, 4},
	}
	if err := badEncoder.Validate(); err == nil {
		t.Fatal("accepted malformed encoder output projection")
	}
	good := BoundaryEncoder{
		HiddenSize:       2,
		BoundaryDim:      2,
		LeftProjection:   Linear{InDim: 2, OutDim: 2, Weight: make([]float32, 4)},
		RightProjection:  Linear{InDim: 2, OutDim: 2, Weight: make([]float32, 4)},
		OutputProjection: Linear{InDim: 4, OutDim: 2, Weight: make([]float32, 8)},
		LayerNorm:        LayerNorm{Weight: []float32{1, 1}, Bias: []float32{0, 0}},
		BosState:         []float32{1, 2},
		EosState:         []float32{3, 4},
	}
	if _, err := good.Forward([][]float32{{1, 2}, {3}}, 1); err == nil {
		t.Fatal("accepted ragged document")
	}
	if _, err := good.Forward([][]float32{{1, 2}}, 2); err == nil {
		t.Fatal("accepted valid_length beyond padding")
	}
}

func manualBoundaryEncoder(enc BoundaryEncoder, text [][]float32, validLength int) BoundaryEncoding {
	mask := make([]bool, len(text)+1)
	for i := 0; i <= validLength; i++ {
		mask[i] = true
	}
	left := make([][]float32, len(text)+1)
	right := make([][]float32, len(text)+1)
	left[0] = append([]float32(nil), enc.BosState...)
	for i := range text {
		left[i+1] = append([]float32(nil), text[i]...)
		right[i] = append([]float32(nil), text[i]...)
	}
	right[len(text)] = append([]float32(nil), enc.EosState...)
	right[validLength] = append([]float32(nil), enc.EosState...)
	leftP := manualLinearRows(left, enc.LeftProjection)
	rightP := manualLinearRows(right, enc.RightProjection)
	rows := len(left)
	merged := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		merged[i] = append(append([]float32(nil), leftP[i]...), rightP[i]...)
	}
	states := manualLinearRows(merged, enc.OutputProjection)
	states = manualLayerNormRows(states, enc.LayerNorm)
	for _, block := range enc.AttentionBlocks {
		states = manualAttentionBlock(block, states, mask)
	}
	for _, block := range enc.RefinementBlocks {
		states = manualSwiGLUBlock(block, states)
	}
	for i := range states {
		if !mask[i] {
			for j := range states[i] {
				states[i][j] = 0
			}
		}
	}
	return BoundaryEncoding{States: states, Mask: mask}
}

func manualSwiGLUBlock(block ResidualSwiGLU, states [][]float32) [][]float32 {
	normed := manualLayerNormRows(states, block.Norm)
	projected := manualLinearRows(normed, block.InputProjection)
	hidden := block.InputProjection.OutDim / 2
	activated := make([][]float32, len(states))
	for row := range projected {
		activated[row] = make([]float32, hidden)
		for i := 0; i < hidden; i++ {
			activated[row][i] = projected[row][i] * manualSilu(projected[row][hidden+i])
		}
	}
	update := manualLinearRows(activated, block.OutputProjection)
	out := cloneMatrix(states)
	for i := range out {
		for j := range out[i] {
			out[i][j] += update[i][j]
		}
	}
	return out
}

func manualAttentionBlock(block BoundaryAttentionBlock, states [][]float32, mask []bool) [][]float32 {
	normed := manualLayerNormRows(states, block.Norm)
	qkv := manualLinearRows(normed, block.QKVProjection)
	rows := len(states)
	dim := block.Norm.Dim()
	headDim := dim / block.NumHeads
	scale := float32(1 / math.Sqrt(float64(headDim)))
	q := make([][]float32, rows)
	k := make([][]float32, rows)
	v := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		q[i] = append([]float32(nil), qkv[i][:dim]...)
		k[i] = append([]float32(nil), qkv[i][dim:2*dim]...)
		v[i] = append([]float32(nil), qkv[i][2*dim:3*dim]...)
	}
	ctx := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		ctx[i] = make([]float32, dim)
		for h := 0; h < block.NumHeads; h++ {
			scores := make([]float64, rows)
			maxScore := math.Inf(-1)
			for j := 0; j < rows; j++ {
				allowed := mask[j] && (block.Window == 0 || absInt(i-j) <= block.Window)
				if i == j {
					allowed = true
				}
				if !allowed {
					scores[j] = math.Inf(-1)
					continue
				}
				dot := 0.0
				for d := 0; d < headDim; d++ {
					dot += float64(q[i][h*headDim+d]) * float64(k[j][h*headDim+d])
				}
				scores[j] = dot * float64(scale)
				if scores[j] > maxScore {
					maxScore = scores[j]
				}
			}
			sum := 0.0
			probs := make([]float64, rows)
			for j := 0; j < rows; j++ {
				if math.IsInf(scores[j], -1) {
					continue
				}
				probs[j] = math.Exp(scores[j] - maxScore)
				sum += probs[j]
			}
			for j := 0; j < rows; j++ {
				if probs[j] == 0 {
					continue
				}
				w := float32(probs[j] / sum)
				for d := 0; d < headDim; d++ {
					ctx[i][h*headDim+d] += w * v[j][h*headDim+d]
				}
			}
		}
	}
	update := manualLinearRows(ctx, block.OutputProjection)
	out := cloneMatrix(states)
	for i := range out {
		if !mask[i] {
			for j := range out[i] {
				out[i][j] = 0
			}
			continue
		}
		for j := range out[i] {
			out[i][j] += update[i][j]
		}
	}
	return out
}

func manualLinearRows(rows [][]float32, lin Linear) [][]float32 {
	out := make([][]float32, len(rows))
	for r := range rows {
		out[r] = make([]float32, lin.OutDim)
		for o := 0; o < lin.OutDim; o++ {
			sum := float32(0)
			base := o * lin.InDim
			for i := 0; i < lin.InDim; i++ {
				sum += rows[r][i] * lin.Weight[base+i]
			}
			if len(lin.Bias) != 0 {
				sum += lin.Bias[o]
			}
			out[r][o] = sum
		}
	}
	return out
}

func manualLayerNormRows(rows [][]float32, ln LayerNorm) [][]float32 {
	out := make([][]float32, len(rows))
	eps := ln.Epsilon
	if eps == 0 {
		eps = 1e-5
	}
	for r := range rows {
		out[r] = make([]float32, len(rows[r]))
		mean := 0.0
		for _, v := range rows[r] {
			mean += float64(v)
		}
		mean /= float64(len(rows[r]))
		variance := 0.0
		for _, v := range rows[r] {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(len(rows[r]))
		invStd := 1 / math.Sqrt(variance+float64(eps))
		for i, v := range rows[r] {
			norm := float32((float64(v) - mean) * invStd)
			out[r][i] = norm*ln.Weight[i] + ln.Bias[i]
		}
	}
	return out
}

func manualSilu(x float32) float32 {
	if x >= 0 {
		return x / (1 + float32(math.Exp(-float64(x))))
	}
	e := float32(math.Exp(float64(x)))
	return x * e / (1 + e)
}

func cloneMatrix(in [][]float32) [][]float32 {
	out := make([][]float32, len(in))
	for i := range in {
		out[i] = append([]float32(nil), in[i]...)
	}
	return out
}

func rowsClose(a, b [][]float32, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if math.Abs(float64(a[i][j]-b[i][j])) > tol {
				return false
			}
		}
	}
	return true
}

func assertCloseMatrix(t *testing.T, name string, got, want [][]float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s rows got=%d want=%d", name, len(got), len(want))
	}
	for i := range got {
		assertCloseVec(t, name+" row", got[i], want[i], tol)
	}
}

func assertCloseVec(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len got=%d want=%d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("%s[%d] got=%g want=%g", name, i, got[i], want[i])
		}
	}
}
