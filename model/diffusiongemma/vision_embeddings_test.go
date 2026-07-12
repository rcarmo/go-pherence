package diffusiongemma

import (
	"math"
	"strings"
	"testing"
)

func TestGemma4FlattenPatchBCHW(t *testing.T) {
	// BCHW image: C0=[0..15], C1=[100..115], C2=[200..215].
	src := make([]float32, 3*4*4)
	for c := 0; c < 3; c++ {
		for i := 0; i < 16; i++ {
			src[c*16+i] = float32(c*100 + i)
		}
	}
	dst := make([]float32, 3*2*2)
	gemma4FlattenPatchBCHW(dst, src, 4, 4, 2, 1, 1)
	want := []float32{10, 110, 210, 11, 111, 211, 14, 114, 214, 15, 115, 215}
	if len(dst) != len(want) {
		t.Fatalf("len=%d want %d", len(dst), len(want))
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%v want %v full=%v", i, dst[i], want[i], dst)
		}
	}
}

func TestRoundVisionBF16InPlace(t *testing.T) {
	got := []float32{1.3431, -0.0326, 0}
	roundVisionBF16InPlace(got)
	want := []float32{1.34375, -0.03271484375, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%v want %v full=%v", i, got[i], want[i], got)
		}
	}
}

func TestGemma4ScalePatchInputInPlace(t *testing.T) {
	patch := []float32{0, 0.25, 0.5, 0.75, 1}
	gemma4ScalePatchInputInPlace(patch)
	want := []float32{-1, -0.5, 0, 0.5, 1}
	for i := range want {
		if patch[i] != want[i] {
			t.Fatalf("patch[%d]=%v want %v", i, patch[i], want[i])
		}
	}
}

func TestStandardizeVisionSoftTokensF32AfterPooling(t *testing.T) {
	values := []float32{1, 2, 3, 4, -1, -2, -3, -4}
	bias := []float32{1, 2, 3, 4}
	scale := []float32{1, 2, 3, 4}
	if err := standardizeVisionSoftTokensF32(values, 4, bias, scale); err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 4, 9, 16, -3, -12, -27, -48}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("values[%d]=%v want %v full=%v", i, values[i], want[i], values)
		}
	}
}

func TestStandardizeVisionSoftTokensF32RejectsShapeMismatch(t *testing.T) {
	if err := standardizeVisionSoftTokensF32([]float32{1, 2, 3}, 2, []float32{0, 0}, []float32{1, 1}); err == nil {
		t.Fatal("expected shape mismatch")
	}
}

func TestNormalizeVisionSoftTokensForProjectionF32(t *testing.T) {
	values := []float32{3, 4, 0, 5}
	if err := normalizeVisionSoftTokensForProjectionF32(values, 2); err != nil {
		t.Fatal(err)
	}
	wantScale := []float64{
		1 / math.Sqrt((9+16)/2.0+1e-6),
		1 / math.Sqrt((0+25)/2.0+1e-6),
	}
	want := []float64{3 * wantScale[0], 4 * wantScale[0], 0, 5 * wantScale[1]}
	for i := range want {
		if delta := math.Abs(float64(values[i]) - want[i]); delta > 1e-6 {
			t.Fatalf("values[%d]=%.9g want %.9g delta %.9g", i, values[i], want[i], delta)
		}
	}
}

func TestNormalizeVisionSoftTokensForProjectionF32RejectsShapeMismatch(t *testing.T) {
	if err := normalizeVisionSoftTokensForProjectionF32([]float32{1, 2, 3}, 2); err == nil {
		t.Fatal("expected shape mismatch")
	}
}

func TestInsertImageEmbeddings(t *testing.T) {
	tokens := []int{105, 258880, 258880, 106}
	emb := make([]float32, len(tokens)*3)
	img := ImageEmbeddingResult{Embeddings: []float32{1, 2, 3, 4, 5, 6}, Shape: [3]int{1, 2, 3}}
	used, err := InsertImageEmbeddings(emb, tokens, img, 258880, 3)
	if err != nil {
		t.Fatal(err)
	}
	if used != 2 {
		t.Fatalf("used=%d want 2", used)
	}
	want := []float32{0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0}
	for i := range want {
		if emb[i] != want[i] {
			t.Fatalf("emb[%d]=%v want %v full=%v", i, emb[i], want[i], emb)
		}
	}
}

func TestInsertImageEmbeddingsRequiresExactImageTokenCount(t *testing.T) {
	emb := make([]float32, 3*2)
	img := ImageEmbeddingResult{Embeddings: []float32{1, 2, 3, 4}, Shape: [3]int{1, 2, 2}}
	if _, err := InsertImageEmbeddings(emb, []int{258880, 7, 8}, img, 258880, 2); err == nil {
		t.Fatal("expected image-token count mismatch")
	}
}

func TestAddVisionPatchXYPositionEmbeddingSyntheticNonSquareGrid(t *testing.T) {
	pos := FloatTensor{
		Shape: []int{2, 3, 2},
		Data: []float32{
			1, 10,
			2, 20,
			3, 30,
			100, 1000,
			200, 2000,
			300, 3000,
		},
	}
	patchW, patchH := 2, 3
	want := []struct {
		px, py int
		row    []float32
	}{
		{px: 0, py: 0, row: []float32{101, 1010}},
		{px: 1, py: 0, row: []float32{102, 1020}},
		{px: 0, py: 1, row: []float32{201, 2010}},
		{px: 1, py: 1, row: []float32{202, 2020}},
		{px: 0, py: 2, row: []float32{301, 3010}},
		{px: 1, py: 2, row: []float32{302, 3020}},
	}
	for i, tc := range want {
		if tc.px >= patchW || tc.py >= patchH {
			t.Fatalf("bad test case %d coords=(%d,%d) grid=%dx%d", i, tc.px, tc.py, patchW, patchH)
		}
		row := []float32{0, 0}
		if err := addVisionPatchXYPositionEmbedding(row, pos, tc.px, tc.py); err != nil {
			t.Fatalf("coords=(%d,%d): %v", tc.px, tc.py, err)
		}
		for j := range tc.row {
			if row[j] != tc.row[j] {
				t.Fatalf("coords=(%d,%d) row[%d]=%v want %v full=%v", tc.px, tc.py, j, row[j], tc.row[j], row)
			}
		}
	}
}

func TestAddVisionPatchXYPositionEmbeddingRejectsOutOfBounds(t *testing.T) {
	pos := FloatTensor{Shape: []int{2, 3, 1}, Data: []float32{1, 2, 3, 10, 20, 30}}
	for _, tc := range []struct{ px, py int }{{px: 3, py: 0}, {px: 0, py: 3}} {
		row := []float32{0}
		err := addVisionPatchXYPositionEmbedding(row, pos, tc.px, tc.py)
		if err == nil || !strings.Contains(err.Error(), "out of bounds") {
			t.Fatalf("coords=(%d,%d) err=%v want out-of-bounds", tc.px, tc.py, err)
		}
	}
}

func TestLocalDiffusionGemmaLoadVisionLayerF32(t *testing.T) {
	dir := localDiffusionGemmaModelDir(t)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	w, err := OpenVisionWeights(dir, meta.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	layer, err := LoadVisionLayerF32(w, 0)
	if err != nil {
		t.Fatal(err)
	}
	hidden := meta.Shape.VisionHiddenSize
	headDim := hidden / meta.Shape.VisionHeads
	if len(layer.InputLayerNorm) != hidden || len(layer.QNorm) != headDim || len(layer.KNorm) != headDim {
		t.Fatalf("norm shapes input=%d q=%d k=%d hidden=%d head_dim=%d", len(layer.InputLayerNorm), len(layer.QNorm), len(layer.KNorm), hidden, headDim)
	}
	if len(layer.QProj) != hidden*hidden || len(layer.KProj) != hidden*hidden || len(layer.VProj) != hidden*hidden || len(layer.OProj) != hidden*hidden {
		t.Fatalf("attention matrix shape mismatch q=%d hidden=%d", len(layer.QProj), hidden)
	}
	if layer.MLPIntermediate != meta.Shape.VisionIntermediateSize || len(layer.MLPGateProj) != layer.MLPIntermediate*hidden || len(layer.MLPDownProj) != hidden*layer.MLPIntermediate {
		t.Fatalf("MLP shape mismatch intermediate=%d configured=%d", layer.MLPIntermediate, meta.Shape.VisionIntermediateSize)
	}
}

func TestLocalDiffusionGemmaOpenVisionWeights(t *testing.T) {
	dir := localDiffusionGemmaModelDir(t)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	w, err := OpenVisionWeights(dir, meta.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if len(w.Globals) != 5 || len(w.Layers) != meta.Shape.VisionLayers {
		t.Fatalf("vision weights globals=%d layers=%d", len(w.Globals), len(w.Layers))
	}
	proj, shape, err := w.RawBF16Tensor("model.encoder.embed_vision.embedding_projection.weight")
	if err != nil {
		t.Fatal(err)
	}
	if proj == nil || len(shape) != 2 || shape[0] != meta.Shape.TextHiddenSize || shape[1] != meta.Shape.VisionHiddenSize {
		t.Fatalf("embed_vision projection shape=%v nil=%v", shape, proj == nil)
	}
	fp := w.ForwardPlan()
	if !fp.Ready || len(fp.Missing) != 0 {
		t.Fatalf("vision forward plan ready=%v missing=%v", fp.Ready, fp.Missing)
	}
	if fp.Globals.PatchInputProj == nil || fp.Globals.PositionEmbeddingTable == nil || fp.Globals.StdBias == nil || fp.Globals.StdScale == nil || fp.Globals.EmbeddingProjection == nil {
		t.Fatalf("vision forward globals incomplete: %+v", fp.Globals)
	}
	if len(fp.Layers) != meta.Shape.VisionLayers {
		t.Fatalf("vision forward layers=%d want %d", len(fp.Layers), meta.Shape.VisionLayers)
	}
	for _, layer := range fp.Layers {
		if layer.QProj == nil || layer.KProj == nil || layer.VProj == nil || layer.OProj == nil || layer.MLPGateProj == nil || layer.MLPUpProj == nil || layer.MLPDownProj == nil {
			t.Fatalf("vision layer %d incomplete: %+v", layer.Layer, layer)
		}
	}
}
