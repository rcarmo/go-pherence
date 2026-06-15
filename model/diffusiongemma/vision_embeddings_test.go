package diffusiongemma

import "testing"

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
	want := []float32{10, 11, 14, 15, 110, 111, 114, 115, 210, 211, 214, 215}
	if len(dst) != len(want) {
		t.Fatalf("len=%d want %d", len(dst), len(want))
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%v want %v full=%v", i, dst[i], want[i], dst)
		}
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
