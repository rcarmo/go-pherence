package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

func TestRunMTPVerifierBatchForwardZeroLayer(t *testing.T) {
	m := newZeroLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 4)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.RunMTPVerifierBatchForward(batch, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	seq, err := m.RunMTPVerifierForward(plan, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(got.VerifierTokens, seq.VerifierTokens) || !sameInts(got.Acceptance.OutputTokens, seq.Acceptance.OutputTokens) || !sameFloat32s(got.FinalActivation, seq.FinalActivation) {
		t.Fatalf("batch=%+v seq=%+v", got, seq)
	}
	if len(got.Logits) != len(seq.Logits) || !sameFloat32s(got.Logits[0], seq.Logits[0]) || !sameFloat32s(got.Logits[1], seq.Logits[1]) {
		t.Fatalf("batch logits=%v seq=%v", got.Logits, seq.Logits)
	}
}

func TestMTPVerifierBatchLayerLoweringDefaultOff(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS", "")
	if mtpVerifierBatchLayerLoweringEnabled() {
		t.Fatal("batch layer lowering enabled by default")
	}
	m := newSingleLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{0}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	if m.mtpVerifierBatchLayerEligible(batch) {
		t.Fatal("eligible batch layer path should be gated off by default")
	}
}

func TestRunMTPVerifierBatchForwardLayeredSequentialFallback(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS", "")
	m := newSingleLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{0}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	got, err := m.RunMTPVerifierBatchForward(batch, kvCacheK, kvCacheV)
	if err != nil {
		t.Fatal(err)
	}
	seqK := make([][]float32, len(m.Layers))
	seqV := make([][]float32, len(m.Layers))
	seqHidden, err := m.runMTPVerifierBatchRowsSequential(batch, seqK, seqV)
	if err != nil {
		t.Fatal(err)
	}
	_, seqLogits, _, err := m.FinishCPUDecodeBatch(seqHidden)
	if err != nil {
		t.Fatal(err)
	}
	assertMTPVerifierBatchMatchesSequential(t, m, plan, got, kvCacheK, kvCacheV, seqLogits, seqK, seqV)
}

func TestRunMTPVerifierBatchForwardLayeredExperimentalMatchesSequential(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS", "1")
	m := newSingleLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{0}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	got, err := m.RunMTPVerifierBatchForward(batch, kvCacheK, kvCacheV)
	if err != nil {
		t.Fatal(err)
	}
	seqK := make([][]float32, len(m.Layers))
	seqV := make([][]float32, len(m.Layers))
	seqHidden, err := m.runMTPVerifierBatchRowsSequential(batch, seqK, seqV)
	if err != nil {
		t.Fatal(err)
	}
	_, seqLogits, _, err := m.FinishCPUDecodeBatch(seqHidden)
	if err != nil {
		t.Fatal(err)
	}
	assertMTPVerifierBatchMatchesSequential(t, m, plan, got, kvCacheK, kvCacheV, seqLogits, seqK, seqV)
}

func TestRunMTPVerifierBatchForwardLayeredExperimentalSlidingWindowMatchesSequential(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS", "1")
	m := newSingleLayerVerifierModel()
	m.Config.SlidingWindow = 2
	m.Config.LayerTypes = []string{"sliding_attention"}
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 4)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(batch.Attention.Layers[0].KVStart, []int{3, 4}) || !sameInts(batch.Attention.Layers[0].KVEndExclusive, []int{5, 6}) {
		t.Fatalf("sliding verifier attention=%+v", batch.Attention.Layers[0])
	}
	initialK := []float32{1, 0, 0, 1, 1, 1, -1, 1}
	initialV := []float32{0, 1, 1, 0, -1, 1, 1, 1}
	kvCacheK := [][]float32{append([]float32(nil), initialK...)}
	kvCacheV := [][]float32{append([]float32(nil), initialV...)}
	got, err := m.RunMTPVerifierBatchForward(batch, kvCacheK, kvCacheV)
	if err != nil {
		t.Fatal(err)
	}
	seqK := [][]float32{append([]float32(nil), initialK...)}
	seqV := [][]float32{append([]float32(nil), initialV...)}
	seqHidden, err := m.runMTPVerifierBatchRowsSequential(batch, seqK, seqV)
	if err != nil {
		t.Fatal(err)
	}
	_, seqLogits, _, err := m.FinishCPUDecodeBatch(seqHidden)
	if err != nil {
		t.Fatal(err)
	}
	assertMTPVerifierBatchMatchesSequential(t, m, plan, got, kvCacheK, kvCacheV, seqLogits, seqK, seqV)
}

func TestRunMTPVerifierBatchForwardLayeredExperimentalGemma4MixedAttentionMatchesSequential(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS", "1")
	identity2 := []float32{1, 0, 0, 1}
	identity4 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	dense := func(data []float32, shape ...int) *tensor.Tensor {
		return tensor.FromFloat32(append([]float32(nil), data...), shape)
	}
	m := &LlamaModel{
		Config: LlamaConfig{ModelType: "gemma4_text", VocabSize: 3, HiddenSize: 4, NumLayers: 2, NumHeads: 1, NumKVHeads: 1, NumGlobalKVHeads: 1, HeadDim: 2, GlobalHeadDim: 4, Intermediate: 4, RMSNormEps: 1e-6, HiddenAct: "gelu_pytorch_tanh", LayerTypes: []string{"sliding_attention", "full_attention"}, SlidingWindow: 2},
		EmbedTokens: tensor.FromFloat32([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
		}, []int{3, 4}),
		Norm:   tensor.Ones([]int{4}),
		LMHead: tensor.FromFloat32([]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, []int{3, 4}),
		Layers: []LlamaLayer{
			{
				InputNorm: tensor.Ones([]int{4}), PostNorm: tensor.Ones([]int{4}), PostFFNNorm: tensor.Ones([]int{4}), LayerScalar: 1, HasKV: true,
				QW: dense([]float32{1, 0, 0, 0, 0, 1, 0, 0}, 4, 2), KW: dense([]float32{1, 0, 0, 0, 0, 1, 0, 0}, 4, 2), VW: dense([]float32{1, 0, 0, 0, 0, 1, 0, 0}, 4, 2), OW: dense([]float32{1, 0, 0, 0, 0, 1, 0, 0}, 2, 4),
				GateW: dense(identity4, 4, 4), UpW: dense(identity4, 4, 4), DownW: dense(identity4, 4, 4), QNorm: tensor.Ones([]int{2}), KNorm: tensor.Ones([]int{2}),
			},
			{
				InputNorm: tensor.Ones([]int{4}), PostNorm: tensor.Ones([]int{4}), PostFFNNorm: tensor.Ones([]int{4}), LayerScalar: 1, HasKV: true,
				QW: dense(identity4, 4, 4), KW: dense(identity4, 4, 4), VW: dense(identity4, 4, 4), OW: dense(identity4, 4, 4), GateW: dense(identity4, 4, 4), UpW: dense(identity4, 4, 4), DownW: dense(identity4, 4, 4), QNorm: tensor.Ones([]int{4}), KNorm: tensor.Ones([]int{4}),
			},
		},
	}
	m.RopeFreqsSWA = nil
	m.RopeFreqsFull = nil
	m.RopeFreqs = nil
	_ = identity2
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 2)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	if !m.mtpVerifierBatchLayerEligible(batch) {
		t.Fatal("mixed Gemma4 verifier batch layer should be eligible when gated on")
	}
	initialK0 := []float32{1, 0, 0, 1}
	initialV0 := []float32{0, 1, 1, 0}
	initialK1 := []float32{1, 0, 0, 1, 1, 1, 0, 0}
	initialV1 := []float32{0, 1, 1, 0, 0, 0, 1, 1}
	kvCacheK := [][]float32{append([]float32(nil), initialK0...), append([]float32(nil), initialK1...)}
	kvCacheV := [][]float32{append([]float32(nil), initialV0...), append([]float32(nil), initialV1...)}
	got, err := m.RunMTPVerifierBatchForward(batch, kvCacheK, kvCacheV)
	if err != nil {
		t.Fatal(err)
	}
	seqK := [][]float32{append([]float32(nil), initialK0...), append([]float32(nil), initialK1...)}
	seqV := [][]float32{append([]float32(nil), initialV0...), append([]float32(nil), initialV1...)}
	seqHidden, err := m.runMTPVerifierBatchRowsSequential(batch, seqK, seqV)
	if err != nil {
		t.Fatal(err)
	}
	_, seqLogits, _, err := m.FinishCPUDecodeBatch(seqHidden)
	if err != nil {
		t.Fatal(err)
	}
	assertMTPVerifierBatchMatchesSequential(t, m, plan, got, kvCacheK, kvCacheV, seqLogits, seqK, seqV)
}

func assertMTPVerifierBatchMatchesSequential(t *testing.T, m *LlamaModel, plan MTPVerifierPlan, got MTPVerifierResult, kvCacheK, kvCacheV [][]float32, seqLogits [][]float32, seqK, seqV [][]float32) {
	t.Helper()
	if len(got.Logits) != len(plan.VerifierTokens) || len(got.FinalActivation) != m.Config.HiddenSize {
		t.Fatalf("batch result logits=%d activation=%d", len(got.Logits), len(got.FinalActivation))
	}
	for i := range got.Logits {
		if !sameFloat32s(got.Logits[i], seqLogits[i]) {
			t.Fatalf("logits row %d batch=%v seq=%v", i, got.Logits[i], seqLogits[i])
		}
	}
	wantKVTokens := plan.StartPos + len(plan.VerifierTokens)
	for l := range m.Layers {
		kvDim, err := m.LayerKVDim(l)
		if err != nil {
			t.Fatal(err)
		}
		if kvDim == 0 {
			if len(kvCacheK[l]) != 0 || len(kvCacheV[l]) != 0 || len(seqK[l]) != 0 || len(seqV[l]) != 0 {
				t.Fatalf("shared/non-KV layer %d has KV batch=%v/%v seq=%v/%v", l, kvCacheK[l], kvCacheV[l], seqK[l], seqV[l])
			}
			continue
		}
		if got, want := len(kvCacheK[l]), wantKVTokens*kvDim; got != want {
			t.Fatalf("layer %d batch staged K len=%d want %d", l, got, want)
		}
		if !sameFloat32s(kvCacheK[l], seqK[l]) || !sameFloat32s(kvCacheV[l], seqV[l]) {
			t.Fatalf("layer %d KV batch K/V=%v/%v seq=%v/%v", l, kvCacheK[l], kvCacheV[l], seqK[l], seqV[l])
		}
	}
}

func TestRunMTPVerifierBatchLayersRejectsOverflowingKVRange(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS", "1")
	m := newSingleLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{0}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	maxInt := int(^uint(0) >> 1)
	batch.Attention.Layers[0].KVStart[0] = maxInt/2 + 1
	batch.Attention.Layers[0].KVEndExclusive[0] = maxInt/2 + 2
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	if _, ok, err := m.runMTPVerifierBatchLayers(batch, kvCacheK, kvCacheV); !ok || err == nil {
		t.Fatalf("runMTPVerifierBatchLayers ok=%v err=%v, want checked range rejection", ok, err)
	}
}

func TestRunMTPVerifierBatchForwardLayeredExperimentalPLIMatchesSequential(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS", "1")
	m := newSingleLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	m.Config.HiddenPerLayer = 2
	m.PerLayerModelProj = []float32{1, 0, 0, 1}
	m.PerLayerProjNorm = []float32{1, 1}
	m.PerLayerProjScale = 1
	m.PerLayerInputScale = 1
	m.EmbedPerLayerScale = 1
	m.Layers[0].PLIGate = []float32{1, 0, 0, 1}
	m.Layers[0].PLIProj = []float32{1, 0, 0, 1}
	m.Layers[0].PLIPostNorm = []float32{1, 1}
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	if !m.mtpVerifierBatchLayerEligible(batch) {
		t.Fatal("PLI verifier batch layer should be eligible when gated on")
	}
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	got, err := m.RunMTPVerifierBatchForward(batch, kvCacheK, kvCacheV)
	if err != nil {
		t.Fatal(err)
	}
	seqK := make([][]float32, len(m.Layers))
	seqV := make([][]float32, len(m.Layers))
	seqHidden, err := m.runMTPVerifierBatchRowsSequential(batch, seqK, seqV)
	if err != nil {
		t.Fatal(err)
	}
	_, seqLogits, _, err := m.FinishCPUDecodeBatch(seqHidden)
	if err != nil {
		t.Fatal(err)
	}
	assertMTPVerifierBatchMatchesSequential(t, m, plan, got, kvCacheK, kvCacheV, seqLogits, seqK, seqV)
}

func TestRunMTPVerifierBatchForwardLayeredExperimentalQuantizedMatchesSequential(t *testing.T) {
	t.Setenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS", "1")
	m := &LlamaModel{
		Config: LlamaConfig{VocabSize: 3, HiddenSize: 8, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 8, Intermediate: 8, RMSNormEps: 1e-6},
		EmbedTokens: tensor.FromFloat32([]float32{
			1, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 1, 0, 0, 0, 0, 0,
		}, []int{3, 8}),
		Norm: tensor.Ones([]int{8}),
		LMHead: tensor.FromFloat32([]float32{
			1, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 1, 0, 0, 0, 0, 0,
		}, []int{3, 8}),
		Layers: []LlamaLayer{{
			InputNorm: tensor.Ones([]int{8}),
			PostNorm:  tensor.Ones([]int{8}),
			HasKV:     true,
			QWq:       syntheticSymQ4Weight(8, 8, 9),
			KWq:       syntheticSymQ4Weight(8, 8, 10),
			VWq:       syntheticSymQ4Weight(8, 8, 9),
			OWq:       syntheticSymQ4Weight(8, 8, 9),
			GateWq:    syntheticSymQ4Weight(8, 8, 9),
			UpWq:      syntheticSymQ4Weight(8, 8, 10),
			DownWq:    syntheticSymQ4Weight(8, 8, 9),
		}},
	}
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	if !m.mtpVerifierBatchLayerEligible(batch) {
		t.Fatal("quantized verifier batch layer should be eligible when gated on")
	}
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	got, err := m.RunMTPVerifierBatchForward(batch, kvCacheK, kvCacheV)
	if err != nil {
		t.Fatal(err)
	}
	seqK := make([][]float32, len(m.Layers))
	seqV := make([][]float32, len(m.Layers))
	seqHidden, err := m.runMTPVerifierBatchRowsSequential(batch, seqK, seqV)
	if err != nil {
		t.Fatal(err)
	}
	_, seqLogits, _, err := m.FinishCPUDecodeBatch(seqHidden)
	if err != nil {
		t.Fatal(err)
	}
	assertMTPVerifierBatchMatchesSequential(t, m, plan, got, kvCacheK, kvCacheV, seqLogits, seqK, seqV)
}

func TestRunMTPVerifierBatchForwardValidation(t *testing.T) {
	m := newZeroLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	bad := batch
	bad.HiddenRows = bad.HiddenRows[:1]
	if _, err := m.RunMTPVerifierBatchForward(bad, nil, nil); err == nil {
		t.Fatal("accepted short hidden rows")
	}
	bad = batch
	bad.HiddenRows = append([][]float32(nil), batch.HiddenRows...)
	bad.HiddenRows[0] = []float32{1}
	if _, err := m.RunMTPVerifierBatchForward(bad, nil, nil); err == nil {
		t.Fatal("accepted wrong hidden width")
	}
	bad = batch
	bad.HiddenRows = cloneMTPVerifierHiddenRows(batch.HiddenRows)
	bad.HiddenFlat = append([]float32(nil), batch.HiddenFlat...)
	bad.HiddenFlat[0]++
	if _, err := m.RunMTPVerifierBatchForward(bad, nil, nil); err == nil {
		t.Fatal("accepted hidden rows that disagree with flat buffer")
	}
	bad = batch
	bad.PerLayerInputs = nil
	if _, err := m.RunMTPVerifierBatchForward(bad, nil, nil); err == nil {
		t.Fatal("accepted missing disabled-PLI row slots")
	}
	bad = batch
	bad.Scratch.Batch++
	if _, err := m.RunMTPVerifierBatchForward(bad, nil, nil); err == nil {
		t.Fatal("accepted stale verifier scratch plan")
	}
	bad = batch
	bad.Attention.KVLen = 99
	if _, err := m.RunMTPVerifierBatchForward(bad, nil, nil); err == nil {
		t.Fatal("accepted bad attention plan")
	}

	mPLI := newSingleLayerVerifierModel()
	mPLI.Config.ModelType = "gemma4_text"
	mPLI.Config.HiddenPerLayer = 2
	mPLI.PerLayerModelProj = []float32{1, 0, 0, 1}
	mPLI.PerLayerProjNorm = []float32{1, 1}
	mPLI.PerLayerProjScale = 1
	mPLI.PerLayerInputScale = 1
	mPLI.EmbedPerLayerScale = 1
	mPLI.EmbedPerLayer = []float32{
		0, 0,
		10, 20,
		30, 40,
	}
	mPLI.Config.VocabPerLayer = 3
	planPLI := mustMTPVerifierPlan(t, mPLI, 0, []int{1}, 0)
	batchPLI, err := NewMTPVerifierBatchInputs(mPLI, planPLI)
	if err != nil {
		t.Fatal(err)
	}
	badPLI := batchPLI
	badPLI.PerLayerInputs = cloneMTPVerifierPLIRows(batchPLI.PerLayerInputs)
	badPLI.PerLayerInputFlat = append([]float32(nil), batchPLI.PerLayerInputFlat...)
	badPLI.PerLayerInputFlat[0]++
	if _, err := mPLI.RunMTPVerifierBatchForward(badPLI, nil, nil); err == nil {
		t.Fatal("accepted PLI rows that disagree with flat buffer")
	}
}

func cloneMTPVerifierPLIRows(in [][][]float32) [][][]float32 {
	out := make([][][]float32, len(in))
	for i := range in {
		out[i] = clonePerLayerInputs(in[i])
	}
	return out
}

func cloneMTPVerifierHiddenRows(in [][]float32) [][]float32 {
	out := make([][]float32, len(in))
	for i := range in {
		out[i] = append([]float32(nil), in[i]...)
	}
	return out
}
