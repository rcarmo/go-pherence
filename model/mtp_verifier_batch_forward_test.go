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
	kvDim, err := m.LayerKVDim(0)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(kvCacheK[0]), len(plan.VerifierTokens)*kvDim; got != want {
		t.Fatalf("batch staged K len=%d want %d", got, want)
	}
	if !sameFloat32s(kvCacheK[0], seqK[0]) || !sameFloat32s(kvCacheV[0], seqV[0]) {
		t.Fatalf("KV batch K/V=%v/%v seq=%v/%v", kvCacheK[0], kvCacheV[0], seqK[0], seqV[0])
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
	bad.Scratch.Batch++
	if _, err := m.RunMTPVerifierBatchForward(bad, nil, nil); err == nil {
		t.Fatal("accepted stale verifier scratch plan")
	}
	bad = batch
	bad.Attention.KVLen = 99
	if _, err := m.RunMTPVerifierBatchForward(bad, nil, nil); err == nil {
		t.Fatal("accepted bad attention plan")
	}
}

func cloneMTPVerifierHiddenRows(in [][]float32) [][]float32 {
	out := make([][]float32, len(in))
	for i := range in {
		out[i] = append([]float32(nil), in[i]...)
	}
	return out
}
