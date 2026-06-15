package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

func TestProjectMTPVerifierLayerQKVBatchMatchesRows(t *testing.T) {
	m := newSingleLayerVerifierModel()
	m.Large = true
	m.Layers[0].QW = tensor.FromFloat32([]float32{1, 2, 3, 4}, []int{2, 2})
	m.Layers[0].KW = tensor.FromFloat32([]float32{2, -1, 1, 3}, []int{2, 2})
	m.Layers[0].VW = tensor.FromFloat32([]float32{-1, 2, 4, 1}, []int{2, 2})
	m.RopeFreqs = nil
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	hiddenFlat := []float32{0.25, -0.5, 1.5, 0.75}
	got, err := m.ProjectMTPVerifierLayerQKVBatch(batch, 0, hiddenFlat)
	if err != nil {
		t.Fatal(err)
	}
	if got.QDim != 2 || got.KVDim != 2 || !got.HasKV || len(got.NormedIn) != len(hiddenFlat) {
		t.Fatalf("projection shape=%+v", got)
	}
	for b := 0; b < 2; b++ {
		singlePlan := mustMTPVerifierPlan(t, m, plan.VerifierTokens[b], nil, plan.Positions[b])
		singleBatch, err := NewMTPVerifierBatchInputs(m, singlePlan)
		if err != nil {
			t.Fatal(err)
		}
		single, err := m.ProjectMTPVerifierLayerQKVBatch(singleBatch, 0, hiddenFlat[b*2:(b+1)*2])
		if err != nil {
			t.Fatal(err)
		}
		if !sameFloat32s(got.Q[b*2:(b+1)*2], single.Q) || !sameFloat32s(got.K[b*2:(b+1)*2], single.K) || !sameFloat32s(got.V[b*2:(b+1)*2], single.V) {
			t.Fatalf("row %d batch q/k/v=%v/%v/%v single=%v/%v/%v", b, got.Q[b*2:(b+1)*2], got.K[b*2:(b+1)*2], got.V[b*2:(b+1)*2], single.Q, single.K, single.V)
		}
	}
}

func TestProjectMTPVerifierLayerQKVBatchDerivesGemma4FullHeadDim(t *testing.T) {
	identity4 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	m := &LlamaModel{
		Config: LlamaConfig{ModelType: "gemma4_text", VocabSize: 2, HiddenSize: 4, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, NumGlobalKVHeads: 1, HeadDim: 2, GlobalHeadDim: 4, LayerTypes: []string{"full_attention"}},
		EmbedTokens: tensor.FromFloat32([]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		}, []int{2, 4}),
		Norm:   tensor.Ones([]int{4}),
		LMHead: tensor.FromFloat32([]float32{1, 0, 0, 0, 0, 1, 0, 0}, []int{2, 4}),
		Layers: []LlamaLayer{{
			InputNorm: tensor.Ones([]int{4}),
			PostNorm:  tensor.Ones([]int{4}),
			HasKV:     true,
			QW:        tensor.FromFloat32(append([]float32(nil), identity4...), []int{4, 4}),
			KW:        tensor.FromFloat32(append([]float32(nil), identity4...), []int{4, 4}),
			VW:        tensor.FromFloat32(append([]float32(nil), identity4...), []int{4, 4}),
			OW:        tensor.FromFloat32(append([]float32(nil), identity4...), []int{4, 4}),
			GateW:     tensor.FromFloat32(append([]float32(nil), identity4...), []int{4, 4}),
			UpW:       tensor.FromFloat32(append([]float32(nil), identity4...), []int{4, 4}),
			DownW:     tensor.FromFloat32(append([]float32(nil), identity4...), []int{4, 4}),
		}},
	}
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.ProjectMTPVerifierLayerQKVBatch(batch, 0, batch.HiddenFlat)
	if err != nil {
		t.Fatal(err)
	}
	if got.HeadDim != 4 || got.QDim != 4 || got.KVDim != 4 || len(got.Q) != 8 || len(got.K) != 8 || len(got.V) != 8 {
		t.Fatalf("derived full-attention shape=%+v lens q/k/v=%d/%d/%d", got, len(got.Q), len(got.K), len(got.V))
	}
}

func TestProjectMTPVerifierLayerQKVBatchGemma4KEqV(t *testing.T) {
	m := newSingleLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	m.Config.AttentionKEqV = true
	m.Config.RMSNormEps = 1e-6
	m.RopeFreqs = nil
	m.Layers[0].KW = tensor.FromFloat32([]float32{1, 2, 3, 4}, []int{2, 2})
	m.Layers[0].VW = m.Layers[0].KW
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.ProjectMTPVerifierLayerQKVBatch(batch, 0, batch.HiddenFlat)
	if err != nil {
		t.Fatal(err)
	}
	if sameFloat32s(got.K, got.V) {
		t.Fatalf("Gemma4 V should be no-scale normalized after K=V copy; got identical K/V %v", got.K)
	}
	if len(got.K) != len(got.V) || len(got.K) != len(batch.HiddenFlat) {
		t.Fatalf("K/V lengths K=%d V=%d hidden=%d", len(got.K), len(got.V), len(batch.HiddenFlat))
	}
}

func TestProjectMTPVerifierLayerQKVBatchValidation(t *testing.T) {
	m := newSingleLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := (*LlamaModel)(nil).ProjectMTPVerifierLayerQKVBatch(batch, 0, batch.HiddenFlat); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, err := m.ProjectMTPVerifierLayerQKVBatch(batch, -1, batch.HiddenFlat); err == nil {
		t.Fatal("accepted bad layer")
	}
	if _, err := m.ProjectMTPVerifierLayerQKVBatch(batch, 0, batch.HiddenFlat[:1]); err == nil {
		t.Fatal("accepted short hidden flat")
	}
	bad := *m
	bad.Layers = append([]LlamaLayer(nil), m.Layers...)
	bad.Layers[0].QW = nil
	if _, err := (&bad).ProjectMTPVerifierLayerQKVBatch(batch, 0, batch.HiddenFlat); err == nil {
		t.Fatal("accepted missing Q weight")
	}
}
