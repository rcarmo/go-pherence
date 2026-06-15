package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

func syntheticSymQ4Weight(inDim, outDim int, q int32) *QuantWeight {
	qw := &QuantWeight{InDim: inDim, OutDim: outDim, QWeight: make([]int32, (inDim/8)*outDim), GIdx: make([]int32, inDim), Scales: make([]float32, outDim)}
	for i := range qw.Scales {
		qw.Scales[i] = 1
	}
	packed := int32(0)
	for bit := 0; bit < 8; bit++ {
		packed |= (q & 0xF) << (uint(bit) * 4)
	}
	for i := range qw.QWeight {
		qw.QWeight[i] = packed
	}
	return qw
}

func TestMTPVerifierProjBatchAnyQuantMatchesRows(t *testing.T) {
	m := &LlamaModel{}
	qw := syntheticSymQ4Weight(8, 2, 9) // each unpacked weight is +1 after symmetric zero-point 8
	x := []float32{
		1, 2, 3, 4, 5, 6, 7, 8,
		-1, -2, 3, 4, -5, 6, -7, 8,
	}
	out := make([]float32, 4)
	if !m.projBatchAny(out, x, 2, nil, nil, qw, 8, 2) {
		t.Fatal("projBatchAny quant rejected")
	}
	for b := 0; b < 2; b++ {
		row := make([]float32, 2)
		m.mvQ(row, x[b*8:(b+1)*8], qw)
		if !sameFloat32s(out[b*2:(b+1)*2], row) {
			t.Fatalf("row %d batch=%v row=%v", b, out[b*2:(b+1)*2], row)
		}
	}
	if m.projBatchAny(make([]float32, 1), x, 2, nil, nil, qw, 8, 2) {
		t.Fatal("projBatchAny quant accepted short output")
	}
}

func TestProjectMTPVerifierLayerQKVBatchQuantKEqV(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{VocabSize: 3, HiddenSize: 8, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 8, Intermediate: 8, AttentionKEqV: true, RMSNormEps: 1e-6},
		Layers: []LlamaLayer{{
			InputNorm: tensor.Ones([]int{8}), HasKV: true,
			QWq: syntheticSymQ4Weight(8, 8, 9),
			KWq: syntheticSymQ4Weight(8, 8, 10),
		}},
	}
	batch := MTPVerifierBatchInputs{Plan: MTPVerifierPlan{InputToken: 0, VerifierTokens: []int{0, 1}, DraftedTokens: []int{1}, Positions: []int{0, 1}}, HiddenFlat: make([]float32, 16), HiddenRows: make([][]float32, 2)}
	for i := range batch.HiddenFlat {
		batch.HiddenFlat[i] = float32((i%7)-3) * 0.25
	}
	batch.HiddenRows[0] = batch.HiddenFlat[:8]
	batch.HiddenRows[1] = batch.HiddenFlat[8:]
	got, err := m.ProjectMTPVerifierLayerQKVBatch(batch, 0, batch.HiddenFlat)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.K) != 16 || len(got.V) != 16 {
		t.Fatalf("K/V len=%d/%d", len(got.K), len(got.V))
	}
	if !sameFloat32s(got.K, got.V) {
		t.Fatalf("quant K=V path diverged K=%v V=%v", got.K, got.V)
	}
	m.Layers[0].VWq = m.Layers[0].KWq
	got, err = m.ProjectMTPVerifierLayerQKVBatch(batch, 0, batch.HiddenFlat)
	if err != nil {
		t.Fatal(err)
	}
	if !sameFloat32s(got.K, got.V) {
		t.Fatalf("quant shared-pointer K=V path diverged K=%v V=%v", got.K, got.V)
	}
}

func TestForwardLayerQuantKEqV(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{VocabSize: 3, HiddenSize: 8, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 8, Intermediate: 8, AttentionKEqV: true, RMSNormEps: 1e-6},
		Layers: []LlamaLayer{{
			InputNorm: tensor.Ones([]int{8}), PostNorm: tensor.Ones([]int{8}), HasKV: true,
			QWq: syntheticSymQ4Weight(8, 8, 9), KWq: syntheticSymQ4Weight(8, 8, 10),
			OWq: syntheticSymQ4Weight(8, 8, 9), GateWq: syntheticSymQ4Weight(8, 8, 9), UpWq: syntheticSymQ4Weight(8, 8, 9), DownWq: syntheticSymQ4Weight(8, 8, 9),
		}},
	}
	hidden := []float32{1, -2, 3, -4, 5, -6, 7, -8}
	kvK, kvV := make([][]float32, 1), make([][]float32, 1)
	if got := m.ForwardLayer(append([]float32(nil), hidden...), 0, 0, 0, kvK, kvV); got == nil {
		t.Fatal("ForwardLayer rejected omitted-V QAT K=V path")
	}
	if len(kvK[0]) != 8 || len(kvV[0]) != 8 || !sameFloat32s(kvK[0], kvV[0]) {
		t.Fatalf("ForwardLayer QAT K=V omitted-V path K=%v V=%v", kvK[0], kvV[0])
	}
	m.Layers[0].VWq = m.Layers[0].KWq
	kvK, kvV = make([][]float32, 1), make([][]float32, 1)
	if got := m.ForwardLayer(append([]float32(nil), hidden...), 0, 0, 0, kvK, kvV); got == nil {
		t.Fatal("ForwardLayer rejected shared-pointer QAT K=V path")
	}
	if len(kvK[0]) != 8 || len(kvV[0]) != 8 || !sameFloat32s(kvK[0], kvV[0]) {
		t.Fatalf("ForwardLayer QAT K=V shared-pointer path K=%v V=%v", kvK[0], kvV[0])
	}
}

func TestForwardMTPPromptLayerQuantKEqV(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{VocabSize: 3, HiddenSize: 8, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 8, Intermediate: 8, AttentionKEqV: true, RMSNormEps: 1e-6},
		Layers: []LlamaLayer{{
			InputNorm: tensor.Ones([]int{8}), PostNorm: tensor.Ones([]int{8}), HasKV: true,
			QWq: syntheticSymQ4Weight(8, 8, 9), KWq: syntheticSymQ4Weight(8, 8, 10),
			OWq: syntheticSymQ4Weight(8, 8, 9), GateWq: syntheticSymQ4Weight(8, 8, 9), UpWq: syntheticSymQ4Weight(8, 8, 9), DownWq: syntheticSymQ4Weight(8, 8, 9),
		}},
	}
	hidden := []float32{1, -2, 3, -4, 5, -6, 7, -8}
	kvK, kvV := make([][]float32, 1), make([][]float32, 1)
	if _, err := m.forwardMTPPromptLayer(append([]float32(nil), hidden...), nil, 0, 0, kvK, kvV, make([]float32, 1), make([]float32, 8)); err != nil {
		t.Fatal(err)
	}
	if len(kvK[0]) != 8 || len(kvV[0]) != 8 || !sameFloat32s(kvK[0], kvV[0]) {
		t.Fatalf("prompt QAT K=V omitted-V path K=%v V=%v", kvK[0], kvV[0])
	}
	m.Layers[0].VWq = m.Layers[0].KWq
	kvK, kvV = make([][]float32, 1), make([][]float32, 1)
	if _, err := m.forwardMTPPromptLayer(append([]float32(nil), hidden...), nil, 0, 0, kvK, kvV, make([]float32, 1), make([]float32, 8)); err != nil {
		t.Fatal(err)
	}
	if len(kvK[0]) != 8 || len(kvV[0]) != 8 || !sameFloat32s(kvK[0], kvV[0]) {
		t.Fatalf("prompt QAT K=V shared-pointer path K=%v V=%v", kvK[0], kvV[0])
	}
}
