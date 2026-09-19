package model

import (
	"math"
	"math/rand"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
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
		// Dense batch GEMM and singleton GEMV use different reduction/FMA
		// orders. This API promises numerical parity, not bitwise identity;
		// keep exact comparisons in the quantised trajectory fixtures.
		if !mtpDenseClose(got.Q[b*2:(b+1)*2], single.Q) || !mtpDenseClose(got.K[b*2:(b+1)*2], single.K) || !mtpDenseClose(got.V[b*2:(b+1)*2], single.V) {
			t.Fatalf("row %d batch q/k/v=%v/%v/%v single=%v/%v/%v", b, got.Q[b*2:(b+1)*2], got.K[b*2:(b+1)*2], got.V[b*2:(b+1)*2], single.Q, single.K, single.V)
		}
	}
}

func mtpDenseClose(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		av, bv := float64(a[i]), float64(b[i])
		if math.IsNaN(av) || math.IsNaN(bv) || math.IsInf(av, 0) || math.IsInf(bv, 0) || math.Abs(av-bv) > 1e-6*(1+math.Max(math.Abs(av), math.Abs(bv))) {
			return false
		}
	}
	return true
}

func TestMTPDenseProjectionReductionBound(t *testing.T) {
	// Exercise dot/GEMV and tiled GEMM shapes against a float64 sum. Bound
	// rounding by the absolute products, so cancellation cannot hide errors.
	rng := rand.New(rand.NewSource(5))
	for _, width := range []int{2, 8, 64, 128} {
		const rows, batch = 64, 3
		w, x := make([]float32, rows*width), make([]float32, batch*width)
		for i := range w {
			w[i] = rng.Float32()*2 - 1
		}
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		m := &LlamaModel{Large: true}
		weight := tensor.FromFloat32(w, []int{rows, width})
		out := make([]float32, batch*rows)
		if !m.projBatch(out, x, batch, weight, nil, width, rows) {
			t.Fatal("batch rejected")
		}
		for b := 0; b < batch; b++ {
			single := make([]float32, rows)
			if !m.projBatch(single, x[b*width:(b+1)*width], 1, weight, nil, width, rows) {
				t.Fatal("row rejected")
			}
			for r := 0; r < rows; r++ {
				var want, absProducts float64
				for k := 0; k < width; k++ {
					p := float64(x[b*width+k]) * float64(w[r*width+k])
					want += p
					absProducts += math.Abs(p)
				}
				bound := float64(width)*math.Ldexp(1, -23)*absProducts + 1e-7
				for _, got := range []float32{out[b*rows+r], single[r]} {
					if math.IsNaN(float64(got)) || math.Abs(float64(got)-want) > bound {
						t.Fatalf("width=%d row=%d got=%g want=%g bound=%g", width, r, got, want, bound)
					}
				}
			}
		}
	}
}

func TestMTPDenseCloseRejectsDrift(t *testing.T) {
	for _, bad := range []float32{1.01, float32(math.NaN()), float32(math.Inf(1))} {
		if mtpDenseClose([]float32{1}, []float32{bad}) {
			t.Fatalf("accepted %g", bad)
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
	if len(got.K) != len(got.V) || len(got.K) != len(batch.HiddenFlat) {
		t.Fatalf("K/V lengths K=%d V=%d hidden=%d", len(got.K), len(got.V), len(batch.HiddenFlat))
	}
	wantK := make([]float32, len(got.K))
	wantV := make([]float32, len(got.V))
	for b := range batch.HiddenRows {
		proj := make([]float32, got.KVDim)
		m.mv(proj, got.NormedIn[b*m.Config.HiddenSize:(b+1)*m.Config.HiddenSize], m.Layers[0].KW.Data(), m.Config.HiddenSize, got.KVDim)
		copy(wantK[b*got.KVDim:(b+1)*got.KVDim], proj)
		copy(wantV[b*got.KVDim:(b+1)*got.KVDim], proj)
		kRow := wantK[b*got.KVDim : (b+1)*got.KVDim]
		rmsNormInPlace(kRow, m.Layers[0].KNorm.Data(), float32(m.Config.RMSNormEps))
		simd.RMSNormNoScale(wantV[b*got.KVDim:(b+1)*got.KVDim], float32(m.Config.RMSNormEps))
		freqs, rotHalf := m.ensureGemma4RoPE(0, plan.Positions[b])
		applyRoPEPartial(kRow, freqs, plan.Positions[b], got.KVHeads, got.HeadDim, rotHalf)
	}
	if !sameFloat32s(got.K, wantK) || !sameFloat32s(got.V, wantV) {
		t.Fatalf("Gemma4 K=V post-processing K/V=%v/%v, want K-norm/no-scale-V %v/%v", got.K, got.V, wantK, wantV)
	}
	if sameFloat32s(got.K, got.V) {
		t.Fatalf("Gemma4 V should be no-scale normalized from the K projection, not aliased after K norm; got identical K/V %v", got.K)
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
	bad = *m
	bad.Layers = append([]LlamaLayer(nil), m.Layers...)
	bad.Layers[0].QNorm = tensor.Ones([]int{m.Config.HeadDim})
	bad.Layers[0].KNorm = nil
	if _, err := (&bad).ProjectMTPVerifierLayerQKVBatch(batch, 0, batch.HiddenFlat); err == nil {
		t.Fatal("accepted QNorm without required KNorm")
	}
}
