package model

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"strings"

	"github.com/rcarmo/go-pherence/backends/mlx"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/tensor"
)

// RunMTPVerifierBatchForward is the verifier-batch execution entry point. It
// consumes materialized MTPVerifierBatchInputs for all verifier rows. The current
// lowering executes rows/layers sequentially while preserving the explicit batch
// and attention-plan contract; future SIMD/GPU work can replace the inner layer
// loop without changing callers.
func (m *LlamaModel) RunMTPVerifierBatchForward(batch MTPVerifierBatchInputs, kvCacheK, kvCacheV [][]float32) (MTPVerifierResult, error) {
	if err := m.validateMTPVerifierBatchForwardInputs(batch, kvCacheK, kvCacheV); err != nil {
		return MTPVerifierResult{}, err
	}
	finalHiddenRows, ok, err := m.runMTPVerifierBatchLayers(batch, kvCacheK, kvCacheV)
	if err != nil {
		return MTPVerifierResult{}, err
	}
	if !ok {
		finalHiddenRows, err = m.runMTPVerifierBatchRowsSequential(batch, kvCacheK, kvCacheV)
		if err != nil {
			return MTPVerifierResult{}, err
		}
	}
	finalActivations, logitsRows, _, err := m.FinishCPUDecodeBatch(finalHiddenRows)
	if err != nil {
		return MTPVerifierResult{}, fmt.Errorf("verifier batch decode finish: %w", err)
	}
	return NewMTPVerifierResultRowsForModel(m, batch.Plan.InputToken, batch.Plan.DraftedTokens, logitsRows, finalActivations)
}

func (m *LlamaModel) runMTPVerifierBatchRowsSequential(batch MTPVerifierBatchInputs, kvCacheK, kvCacheV [][]float32) ([][]float32, error) {
	finalHiddenRows := make([][]float32, len(batch.HiddenRows))
	attnRows := batch.Scratch.MaxAttentionRows
	if attnRows < 1 {
		attnRows = 1
	}
	attnOutWidth := batch.Scratch.MaxQDim
	if attnOutWidth < 1 {
		attnOutWidth = m.Config.NumHeads * m.Config.HeadDim
	}
	attnScoresScratch := make([]float32, attnRows)
	attnOutScratch := make([]float32, attnOutWidth)
	for i, row := range batch.HiddenRows {
		hidden := append([]float32(nil), row...)
		pos := batch.Plan.Positions[i]
		perLayerInputs := batch.PerLayerInputs[i]
		for l := 0; l < m.Config.NumLayers; l++ {
			if perLayerInputs != nil {
				var err error
				hidden, err = m.forwardMTPPromptLayer(hidden, perLayerInputs, l, pos, kvCacheK, kvCacheV, attnScoresScratch, attnOutScratch)
				if err != nil {
					return nil, fmt.Errorf("verifier batch forward layer %d at position %d: %w", l, pos, err)
				}
				continue
			}
			hidden = m.ForwardLayer(hidden, l, pos, pos, kvCacheK, kvCacheV)
			if hidden == nil {
				return nil, fmt.Errorf("verifier batch forward layer %d at position %d failed", l, pos)
			}
		}
		finalHiddenRows[i] = hidden
	}
	return finalHiddenRows, nil
}

func (m *LlamaModel) runMTPVerifierBatchLayers(batch MTPVerifierBatchInputs, kvCacheK, kvCacheV [][]float32) ([][]float32, bool, error) {
	if !m.mtpVerifierBatchLayerEligible(batch) {
		return nil, false, nil
	}
	B, h := batch.Scratch.Batch, batch.Scratch.HiddenSize
	bHidden := append([]float32(nil), batch.HiddenFlat...)
	bResidual := make([]float32, len(bHidden))
	bAttnOut := make([]float32, B*batch.Scratch.MaxQDim)
	bOOut := make([]float32, B*h)
	bMlpIn := make([]float32, B*h)
	bGate := make([]float32, B*batch.Scratch.MaxIntermediate)
	bUp := make([]float32, B*batch.Scratch.MaxIntermediate)
	bDown := make([]float32, B*h)
	var bPLIGate, bPLIProj []float32
	if batch.HasPerLayerInputs && m.Config.HiddenPerLayer > 0 {
		bPLIGate = make([]float32, B*m.Config.HiddenPerLayer)
		bPLIProj = make([]float32, B*h)
	}
	attnScores := make([]float32, batch.Scratch.MaxAttentionRows)
	isGemma := m.Config.ModelType == "gemma3_text" || m.Config.ModelType == "gemma4_text"
	eps := float32(m.Config.RMSNormEps)
	for l := 0; l < m.Config.NumLayers; l++ {
		layer := &m.Layers[l]
		copy(bResidual, bHidden)
		qkv, err := m.ProjectMTPVerifierLayerQKVBatch(batch, l, bHidden)
		if err != nil {
			return nil, true, err
		}
		if qkv.HasKV {
			for b := 0; b < B; b++ {
				kvCacheK[l] = append(kvCacheK[l], qkv.K[b*qkv.KVDim:(b+1)*qkv.KVDim]...)
				kvCacheV[l] = append(kvCacheV[l], qkv.V[b*qkv.KVDim:(b+1)*qkv.KVDim]...)
			}
		}
		lp := batch.Attention.Layers[l]
		for b := 0; b < B; b++ {
			start, end := lp.KVStart[b], lp.KVEndExclusive[b]
			attnSeqLen := end - start
			if attnSeqLen <= 0 {
				return nil, true, fmt.Errorf("verifier batch layer %d row %d invalid attention range [%d,%d)", l, b, start, end)
			}
			kOff, kEnd := start*qkv.KVDim, end*qkv.KVDim
			if kEnd > len(kvCacheK[l]) || kEnd > len(kvCacheV[l]) {
				return nil, true, fmt.Errorf("verifier batch layer %d row %d KV range [%d,%d) exceeds K/V=%d/%d", l, b, kOff, kEnd, len(kvCacheK[l]), len(kvCacheV[l]))
			}
			scale := float32(1.0 / math.Sqrt(float64(qkv.HeadDim)))
			if m.Config.ModelType == "gemma4_text" {
				scale = 1.0
			}
			gqaAttentionScaleInto(bAttnOut[b*qkv.QDim:(b+1)*qkv.QDim], attnScores[:attnSeqLen], qkv.Q[b*qkv.QDim:(b+1)*qkv.QDim], kvCacheK[l][kOff:kEnd], kvCacheV[l][kOff:kEnd], attnSeqLen, m.Config.NumHeads, qkv.KVHeads, qkv.HeadDim, scale)
		}
		if !m.projBatchAny(bOOut[:B*h], bAttnOut[:B*qkv.QDim], B, layer.OW, layer.OWm, layer.OWq, qkv.QDim, h) {
			return nil, true, fmt.Errorf("verifier batch layer %d O projection rejected", l)
		}
		if layer.PreFFNNorm != nil {
			for b := 0; b < B; b++ {
				o := bOOut[b*h : (b+1)*h]
				rmsNormInPlace(o, layer.PostNorm.Data(), eps)
				hid := bHidden[b*h : (b+1)*h]
				simd.VecAdd(hid, bResidual[b*h:(b+1)*h], o)
			}
			copy(bResidual, bHidden)
			for b := 0; b < B; b++ {
				in := bMlpIn[b*h : (b+1)*h]
				copy(in, bHidden[b*h:(b+1)*h])
				if isGemma {
					simd.RMSNormBF16(in, layer.PreFFNNorm.Data(), eps)
				} else {
					rmsNormInPlace(in, layer.PreFFNNorm.Data(), eps)
				}
			}
		} else {
			for b := 0; b < B; b++ {
				hid := bHidden[b*h : (b+1)*h]
				simd.VecAdd(hid, bResidual[b*h:(b+1)*h], bOOut[b*h:(b+1)*h])
			}
			copy(bResidual, bHidden)
			for b := 0; b < B; b++ {
				rmsNormInPlace(bHidden[b*h:(b+1)*h], layer.PostNorm.Data(), eps)
			}
			copy(bMlpIn, bHidden)
		}
		inter := m.layerInterFor(layer)
		if !m.projBatchAny(bGate[:B*inter], bMlpIn[:B*h], B, layer.GateW, layer.GateWm, layer.GateWq, h, inter) || !m.projBatchAny(bUp[:B*inter], bMlpIn[:B*h], B, layer.UpW, layer.UpWm, layer.UpWq, h, inter) {
			return nil, true, fmt.Errorf("verifier batch layer %d MLP gate/up rejected", l)
		}
		for b := 0; b < B; b++ {
			gate := bGate[b*inter : (b+1)*inter]
			up := bUp[b*inter : (b+1)*inter]
			if isGemma {
				simd.ToBF16(gate)
				simd.ToBF16(up)
			}
			if m.Config.HiddenAct == "gelu_pytorch_tanh" {
				simd.GELUTanhMul(gate, gate, up)
				if isGemma {
					simd.ToBF16(gate)
				}
			} else {
				simd.VecSiLUMul(gate, gate, up)
			}
		}
		if !m.projBatchAny(bDown[:B*h], bGate[:B*inter], B, layer.DownW, layer.DownWm, layer.DownWq, inter, h) {
			return nil, true, fmt.Errorf("verifier batch layer %d MLP down rejected", l)
		}
		for b := 0; b < B; b++ {
			down := bDown[b*h : (b+1)*h]
			if isGemma {
				simd.ToBF16(down)
			}
			if layer.PostFFNNorm != nil {
				if isGemma {
					rmsNormBF16(down, layer.PostFFNNorm.Data(), eps)
				} else {
					rmsNormInPlace(down, layer.PostFFNNorm.Data(), eps)
				}
			}
			hid := bHidden[b*h : (b+1)*h]
			simd.VecAdd(hid, bResidual[b*h:(b+1)*h], down)
		}
		if batch.HasPerLayerInputs && layer.PLIGate != nil {
			hpl := m.Config.HiddenPerLayer
			if hpl <= 0 || len(layer.PLIGate) < hpl*h || len(layer.PLIProj) < h*hpl || len(layer.PLIPostNorm) < h || len(bPLIGate) < B*hpl || len(bPLIProj) < B*h {
				return nil, true, fmt.Errorf("verifier batch layer %d invalid PLI dims", l)
			}
			if !simd.GemmRowsParallel(bPLIGate[:B*hpl], bHidden[:B*h], layer.PLIGate, B, hpl, h) {
				return nil, true, fmt.Errorf("verifier batch layer %d PLI gate rejected", l)
			}
			for b := 0; b < B; b++ {
				gate := bPLIGate[b*hpl : (b+1)*hpl]
				if l >= len(batch.PerLayerInputs[b]) || len(batch.PerLayerInputs[b][l]) < hpl {
					return nil, true, fmt.Errorf("verifier batch layer %d row %d missing PLI row", l, b)
				}
				simd.GELUTanhMul(gate, gate, batch.PerLayerInputs[b][l][:hpl])
			}
			if !simd.GemmRowsParallel(bPLIProj[:B*h], bPLIGate[:B*hpl], layer.PLIProj, B, h, hpl) {
				return nil, true, fmt.Errorf("verifier batch layer %d PLI projection rejected", l)
			}
			for b := 0; b < B; b++ {
				proj := bPLIProj[b*h : (b+1)*h]
				rmsNormInPlace(proj, layer.PLIPostNorm, eps)
				hid := bHidden[b*h : (b+1)*h]
				simd.VecAdd(hid, hid, proj)
			}
		}
		for b := 0; b < B; b++ {
			hid := bHidden[b*h : (b+1)*h]
			if layer.LayerScalar != 1.0 {
				simd.VecScale(hid, hid, layer.LayerScalar)
			}
			if isGemma {
				simd.ToBF16(hid)
			}
		}
	}
	out := make([][]float32, B)
	for b := 0; b < B; b++ {
		out[b] = append([]float32(nil), bHidden[b*h:(b+1)*h]...)
	}
	return out, true, nil
}

func (m *LlamaModel) projBatchAny(out, x []float32, B int, dense *tensor.Tensor, mlxw *mlx.QuantWeight, qw *QuantWeight, inDim, outDim int) bool {
	if qw != nil {
		if B <= 0 || inDim <= 0 || outDim <= 0 || len(out) < B*outDim || len(x) < B*inDim || qw.InDim != inDim || qw.OutDim != outDim {
			return false
		}
		for b := 0; b < B; b++ {
			m.mvQ(out[b*outDim:(b+1)*outDim], x[b*inDim:(b+1)*inDim], qw)
		}
		return true
	}
	return m.projBatch(out, x, B, dense, mlxw, inDim, outDim)
}

func hasMTPVerifierProjection(dense *tensor.Tensor, mlxw *mlx.QuantWeight, qw *QuantWeight) bool {
	return dense != nil || mlxw != nil || qw != nil
}

func (m *LlamaModel) mtpVerifierBatchLayerEligible(batch MTPVerifierBatchInputs) bool {
	// Keep full-layer batch lowering behind a correctness gate until the row-loop
	// parity suite is complete. The QKV projection primitive is tested and ready,
	// but the complete attention+MLP layer replacement must not silently change
	// verifier acceptance.
	if !mtpVerifierBatchLayerLoweringEnabled() || m == nil || m.Config.NumLayers <= 0 || m.Config.HiddenSize <= 0 || m.Config.NumHeads <= 0 || m.Config.HeadDim <= 0 {
		return false
	}
	for l := 0; l < m.Config.NumLayers; l++ {
		layer := &m.Layers[l]
		if !layer.HasKV || layer.IsMoE {
			return false
		}
		if !hasMTPVerifierProjection(layer.QW, layer.QWm, layer.QWq) || !hasMTPVerifierProjection(layer.KW, layer.KWm, layer.KWq) || !hasMTPVerifierProjection(layer.OW, layer.OWm, layer.OWq) || !hasMTPVerifierProjection(layer.GateW, layer.GateWm, layer.GateWq) || !hasMTPVerifierProjection(layer.UpW, layer.UpWm, layer.UpWq) || !hasMTPVerifierProjection(layer.DownW, layer.DownWm, layer.DownWq) {
			return false
		}
		if !(m.Config.AttentionKEqV && ((layer.KW != nil && (layer.VW == nil || layer.VW == layer.KW)) || (layer.KWm != nil && (layer.VWm == nil || layer.VWm == layer.KWm)) || (layer.KWq != nil && (layer.VWq == nil || layer.VWq == layer.KWq)))) && !hasMTPVerifierProjection(layer.VW, layer.VWm, layer.VWq) {
			return false
		}
		if layer.InputNorm == nil || layer.PostNorm == nil {
			return false
		}
	}
	return true
}

func mtpVerifierBatchLayerLoweringEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func (m *LlamaModel) validateMTPVerifierBatchForwardInputs(batch MTPVerifierBatchInputs, kvCacheK, kvCacheV [][]float32) error {
	if err := validateMTPVerifierPlanForModel(m, batch.Plan); err != nil {
		return err
	}
	if len(batch.HiddenRows) != len(batch.Plan.VerifierTokens) {
		return fmt.Errorf("MTP verifier batch hidden rows=%d, want verifier tokens=%d", len(batch.HiddenRows), len(batch.Plan.VerifierTokens))
	}
	B := len(batch.Plan.VerifierTokens)
	h := m.Config.HiddenSize
	wantHiddenFlat, ok := checkedProduct(B, h)
	if !ok {
		return fmt.Errorf("MTP verifier batch hidden flat size overflow")
	}
	if len(batch.HiddenFlat) != wantHiddenFlat {
		return fmt.Errorf("MTP verifier batch hidden flat len=%d, want %d", len(batch.HiddenFlat), wantHiddenFlat)
	}
	for i, row := range batch.HiddenRows {
		if len(row) != h {
			return fmt.Errorf("MTP verifier batch hidden row %d len=%d, want %d", i, len(row), h)
		}
		flat := batch.HiddenFlat[i*h : (i+1)*h]
		for j, v := range row {
			if v != flat[j] {
				return fmt.Errorf("MTP verifier batch hidden row %d differs from flat buffer at col %d", i, j)
			}
		}
	}
	if err := validateMTPVerifierBatchPLIFlat(m, batch); err != nil {
		return err
	}
	if err := batch.Attention.ValidateAgainst(batch.Plan, m); err != nil {
		return err
	}
	wantScratch, err := NewMTPVerifierBatchScratchPlan(m, batch)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(batch.Scratch, wantScratch) {
		return fmt.Errorf("MTP verifier batch scratch plan does not match model/batch shapes")
	}
	if len(kvCacheK) != m.Config.NumLayers || len(kvCacheV) != m.Config.NumLayers {
		return fmt.Errorf("KV cache layers K/V=%d/%d, want %d", len(kvCacheK), len(kvCacheV), m.Config.NumLayers)
	}
	for l := 0; l < m.Config.NumLayers; l++ {
		layer := &m.Layers[l]
		kvDim, err := m.LayerKVDim(l)
		if err != nil {
			return err
		}
		if kvDim == 0 {
			if layer.HasKV {
				return fmt.Errorf("layer %d has invalid zero KV dim", l)
			}
			if layer.KVSourceLayer < 0 || layer.KVSourceLayer >= m.Config.NumLayers {
				return fmt.Errorf("shared-KV layer %d source %d out of range [0,%d)", l, layer.KVSourceLayer, m.Config.NumLayers)
			}
			sourceDim, err := m.LayerKVDim(layer.KVSourceLayer)
			if err != nil {
				return err
			}
			if sourceDim == 0 {
				return fmt.Errorf("shared-KV layer %d source %d does not append KV", l, layer.KVSourceLayer)
			}
			if len(kvCacheK[l]) != 0 || len(kvCacheV[l]) != 0 {
				return fmt.Errorf("shared/non-KV layer %d owns unexpected K/V cache entries %d/%d", l, len(kvCacheK[l]), len(kvCacheV[l]))
			}
			continue
		}
		want, ok := checkedProduct(batch.Plan.StartPos, kvDim)
		if !ok {
			return fmt.Errorf("verifier KV history length overflows for layer %d start=%d kvDim=%d", l, batch.Plan.StartPos, kvDim)
		}
		if len(kvCacheK[l]) != want || len(kvCacheV[l]) != want {
			return fmt.Errorf("layer %d KV history K/V=%d/%d, want %d for start position %d", l, len(kvCacheK[l]), len(kvCacheV[l]), want, batch.Plan.StartPos)
		}
	}
	return nil
}

func validateMTPVerifierBatchPLIFlat(m *LlamaModel, batch MTPVerifierBatchInputs) error {
	B := len(batch.Plan.VerifierTokens)
	if !batch.HasPerLayerInputs {
		if len(batch.PerLayerInputFlat) != 0 {
			return fmt.Errorf("MTP verifier batch has PLI flat len=%d while PLI is disabled", len(batch.PerLayerInputFlat))
		}
		for i, rows := range batch.PerLayerInputs {
			if len(rows) != 0 {
				return fmt.Errorf("MTP verifier batch has PLI rows for token %d while PLI is disabled", i)
			}
		}
		return nil
	}
	if len(batch.PerLayerInputs) != B {
		return fmt.Errorf("MTP verifier batch PLI rows=%d, want verifier tokens=%d", len(batch.PerLayerInputs), B)
	}
	hpl := m.Config.HiddenPerLayer
	if hpl <= 0 {
		return fmt.Errorf("MTP verifier batch PLI hidden width=%d", hpl)
	}
	perToken, ok := checkedProduct(m.Config.NumLayers, hpl)
	if !ok {
		return fmt.Errorf("MTP verifier batch PLI per-token size overflow")
	}
	wantFlat, ok := checkedProduct(B, perToken)
	if !ok {
		return fmt.Errorf("MTP verifier batch PLI flat size overflow")
	}
	if len(batch.PerLayerInputFlat) != wantFlat {
		return fmt.Errorf("MTP verifier batch PLI flat len=%d, want %d", len(batch.PerLayerInputFlat), wantFlat)
	}
	for i, rows := range batch.PerLayerInputs {
		if len(rows) != m.Config.NumLayers {
			return fmt.Errorf("MTP verifier batch PLI token %d layers=%d, want %d", i, len(rows), m.Config.NumLayers)
		}
		for l, row := range rows {
			if len(row) != hpl {
				return fmt.Errorf("MTP verifier batch PLI token %d layer %d len=%d, want %d", i, l, len(row), hpl)
			}
			flat := batch.PerLayerInputFlat[i*perToken+l*hpl : i*perToken+(l+1)*hpl]
			for j, v := range row {
				if v != flat[j] {
					return fmt.Errorf("MTP verifier batch PLI token %d layer %d differs from flat buffer at col %d", i, l, j)
				}
			}
		}
	}
	return nil
}
