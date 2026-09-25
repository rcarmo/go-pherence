package qwen

import (
	"fmt"
	"math"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	llmops "github.com/rcarmo/go-pherence/model/internal/ops"
)

// ForwardTextBranch evaluates a fresh, bounded state/question/candidate branch.
// Full attention is bidirectional within each node and sees ancestors; linear
// attention is causal along this branch. Positions are explicit RoPE positions,
// not cache offsets. No recurrent or KV state survives the call. This reference
// path uses F32 arithmetic, not the BF16 trajectory of a released HF forward.
// It leaves the existing causal sequence and MTP APIs unchanged.
func (m *Qwen35BaseModel) ForwardTextBranch(inputs [][]float32, positions []int, stateLen, questionLen int, rope []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([][]float32, error) {
	n := len(inputs)
	if m == nil || n < 3 || n > 4096 || len(positions) != n || stateLen < 1 || stateLen >= n || questionLen < 1 || questionLen >= n-stateLen || meta.BF16Trajectory || meta.HiddenSize < 1 || eps <= 0 || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return nil, fmt.Errorf("qwen: invalid F32 text branch")
	}
	if meta.HiddenSize > 4096 || meta.IntermediateSize < 1 || meta.IntermediateSize > 32768 || meta.HeadDim < 1 || meta.HeadDim > 256 || meta.NumAttentionHeads < 1 || meta.NumAttentionHeads > 64 || meta.NumKeyValueHeads < 1 || meta.NumKeyValueHeads > meta.NumAttentionHeads || meta.NumAttentionHeads%meta.NumKeyValueHeads != 0 || meta.LinearNumKeyHeads < 0 || meta.LinearNumKeyHeads > 64 || meta.LinearNumValueHeads < 0 || meta.LinearNumValueHeads > 64 || meta.LinearKeyHeadDim < 0 || meta.LinearKeyHeadDim > 256 || meta.LinearValueHeadDim < 0 || meta.LinearValueHeadDim > 256 || meta.LinearConvKernelDim < 0 || meta.LinearConvKernelDim > 16 {
		return nil, fmt.Errorf("qwen: invalid branch dimensions")
	}
	cur := make([][]float32, n)
	for i, in := range inputs {
		if len(in) != meta.HiddenSize || positions[i] < 0 || positions[i] >= 4096 || (i > 0 && positions[i] <= positions[i-1]) {
			return nil, fmt.Errorf("qwen: invalid branch token %d", i)
		}
		for _, v := range in {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("qwen: nonfinite branch input")
			}
		}
		if len(rope) > 0 && len(rope) < (positions[i]+1)*Qwen35RotaryHalf(meta)*2 {
			return nil, fmt.Errorf("qwen: short branch RoPE table")
		}
		cur[i] = append([]float32(nil), in...)
	}
	if len(m.Layers) == 0 || len(m.Layers) != meta.MainLayerCount() {
		return nil, fmt.Errorf("qwen: branch layer count mismatch")
	}
	for i, layer := range m.Layers {
		var err error
		switch layer.Kind {
		case Qwen35LinearAttentionLayerKind:
			if err = ValidateQwen35LinearAttentionLayer(layer.Linear, meta, "branch"); err != nil {
				return nil, err
			}
			var state Qwen35LinearAttentionState
			state, err = NewQwen35LinearAttentionState(meta)
			if err != nil {
				return nil, err
			}
			for t := range cur {
				cur[t], err = layer.Linear.forwardWithStateMutating(cur[t], &state, eps, meta)
				if err != nil {
					return nil, fmt.Errorf("qwen: branch layer %d token %d: %w", i, t, err)
				}
			}
		case Qwen35FullAttentionLayerKind:
			cur, err = layer.Full.forwardFullTextBranch(cur, positions, stateLen, questionLen, rope, eps, meta)
		default:
			return nil, fmt.Errorf("qwen: unsupported branch layer %q", layer.Kind)
		}
		if err != nil {
			return nil, fmt.Errorf("qwen: branch layer %d: %w", i, err)
		}
	}
	for _, row := range cur {
		for _, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("qwen: nonfinite branch output")
			}
		}
	}
	return cur, nil
}

func (l *Qwen35FullAttentionLayer) forwardFullTextBranch(in [][]float32, positions []int, stateLen, questionLen int, rope []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([][]float32, error) {
	if err := ValidateQwen35FullAttentionLayer(l, meta, "branch"); err != nil {
		return nil, err
	}
	h, hd, nh, nkv := meta.HiddenSize, meta.HeadDim, meta.NumAttentionHeads, meta.NumKeyValueHeads
	if hd < 1 || nh < 1 || nkv < 1 || nh%nkv != 0 {
		return nil, fmt.Errorf("qwen: invalid branch heads")
	}
	n := len(in)
	qs, ks, vs, gates := make([][]float32, n), make([][]float32, n), make([][]float32, n), make([][]float32, n)
	for t, input := range in {
		norm := append([]float32(nil), input...)
		rmsNormQwen35InPlace(norm, l.InputNorm.Data(), eps, meta.ZeroCenteredRMSNorm)
		qg := make([]float32, 2*nh*hd)
		if err := qwen35LinearInto(qg, norm, l.QW, l.QWQ, l.QWm, h, len(qg), "q_proj"); err != nil {
			return nil, err
		}
		qs[t], gates[t], _ = splitQwen35FullQGate(qg, nh, hd)
		ks[t], vs[t] = make([]float32, nkv*hd), make([]float32, nkv*hd)
		if err := qwen35LinearInto(ks[t], norm, l.KW, l.KWQ, l.KWm, h, len(ks[t]), "k_proj"); err != nil {
			return nil, err
		}
		if err := qwen35LinearInto(vs[t], norm, l.VW, l.VWQ, l.VWm, h, len(vs[t]), "v_proj"); err != nil {
			return nil, err
		}
		// Unlike the legacy head-normalisation helper, do not round to BF16.
		for head := 0; head < nh; head++ {
			rmsNormQwen35InPlace(qs[t][head*hd:(head+1)*hd], l.QNorm.Data(), eps, meta.ZeroCenteredRMSNorm)
		}
		for head := 0; head < nkv; head++ {
			rmsNormQwen35InPlace(ks[t][head*hd:(head+1)*hd], l.KNorm.Data(), eps, meta.ZeroCenteredRMSNorm)
		}
		if len(rope) > 0 {
			llmops.ApplyRoPEPartial(qs[t], rope, positions[t], nh, hd, Qwen35RotaryHalf(meta))
			llmops.ApplyRoPEPartial(ks[t], rope, positions[t], nkv, hd, Qwen35RotaryHalf(meta))
		}
	}
	out := make([][]float32, n)
	for t, input := range in {
		// All visible keys form an ancestor prefix ending at this node's end.
		end := n
		if t < stateLen {
			end = stateLen
		} else if t < stateLen+questionLen {
			end = stateLen + questionLen
		}
		allK, allV := make([]float32, 0, end*nkv*hd), make([]float32, 0, end*nkv*hd)
		for j := 0; j < end; j++ {
			allK = append(allK, ks[j]...)
			allV = append(allV, vs[j]...)
		}
		attn := qwenMTPGroupedAttention(qs[t], allK, allV, nh, nkv, hd)
		for i := range attn {
			attn[i] *= sigmoid(gates[t][i])
		}
		resid := make([]float32, h)
		if err := qwen35LinearInto(resid, attn, l.OW, l.OWQ, l.OWm, len(attn), h, "o_proj"); err != nil {
			return nil, err
		}
		for i := range resid {
			resid[i] += input[i]
		}
		norm := append([]float32(nil), resid...)
		rmsNormQwen35InPlace(norm, l.PostNorm.Data(), eps, meta.ZeroCenteredRMSNorm)
		gate, up := make([]float32, meta.IntermediateSize), make([]float32, meta.IntermediateSize)
		if err := qwen35LinearInto(gate, norm, l.GateW, l.GateWQ, l.GateWm, h, len(gate), "mlp.gate"); err != nil {
			return nil, err
		}
		if err := qwen35LinearInto(up, norm, l.UpW, l.UpWQ, l.UpWm, h, len(up), "mlp.up"); err != nil {
			return nil, err
		}
		for i := range gate {
			gate[i] = gate[i] * sigmoid(gate[i]) * up[i]
		}
		out[t] = make([]float32, h)
		if err := qwen35LinearInto(out[t], gate, l.DownW, l.DownWQ, l.DownWm, len(gate), h, "mlp.down"); err != nil {
			return nil, err
		}
		for i := range resid {
			out[t][i] += resid[i]
		}
	}
	return out, nil
}
