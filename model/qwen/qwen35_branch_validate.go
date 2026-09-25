package qwen

import (
	"fmt"
	"math"

	cfg "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/tensor"
)

// ValidateQwen35F32Branch checks the fixed geometry and dense finite weights
// consumed by the accelerated 0.8B branch executors, before packing or upload.
// The caller must keep these weights immutable for the executor's lifetime.
func ValidateQwen35F32Branch(m *Qwen35BaseModel, meta cfg.QwenNativeMTPMetadata, maxTokens int) error {
	if m == nil || len(m.Layers) != 24 || len(m.Layers) != meta.MainLayerCount() || meta.BF16Trajectory || meta.QuantBits != 0 || !meta.ZeroCenteredRMSNorm || meta.HiddenSize != 1024 || meta.IntermediateSize != 3584 || meta.NumAttentionHeads != 8 || meta.NumKeyValueHeads != 2 || meta.HeadDim != 256 || meta.LinearNumKeyHeads != 16 || meta.LinearNumValueHeads != 16 || meta.LinearKeyHeadDim != 128 || meta.LinearValueHeadDim != 128 || meta.LinearConvKernelDim != 4 || meta.PartialRotaryFactor != 0.25 || maxTokens < 3 || maxTokens > 512 {
		return fmt.Errorf("qwen: unsupported accelerated branch configuration")
	}
	for i, layer := range m.Layers {
		want := Qwen35LinearAttentionLayerKind
		if (i+1)%4 == 0 {
			want = Qwen35FullAttentionLayerKind
		}
		if layer.Kind != want {
			return fmt.Errorf("qwen: unexpected accelerated layer %d", i)
		}
		var tensors []*tensor.Tensor
		if layer.Kind == Qwen35LinearAttentionLayerKind {
			l := layer.Linear
			if err := ValidateQwen35LinearAttentionLayer(l, meta, "accelerated"); err != nil {
				return err
			}
			tensors = []*tensor.Tensor{l.InputNorm, l.PostNorm, l.QKVW, l.GateW, l.Conv1D, l.AlphaW, l.BetaW, l.A, l.DTBias, l.Norm, l.OutW, l.MLPGateW, l.MLPUpW, l.MLPDownW}
		} else {
			l := layer.Full
			if err := ValidateQwen35FullAttentionLayer(l, meta, "accelerated"); err != nil {
				return err
			}
			tensors = []*tensor.Tensor{l.InputNorm, l.PostNorm, l.QW, l.KW, l.VW, l.OW, l.QNorm, l.KNorm, l.GateW, l.UpW, l.DownW}
		}
		for _, t := range tensors {
			if t == nil || t.DType() != tensor.Float32 || len(t.Data()) != t.Numel() {
				return fmt.Errorf("qwen: layer %d requires dense F32 weights", i)
			}
			for _, v := range t.Data() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return fmt.Errorf("qwen: nonfinite layer %d weight", i)
				}
			}
		}
	}
	return nil
}
