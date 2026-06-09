package whisper

import "math"

// FFNDiagnostic reports output drift for one FFN variant at one encoder layer.
type FFNDiagnostic struct {
	Layer    int
	Variant  string
	SeqLen   int
	DModel   int
	FFNDim   int
	MaxAbs   float64
	MeanAbs  float64
	RMSE     float64
	RelRMSE  float64
	Cosine   float64
	Compared int
	OK       bool
}

// DiagnoseFFN compares per-layer FFN outputs against the normal native path.
// The encoder state advances with the baseline output so all variants for a
// layer see the same pre-MLP input and residual. maxLayers <= 0 means all layers.
func (enc *Encoder) DiagnoseFFN(mel []float32, T, maxLayers int) []FFNDiagnostic {
	cfg := enc.cfg
	dModel := cfg.EncoderDModel
	ffnDim := cfg.EncoderFFNDim

	h := conv1dForwardFast(mel, enc.Conv1Weight, enc.Conv1Bias, cfg.NumMelBins, T, dModel, 3, 1, 1)
	gelu(h)
	h = conv1dForwardFast(h, enc.Conv2Weight, enc.Conv2Bias, dModel, T, dModel, 3, 2, 1)
	gelu(h)
	T2 := T / 2
	ht := transpose2D(h, dModel, T2)
	for t := 0; t < T2 && t < cfg.MaxLength; t++ {
		for d := 0; d < dModel; d++ {
			ht[t*dModel+d] += enc.PosEmbed[t*dModel+d]
		}
	}

	layers := len(enc.Layers)
	if maxLayers > 0 && maxLayers < layers {
		layers = maxLayers
	}
	var out []FFNDiagnostic
	resetA100Timers()
	for i := 0; i < layers; i++ {
		layer := &enc.Layers[i]
		attnIn := layerNorm(ht, layer.AttnLNWeight, layer.AttnLNBias, T2, dModel)
		q := linearForwardOpt(attnIn, layer.QWeight, layer.QBias, T2, dModel, dModel)
		k := linearForwardOpt(attnIn, layer.KWeight, layer.KBias, T2, dModel, dModel)
		v := linearForwardOpt(attnIn, layer.VWeight, layer.VBias, T2, dModel, dModel)
		attnOut := fullAttention(q, k, v, T2, T2, cfg.EncoderHeads, cfg.HeadDim)
		projected := linearForwardOpt(attnOut, layer.OWeight, layer.OBias, T2, dModel, dModel)
		for j := range projected {
			projected[j] += ht[j]
		}
		mlpIn := layerNorm(projected, layer.MLPLNWeight, layer.MLPLNBias, T2, dModel)

		baseline := ffnBaseline(mlpIn, layer, projected, T2, dModel, ffnDim)
		if fc1, ok := ffnA100FC1NativeFC2(mlpIn, layer, projected, T2, dModel, ffnDim); ok {
			out = append(out, compareFFN(i, "a100_fc1_native_fc2", baseline, fc1, T2, dModel, ffnDim))
		} else {
			out = append(out, FFNDiagnostic{Layer: i, Variant: "a100_fc1_native_fc2", SeqLen: T2, DModel: dModel, FFNDim: ffnDim, OK: false})
		}
		if fused, ok := forwardA100FFNFusedRaw(mlpIn, layer, projected, T2, dModel, ffnDim); ok {
			out = append(out, compareFFN(i, "a100_fused_ffn", baseline, fused, T2, dModel, ffnDim))
		} else {
			out = append(out, FFNDiagnostic{Layer: i, Variant: "a100_fused_ffn", SeqLen: T2, DModel: dModel, FFNDim: ffnDim, OK: false})
		}
		ht = baseline
	}
	return out
}

func ffnBaseline(mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) []float32 {
	hidden := linearForwardOpt(mlpIn, layer.FC1Weight, layer.FC1Bias, seqLen, dModel, ffnDim)
	gelu(hidden)
	out := linearForwardOpt(hidden, layer.FC2Weight, layer.FC2Bias, seqLen, ffnDim, dModel)
	for i := range residual {
		out[i] += residual[i]
	}
	return out
}

func ffnA100FC1NativeFC2(mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	return forwardA100FFNFC1NativeFC2Raw(mlpIn, layer, residual, seqLen, dModel, ffnDim)
}

func compareFFN(layer int, variant string, baseline, candidate []float32, seqLen, dModel, ffnDim int) FFNDiagnostic {
	n := len(baseline)
	if len(candidate) < n {
		n = len(candidate)
	}
	var maxAbs, sumAbs, sumSq, baseSq, candSq, dot float64
	for i := 0; i < n; i++ {
		b := float64(baseline[i])
		c := float64(candidate[i])
		d := c - b
		ad := math.Abs(d)
		if ad > maxAbs {
			maxAbs = ad
		}
		sumAbs += ad
		sumSq += d * d
		baseSq += b * b
		candSq += c * c
		dot += b * c
	}
	meanAbs, rmse, rel, cos := math.NaN(), math.NaN(), math.NaN(), math.NaN()
	if n > 0 {
		meanAbs = sumAbs / float64(n)
		rmse = math.Sqrt(sumSq / float64(n))
		if baseSq > 0 {
			rel = math.Sqrt(sumSq / baseSq)
		}
		if baseSq > 0 && candSq > 0 {
			cos = dot / math.Sqrt(baseSq*candSq)
		}
	}
	return FFNDiagnostic{Layer: layer, Variant: variant, SeqLen: seqLen, DModel: dModel, FFNDim: ffnDim, MaxAbs: maxAbs, MeanAbs: meanAbs, RMSE: rmse, RelRMSE: rel, Cosine: cos, Compared: n, OK: len(baseline) == len(candidate)}
}
