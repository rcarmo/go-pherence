package diffusiongemma

import "fmt"

func cloneEncoderKVLayer(layer EncoderKVLayer) EncoderKVLayer {
	out := layer
	out.Keys = append([]float32(nil), layer.Keys...)
	out.Values = append([]float32(nil), layer.Values...)
	return out
}

func appendEncoderKVLayer(prefix EncoderKVLayer, suffix EncoderKVLayer) (EncoderKVLayer, error) {
	if suffix.SeqLen == 0 {
		return cloneEncoderKVLayer(prefix), nil
	}
	if prefix.SeqLen == 0 {
		return cloneEncoderKVLayer(suffix), nil
	}
	if prefix.KVHeads != suffix.KVHeads || prefix.HeadDim != suffix.HeadDim {
		return EncoderKVLayer{}, fmt.Errorf("encoder KV append shape mismatch prefix=[heads=%d dim=%d] suffix=[heads=%d dim=%d]", prefix.KVHeads, prefix.HeadDim, suffix.KVHeads, suffix.HeadDim)
	}
	if prefix.KVHeads <= 0 || prefix.HeadDim <= 0 {
		return EncoderKVLayer{}, fmt.Errorf("encoder KV append invalid shape heads=%d dim=%d", prefix.KVHeads, prefix.HeadDim)
	}
	row := prefix.KVHeads * prefix.HeadDim
	prefixNeed := prefix.SeqLen * row
	suffixNeed := suffix.SeqLen * row
	if len(prefix.Keys) < prefixNeed || len(prefix.Values) < prefixNeed || len(suffix.Keys) < suffixNeed || len(suffix.Values) < suffixNeed {
		return EncoderKVLayer{}, fmt.Errorf("encoder KV append short buffers prefix_seq=%d suffix_seq=%d row=%d", prefix.SeqLen, suffix.SeqLen, row)
	}
	out := EncoderKVLayer{SeqLen: prefix.SeqLen + suffix.SeqLen, KVHeads: prefix.KVHeads, HeadDim: prefix.HeadDim}
	out.Keys = make([]float32, (prefix.SeqLen+suffix.SeqLen)*row)
	out.Values = make([]float32, len(out.Keys))
	copy(out.Keys, prefix.Keys[:prefixNeed])
	copy(out.Keys[prefixNeed:], suffix.Keys[:suffixNeed])
	copy(out.Values, prefix.Values[:prefixNeed])
	copy(out.Values[prefixNeed:], suffix.Values[:suffixNeed])
	return out, nil
}

func appendEncoderKVLayers(prefix, suffix []EncoderKVLayer) ([]EncoderKVLayer, error) {
	if len(suffix) == 0 {
		out := make([]EncoderKVLayer, len(prefix))
		for i := range prefix {
			out[i] = cloneEncoderKVLayer(prefix[i])
		}
		return out, nil
	}
	if len(prefix) == 0 {
		out := make([]EncoderKVLayer, len(suffix))
		for i := range suffix {
			layer, err := appendEncoderKVLayer(EncoderKVLayer{}, suffix[i])
			if err != nil {
				return nil, err
			}
			out[i] = layer
		}
		return out, nil
	}
	if len(prefix) != len(suffix) {
		return nil, fmt.Errorf("encoder KV append layer count mismatch prefix=%d suffix=%d", len(prefix), len(suffix))
	}
	out := make([]EncoderKVLayer, len(prefix))
	for i := range prefix {
		layer, err := appendEncoderKVLayer(prefix[i], suffix[i])
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
		out[i] = layer
	}
	return out, nil
}
