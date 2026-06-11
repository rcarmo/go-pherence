//go:build riscv64

package ideogram4

func k3VAESpatialAttention(q, k, v FeatureMap, scale float32) (FeatureMap, bool, error) {
	if !k3Enabled() {
		return FeatureMap{}, false, nil
	}
	if q.C != k.C || q.C != v.C || q.H != k.H || q.H != v.H || q.W != k.W || q.W != v.W {
		return FeatureMap{}, true, nil
	}
	// K3 runtime seam for VAE spatial attention. Current body preserves scalar
	// semantics via the generic attention implementation; replace with tiled RVV
	// attention for high-resolution decode.
	out, err := vaeSpatialAttentionCPU(q, k, v, scale)
	return out, true, err
}
