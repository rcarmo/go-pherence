//go:build !riscv64

package ideogram4

func k3VAESpatialAttention(_, _, _ FeatureMap, _ float32) (FeatureMap, bool, error) {
	return FeatureMap{}, false, nil
}
