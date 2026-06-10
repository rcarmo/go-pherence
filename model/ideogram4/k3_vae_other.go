//go:build !riscv64

package ideogram4

func k3GroupNorm(_ FeatureMap, _ int, _, _ []float32, _ float32) (FeatureMap, bool, error) {
	return FeatureMap{}, false, nil
}
func k3UpsampleNearest(_ FeatureMap, _ int) (FeatureMap, bool, error) {
	return FeatureMap{}, false, nil
}
func k3RGB(_ FeatureMap) (Image, bool) { return Image{}, false }
