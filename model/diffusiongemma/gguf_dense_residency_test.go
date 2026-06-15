package diffusiongemma

import "testing"

func TestGGUFDenseLayerResidentPolicy(t *testing.T) {
	all := GPUDispatcher{}
	for _, layer := range []int{0, 1, 29} {
		if !all.ggufDenseLayerResident(layer) {
			t.Fatalf("zero prefix should preserve legacy all-resident behavior for layer %d", layer)
		}
	}
	bounded := GPUDispatcher{ResidentLayerPrefix: 2}
	if !bounded.ggufDenseLayerResident(0) || !bounded.ggufDenseLayerResident(1) {
		t.Fatalf("positive prefix should keep layers below prefix resident")
	}
	if bounded.ggufDenseLayerResident(2) || bounded.ggufDenseLayerResident(29) {
		t.Fatalf("positive prefix should make layers at/above prefix temporary")
	}
	enc := bounded.cpuFallback()
	if !enc.ggufDenseLayerResident(1) || enc.ggufDenseLayerResident(2) {
		t.Fatalf("encoder dispatcher fallback should use identical GGUF dense residency policy")
	}
}
