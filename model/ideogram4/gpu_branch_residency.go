package ideogram4

import "os"

func withBranchLayerCache(branch string, fn func() ([]float32, error)) ([]float32, error) {
	if branch == "" {
		return fn()
	}
	keys := []string{
		"GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW",
		"GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_START",
	}
	old := make(map[string]string, len(keys))
	set := make(map[string]bool, len(keys))
	for _, k := range keys {
		old[k], set[k] = os.LookupEnv(k)
	}
	if v, ok := os.LookupEnv("GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW_" + branch); ok && v != "" {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW", v)
	}
	if v, ok := os.LookupEnv("GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_START_" + branch); ok && v != "" {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_START", v)
	}
	defer func() {
		for _, k := range keys {
			if set[k] {
				_ = os.Setenv(k, old[k])
			} else {
				_ = os.Unsetenv(k)
			}
		}
	}()
	return fn()
}
