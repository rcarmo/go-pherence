package ideogram4

import (
	"os"
	"strings"
)

func enableDiTVectorGPUForPhase() func() {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_DIT_VECTOR")))
	if v != "1" && v != "true" && v != "yes" && v != "on" {
		return func() {}
	}
	keys := []string{
		"GO_PHERENCE_IDEOGRAM4_GPU_NORM",
		"GO_PHERENCE_IDEOGRAM4_GPU_MROPE",
		"GO_PHERENCE_IDEOGRAM4_GPU_ATTN",
		"GO_PHERENCE_IDEOGRAM4_GPU_MLP",
	}
	old := make(map[string]string, len(keys))
	set := make(map[string]bool, len(keys))
	for _, k := range keys {
		old[k], set[k] = os.LookupEnv(k)
		_ = os.Setenv(k, "1")
	}
	return func() {
		for _, k := range keys {
			if set[k] {
				_ = os.Setenv(k, old[k])
			} else {
				_ = os.Unsetenv(k)
			}
		}
	}
}
