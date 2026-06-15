// Package k3flags applies the shared Ideogram4 K3/RVV environment toggles for
// the image command-line tools.
package k3flags

import (
	"fmt"
	"os"
)

// Apply sets the GO_PHERENCE_IDEOGRAM4_K3* environment variables from CLI flags.
func Apply(k3 bool, threads int, prewarm bool) {
	if k3 {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_K3", "1")
	}
	if threads > 0 {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_K3_THREADS", fmt.Sprintf("%d", threads))
	}
	if prewarm {
		_ = os.Setenv("GO_PHERENCE_IDEOGRAM4_K3_PREWARM", "1")
	}
}
