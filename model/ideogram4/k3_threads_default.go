//go:build !riscv64

package ideogram4

import (
	"os"
	"strconv"
	"strings"
)

func k3Threads() int {
	if s := strings.TrimSpace(os.Getenv("GO_PHERENCE_IDEOGRAM4_K3_THREADS")); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			return n
		}
	}
	return 8
}
