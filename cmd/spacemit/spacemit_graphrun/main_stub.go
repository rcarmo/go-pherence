//go:build !ggml || !cgo || !linux

package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "spacemit_graphrun requires GGML headers/libraries; rebuild with -tags ggml on a configured K3/GGML system")
	os.Exit(2)
}
