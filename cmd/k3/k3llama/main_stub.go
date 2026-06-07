//go:build !llamacpp || !cgo

package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "k3llama requires llama.cpp headers/libraries; rebuild with -tags llamacpp on a configured system")
	os.Exit(2)
}
