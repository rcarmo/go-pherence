//go:build !linux || !riscv64

package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "cmd/spacemit/testi8i4 is only supported on linux/riscv64")
	os.Exit(1)
}
