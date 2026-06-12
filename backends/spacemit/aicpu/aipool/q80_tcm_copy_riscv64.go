//go:build riscv64

package aipool

import "github.com/rcarmo/go-pherence/backends/spacemit/rvv"

func q80CopyTCMBytes(dst, src []byte) { rvv.CopyTCMBytes(dst, src) }
