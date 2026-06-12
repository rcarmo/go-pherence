//go:build !riscv64

package aipool

func q80CopyTCMBytes(dst, src []byte) { copy(dst, src) }
