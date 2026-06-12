//go:build !riscv64

package ime2

func K3I8I4M1(a *byte, b *byte, c *float32, kBlks int, nBlks int) {
	panic("K3I8I4M1: not implemented on this platform")
}

func K3I8I4M4(a *byte, b *byte, c *float32, kBlks int, ldcBytes int) {
	panic("K3I8I4M4: not implemented on this platform")
}

func k3I8I4M1Residual(a *byte, b *byte, residual *float32, c *float32, kBlks int, nBlks int) {
	panic("k3I8I4M1Residual: not implemented on this platform")
}

func K3I8I8M1(a *byte, b *byte, c *float32, kBlks int, nBlks int) {
	panic("K3I8I8M1: not implemented on this platform")
}

func K3I8I8M4(a *byte, b *byte, c *float32, kBlks int, ldcBytes int) {
	panic("K3I8I8M4: not implemented on this platform")
}
