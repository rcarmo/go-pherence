package ime2

// IME2 i8×i8 (int8 weights, int8 activations) GEMM kernels — native RVV vmadot
// assembly with VLEN=1024. Companions to the i8i4 kernels in this package.

//go:noescape
func K3I8I8M1(a *byte, b *byte, c *float32, kBlks int, nBlks int)

//go:noescape
func K3I8I8M1Groups(a *byte, b *byte, c *float32, kBlks int, nGroups int)

//go:noescape
func K3I8I8M4(a *byte, b *byte, c *float32, kBlks int, ldcBytes int)
