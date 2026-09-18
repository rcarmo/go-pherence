package simd

//go:noescape
func siluFinishAsm(dst, gate, up, exp []float32)

func siluFinish(dst, gate, up, exp []float32) {
	n := 0
	if HasVecAsm {
		n = len(dst) &^ 7
		if n > 0 {
			siluFinishAsm(dst[:n], gate[:n], up[:n], exp[:n])
		}
	}
	for i := n; i < len(dst); i++ {
		dst[i] = (gate[i] / (1 + exp[i])) * up[i]
	}
}
