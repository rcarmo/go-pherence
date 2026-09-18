//go:build !cgo || !q4kcshim || !linux || !riscv64

package q4kcshim

func CallIME2GemmI8I4(a []byte, b []byte, out []float32, countN, kBlks int) {
	panic("q4kcshim: not built for this platform")
}

func CallIME2GemmI8I8(a []byte, b []byte, out []float32, countN, kBlks int) {
	panic("q4kcshim: not built for this platform")
}
