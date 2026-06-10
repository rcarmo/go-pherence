//go:build riscv64

package ideogram4

func k3FP8Prewarm(f *FP8Linear) bool {
	if f == nil || !k3Enabled() {
		return false
	}
	f.k3.ensureWeightF16(f)
	return true
}
