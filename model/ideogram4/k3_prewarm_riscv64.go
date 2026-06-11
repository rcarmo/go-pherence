//go:build riscv64

package ideogram4

func k3FP8Prewarm(f *FP8Linear) bool {
	if f == nil || !k3Enabled() {
		return false
	}
	if k3A100Q8Enabled() {
		return f.k3.ensureWeightQ80RowScale(f).Valid
	}
	f.k3.ensureWeightF16(f)
	return true
}
