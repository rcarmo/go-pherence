//go:build !riscv64

package ideogram4

func k3CFGStep(_ Latents, _, _ Latents, _, _ float32) (Latents, bool, error) {
	return Latents{}, false, nil
}
