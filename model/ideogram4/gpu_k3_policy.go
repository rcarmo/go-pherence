package ideogram4

import "errors"

var errK3GPUDisabled = errors.New("ideogram4 gpu disabled in K3 native mode")

// gpuDisabledByK3 hard-disables all Ideogram NVIDIA paths when the native K3
// path is selected. Milk-V/K3 has no NVIDIA hardware; K3 runs must use only
// X100/RVV/A100/IME paths and normal CPU fallbacks.
func gpuDisabledByK3() bool {
	return k3Enabled()
}
