package k3

// Select returns the highest-priority available OpBackend.
// Probes backends in order: SpacemiT → Vulkan → CPU SIMD.
func Select() (OpBackend, Tier) {
	caps := Probe()
	switch caps.BestTier() {
	case TierSpacemiT:
		return SpacemiTBackend{}, TierSpacemiT
	case TierVulkan:
		return VulkanBackend{}, TierVulkan
	default:
		return SIMDBackend{}, TierCPU
	}
}

// SelectAll returns all available backends in descending priority order.
func SelectAll() []OpBackend {
	caps := Probe()
	var out []OpBackend
	if caps.SpacemiT {
		out = append(out, SpacemiTBackend{})
	}
	if caps.Vulkan {
		out = append(out, VulkanBackend{})
	}
	out = append(out, SIMDBackend{})
	return out
}
