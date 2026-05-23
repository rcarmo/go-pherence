// Package k3 provides a prioritised compute-backend stack for the SpacemiT K3
// SoC (MilkV Jupiter 2 and similar boards).
//
// Priority at runtime (highest first):
//
//  1. SpacemiT ORT – A100/X100 AI cores via TCM + SpaceMITExecutionProvider
//  2. Vulkan        – PowerVR BXM-4-64 MC1 GPU (Vulkan 1.3, OpenCL 3.0)
//  3. CPU SIMD      – RVV 1.0 + VLEN=512 tuned paths (always available)
package k3

import (
	"fmt"

	"github.com/rcarmo/go-pherence/backends/spacemitort"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

// Tier identifies which backend is active.
type Tier int

const (
	TierCPU      Tier = iota // RVV SIMD – always available
	TierVulkan               // PowerVR GPU compute
	TierSpacemiT             // SpacemiT ORT / A100 AI cores
)

func (t Tier) String() string {
	switch t {
	case TierCPU:
		return "CPU-SIMD-RVV"
	case TierVulkan:
		return "Vulkan-PowerVR"
	case TierSpacemiT:
		return "SpacemiT-ORT"
	}
	return "unknown"
}

// Capabilities summarises what is available on this board.
type Capabilities struct {
	CPU      bool // always true
	Vulkan   bool
	VulkanDev string
	SpacemiT bool
	ortCaps  spacemitort.Capabilities
}

// Probe detects available backends.
func Probe() Capabilities {
	c := Capabilities{CPU: true}
	vulkan.VulkanInit()
	c.Vulkan = vulkan.VulkanReady()
	c.VulkanDev = vulkan.VulkanDeviceName()
	c.ortCaps = spacemitort.RuntimeCapabilities()
	c.SpacemiT = c.ortCaps.LikelyUsable
	return c
}

// BestTier returns the highest-priority available backend tier.
func (c Capabilities) BestTier() Tier {
	if c.SpacemiT {
		return TierSpacemiT
	}
	if c.Vulkan {
		return TierVulkan
	}
	return TierCPU
}

// Summary returns a human-readable table of backend availability.
func (c Capabilities) Summary() string {
	ortc := c.ortCaps
	return fmt.Sprintf(
		"K3 backend capabilities\n"+
			"  CPU SIMD (RVV)   : available\n"+
			"  Vulkan GPU       : %v  device=%s\n"+
			"  SpacemiT ORT     : %v\n"+
			"    /dev/tcm       : %v\n"+
			"    libspine_tcm   : %v\n"+
			"    libspacemit_ep : %v\n"+
			"    libonnxruntime : %v\n"+
			"    python ORT pkg : %v\n"+
			"  Best tier        : %s\n",
		c.Vulkan, c.VulkanDev,
		c.SpacemiT,
		ortc.HasDevTCM,
		ortc.HasSpineTCMLibrary,
		ortc.HasSpacemitEPLibrary,
		ortc.HasONNXRuntimeLibrary,
		ortc.HasPythonORTPackage,
		c.BestTier(),
	)
}
