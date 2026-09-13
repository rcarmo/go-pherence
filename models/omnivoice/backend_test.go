package omnivoice

import (
	"strings"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func TestParseBackendMode(t *testing.T) {
	cases := map[string]BackendMode{
		"":       BackendAuto,
		"auto":   BackendAuto,
		"cpu":    BackendCPU,
		"simd":   BackendCPU,
		"vulkan": BackendVulkan,
	}
	for input, want := range cases {
		got, err := ParseBackendMode(input)
		if err != nil {
			t.Fatalf("ParseBackendMode(%q): %v", input, err)
		}
		if got != want {
			t.Fatalf("ParseBackendMode(%q)=%q want %q", input, got, want)
		}
	}
	if _, err := ParseBackendMode("cuda"); err == nil {
		t.Fatal("unexpected success for unsupported backend")
	}
}

func TestDiscoverBackendCPUSkipsVulkanProbe(t *testing.T) {
	oldEnv := probeVulkanEnvironment
	oldRuntime := probeVulkanRuntime
	defer func() {
		probeVulkanEnvironment = oldEnv
		probeVulkanRuntime = oldRuntime
	}()
	called := false
	probeVulkanEnvironment = func() vulkanEnvironment {
		called = true
		return vulkanEnvironment{}
	}
	probeVulkanRuntime = func() vulkanRuntimeProbeResult {
		called = true
		return vulkanRuntimeProbeResult{}
	}
	report := DiscoverBackend(BackendCPU)
	if called {
		t.Fatal("explicit CPU discovery should not probe Vulkan")
	}
	if report.CPU.Capabilities != simd.RuntimeCapabilities() {
		t.Fatalf("cpu caps mismatch: got=%+v want=%+v", report.CPU.Capabilities, simd.RuntimeCapabilities())
	}
	if report.Recommended.Backend != ExecutionBackendCPU || !report.Recommended.Implemented {
		t.Fatalf("bad recommendation: %+v", report.Recommended)
	}
}

func TestDiscoverBackendDistinguishesSoftwareVulkan(t *testing.T) {
	oldEnv := probeVulkanEnvironment
	oldRuntime := probeVulkanRuntime
	defer func() {
		probeVulkanEnvironment = oldEnv
		probeVulkanRuntime = oldRuntime
	}()
	probeVulkanEnvironment = func() vulkanEnvironment {
		return vulkanEnvironment{
			LoaderPaths: []string{"/usr/lib/x86_64-linux-gnu/libvulkan.so.1"},
			ICDFiles:    []string{"/usr/share/vulkan/icd.d/lvp_icd.json"},
			DRMNodes:    nil,
		}
	}
	probeVulkanRuntime = func() vulkanRuntimeProbeResult {
		return vulkanRuntimeProbeResult{SoftwareAvailable: true, SoftwareDeviceName: "llvmpipe (LLVM 19.1.7, 256 bits)"}
	}
	report := DiscoverBackend(BackendAuto)
	if report.Vulkan.HardwareAvailable {
		t.Fatal("unexpected hardware Vulkan detection")
	}
	if !report.Vulkan.SoftwareAvailable {
		t.Fatal("expected software Vulkan detection")
	}
	if report.Vulkan.SoftwareDeviceName == "" || !strings.Contains(strings.ToLower(report.Vulkan.SoftwareDeviceName), "llvmpipe") {
		t.Fatalf("bad software device name: %q", report.Vulkan.SoftwareDeviceName)
	}
	if report.Vulkan.Implemented {
		t.Fatal("Vulkan should remain unimplemented for OmniVoice")
	}
	if report.Vulkan.NativeDispatch {
		t.Fatal("OmniVoice should not claim native Vulkan dispatch")
	}
	if report.Recommended.Backend != ExecutionBackendCPU {
		t.Fatalf("bad recommendation: %+v", report.Recommended)
	}
}

func TestSelectBackendAutoPrefersCPUWhenHardwareVulkanExists(t *testing.T) {
	oldEnv := probeVulkanEnvironment
	oldRuntime := probeVulkanRuntime
	defer func() {
		probeVulkanEnvironment = oldEnv
		probeVulkanRuntime = oldRuntime
	}()
	probeVulkanEnvironment = func() vulkanEnvironment {
		return vulkanEnvironment{LoaderPaths: []string{"/usr/lib/libvulkan.so.1"}, DRMNodes: []string{"/dev/dri/renderD128"}}
	}
	probeVulkanRuntime = func() vulkanRuntimeProbeResult {
		return vulkanRuntimeProbeResult{HardwareAvailable: true, HardwareDeviceName: "Intel Arc"}
	}
	selection, err := SelectBackend(BackendAuto)
	if err != nil {
		t.Fatalf("SelectBackend(auto): %v", err)
	}
	if selection.Backend != ExecutionBackendCPU || !selection.Implemented {
		t.Fatalf("bad selection: %+v", selection)
	}
	if !strings.Contains(selection.Reason, "does not dispatch") {
		t.Fatalf("reason %q does not describe the Vulkan implementation gap", selection.Reason)
	}
}

func TestSelectBackendExplicitVulkanFailsLoudly(t *testing.T) {
	oldEnv := probeVulkanEnvironment
	oldRuntime := probeVulkanRuntime
	defer func() {
		probeVulkanEnvironment = oldEnv
		probeVulkanRuntime = oldRuntime
	}()
	probeVulkanEnvironment = func() vulkanEnvironment {
		return vulkanEnvironment{LoaderPaths: []string{"/usr/lib/libvulkan.so.1"}}
	}
	probeVulkanRuntime = func() vulkanRuntimeProbeResult {
		return vulkanRuntimeProbeResult{HardwareAvailable: true, HardwareDeviceName: "Intel Arc"}
	}
	_, err := SelectBackend(BackendVulkan)
	if err == nil {
		t.Fatal("expected explicit Vulkan request to fail")
	}
	if !strings.Contains(err.Error(), "implemented=false") || !strings.Contains(err.Error(), "CPU/SIMD") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCPUReportsSIMDLimits(t *testing.T) {
	report := DiscoverBackend(BackendCPU)
	caps := simd.RuntimeCapabilities()
	if report.CPU.FullGraphSIMD {
		t.Fatal("full graph SIMD claimed despite scalar stages")
	}
	if report.CPU.ApproximateSIMD != (caps.Arch == "amd64" && caps.HasVec) {
		t.Fatal("approximate SIMD flag disagrees with runtime")
	}
	if len(report.CPU.ScalarFallbacks) < 5 {
		t.Fatal("missing fallback limits")
	}
	report.CPU.ScalarFallbacks[0] = "mutated"
	if DiscoverBackend(BackendCPU).CPU.ScalarFallbacks[0] == "mutated" {
		t.Fatal("report leaks shared mutable limits")
	}
}
