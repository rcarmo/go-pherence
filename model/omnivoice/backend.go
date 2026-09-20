package omnivoice

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/commandcapture"
)

// BackendMode names the user-visible OmniVoice execution policy.
type BackendMode string

const (
	BackendAuto   BackendMode = "auto"
	BackendCPU    BackendMode = "cpu"
	BackendVulkan BackendMode = "vulkan"
)

// ExecutionBackend is the concrete backend the native OmniVoice runtime will use.
type ExecutionBackend string

const (
	ExecutionBackendCPU    ExecutionBackend = "cpu"
	ExecutionBackendVulkan ExecutionBackend = "vulkan"
)

// BackendSelection is the resolved execution policy for one request.
type BackendSelection struct {
	Requested   BackendMode      `json:"requested"`
	Backend     ExecutionBackend `json:"backend"`
	Implemented bool             `json:"implemented"`
	Reason      string           `json:"reason,omitempty"`
}

// BackendRecommendation is the best truthful backend recommendation for the
// current host and the currently implemented OmniVoice runtime.
type BackendRecommendation struct {
	Backend     ExecutionBackend `json:"backend"`
	Implemented bool             `json:"implemented"`
	Reason      string           `json:"reason"`
}

// CPUBackendInfo describes the current CPU/SIMD path.
type CPUBackendInfo struct {
	Available       bool              `json:"available"`
	Implemented     bool              `json:"implemented"`
	UsesCheckedSIMD bool              `json:"uses_checked_simd"`
	FullGraphSIMD   bool              `json:"full_graph_simd"`
	LowAllocation   bool              `json:"low_allocation"`
	Capabilities    simd.Capabilities `json:"capabilities"`
	ApproximateSIMD bool              `json:"approximate_nonlinear_simd"`
	ScalarFallbacks []string          `json:"scalar_fallbacks"`
}

// VulkanBackendInfo describes Vulkan availability and why OmniVoice does or does
// not use it.
type VulkanBackendInfo struct {
	Probed                 bool     `json:"probed"`
	LoaderPresent          bool     `json:"loader_present"`
	LoaderPaths            []string `json:"loader_paths,omitempty"`
	ICDFiles               []string `json:"icd_files,omitempty"`
	DRMNodes               []string `json:"drm_nodes,omitempty"`
	HardwareProbeAttempted bool     `json:"hardware_probe_attempted"`
	HardwareAvailable      bool     `json:"hardware_available"`
	HardwareDeviceName     string   `json:"hardware_device_name,omitempty"`
	SoftwareProbeAttempted bool     `json:"software_probe_attempted"`
	SoftwareAvailable      bool     `json:"software_available"`
	SoftwareDeviceName     string   `json:"software_device_name,omitempty"`
	Implemented            bool     `json:"implemented"`
	NativeDispatch         bool     `json:"native_dispatch"`
	LowLevelOps            []string `json:"low_level_ops,omitempty"`
	MissingForOmniVoice    []string `json:"missing_for_omnivoice,omitempty"`
	Reason                 string   `json:"reason,omitempty"`
}

// BackendReport is the backend discovery surface exported by the OmniVoice
// package. It is intentionally conservative and reports implementation gaps
// explicitly instead of silently routing explicit Vulkan requests to CPU.
type BackendReport struct {
	Requested      BackendMode           `json:"requested"`
	CPU            CPUBackendInfo        `json:"cpu"`
	Vulkan         VulkanBackendInfo     `json:"vulkan"`
	Recommended    BackendRecommendation `json:"recommended"`
	RecommendedFor BackendMode           `json:"recommended_for"`
}

type vulkanEnvironment struct {
	LoaderPaths []string
	ICDFiles    []string
	DRMNodes    []string
}

type vulkanRuntimeProbeResult struct {
	HardwareAvailable  bool
	HardwareDeviceName string
	SoftwareAvailable  bool
	SoftwareDeviceName string
}

var (
	probeVulkanEnvironment = defaultVulkanEnvironmentProbe
	probeVulkanRuntime     = defaultVulkanRuntimeProbe
)

var omnivoiceVulkanLowLevelOps = []string{
	"vec_add_f32",
	"vec_add_bf16",
	"rms_norm_f32",
	"rms_norm_bf16",
	"rms_norm_no_scale_f32",
	"gemv_f32",
	"gemv_bf16_mixed",
	"silu_mul_f32",
	"gelu_tanh_mul_f32",
	"rope_partial_f32",
	"attention_score_f32",
}

var omnivoiceVulkanMissingPieces = []string{
	"end-to-end OmniVoice backend selection and dispatch",
	"GPU-resident weight upload and residency plan for q/k/v/o and MLP weights",
	"batched/projection GEMM coverage for q/k/v, o_proj, gate/up/down projections",
	"attention softmax kernel integrated into the native block path",
	"probability-times-value accumulation path for full attention",
	"bounded reusable scratch and transfer planner to avoid per-op host round trips",
	"parity-tested end-to-end OmniVoice block execution on Vulkan",
}

func cpuScalarFallbacks() []string {
	caps := simd.RuntimeCapabilities()
	out := []string{
		"GEMM partial tiles and packing/scatter orchestration",
		"nonlinear range/exception/tail fallbacks",
		"sequential reductions and log normalisers",
		"Gumbel logarithms, top-k selection and token bookkeeping",
		"audio preprocessing, tokenization and WAV I/O",
	}
	if caps.Arch != "amd64" || !caps.HasVec {
		out = append(out, "bounded exp, sine, ELU, erf-GELU and SiLU finishing lack an active SIMD implementation on this host")
	}
	if !caps.HasSGEMM {
		out = append(out, "SGEMM assembly unavailable on this host")
	}
	return out
}

func ParseBackendMode(s string) (BackendMode, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", string(BackendAuto):
		return BackendAuto, nil
	case string(BackendCPU), "simd":
		return BackendCPU, nil
	case string(BackendVulkan):
		return BackendVulkan, nil
	default:
		return "", fmt.Errorf("omnivoice: unsupported backend %q", s)
	}
}

// DiscoverBackend reports the truthful execution surface for the requested mode.
// Explicit CPU requests avoid Vulkan initialization entirely.
func DiscoverBackend(mode BackendMode) BackendReport {
	if mode == "" {
		mode = BackendAuto
	}
	report := BackendReport{
		Requested: mode,
		CPU: CPUBackendInfo{
			Available:       true,
			Implemented:     true,
			UsesCheckedSIMD: true,
			FullGraphSIMD:   false,
			LowAllocation:   true,
			Capabilities:    simd.RuntimeCapabilities(),
			ApproximateSIMD: simd.RuntimeCapabilities().Arch == "amd64" && simd.RuntimeCapabilities().HasVec,
			ScalarFallbacks: cpuScalarFallbacks(),
		},
		RecommendedFor: mode,
	}
	if mode == BackendCPU {
		report.Recommended = BackendRecommendation{
			Backend:     ExecutionBackendCPU,
			Implemented: true,
			Reason:      "explicit CPU mode requested; native reference encoding, generation and decoding use the CPU/SIMD path",
		}
		return report
	}

	report.Vulkan = discoverVulkanBackend()
	report.Recommended = recommendBackend(report)
	return report
}

// SelectBackend resolves one backend request. Explicit Vulkan requests fail when
// Vulkan is unavailable or when OmniVoice has no compatible end-to-end Vulkan
// execution path.
func SelectBackend(mode BackendMode) (BackendSelection, error) {
	report := DiscoverBackend(mode)
	switch report.Requested {
	case BackendCPU:
		return BackendSelection{
			Requested:   report.Requested,
			Backend:     ExecutionBackendCPU,
			Implemented: true,
			Reason:      report.Recommended.Reason,
		}, nil
	case BackendAuto, "":
		return BackendSelection{
			Requested:   BackendAuto,
			Backend:     report.Recommended.Backend,
			Implemented: report.Recommended.Implemented,
			Reason:      report.Recommended.Reason,
		}, nil
	case BackendVulkan:
		if report.Vulkan.HardwareAvailable {
			return BackendSelection{}, fmt.Errorf("omnivoice: backend=vulkan detected hardware device %q, but implemented=false: current native OmniVoice execution is CPU/SIMD only", report.Vulkan.HardwareDeviceName)
		}
		if report.Vulkan.SoftwareAvailable {
			return BackendSelection{}, fmt.Errorf("omnivoice: backend=vulkan resolved only to software device %q, and implemented=false: current native OmniVoice execution is CPU/SIMD only", report.Vulkan.SoftwareDeviceName)
		}
		return BackendSelection{}, fmt.Errorf("omnivoice: backend=vulkan unavailable: %s", unavailableVulkanReason(report.Vulkan))
	default:
		return BackendSelection{}, fmt.Errorf("omnivoice: unsupported backend %q", mode)
	}
}

func recommendBackend(report BackendReport) BackendRecommendation {
	reason := "current OmniVoice native execution is CPU/SIMD only"
	switch {
	case report.Vulkan.HardwareAvailable:
		reason = fmt.Sprintf("hardware Vulkan device %q is present, but OmniVoice does not dispatch to it yet; use CPU/SIMD", report.Vulkan.HardwareDeviceName)
	case report.Vulkan.SoftwareAvailable:
		reason = fmt.Sprintf("only software Vulkan device %q is present; current OmniVoice native execution remains CPU/SIMD", report.Vulkan.SoftwareDeviceName)
	case report.Vulkan.Probed:
		reason = fmt.Sprintf("no usable hardware Vulkan device detected; %s; use CPU/SIMD", unavailableVulkanReason(report.Vulkan))
	}
	return BackendRecommendation{Backend: ExecutionBackendCPU, Implemented: true, Reason: reason}
}

func discoverVulkanBackend() VulkanBackendInfo {
	env := probeVulkanEnvironment()
	info := VulkanBackendInfo{
		Probed:              true,
		LoaderPresent:       len(env.LoaderPaths) > 0,
		LoaderPaths:         env.LoaderPaths,
		ICDFiles:            env.ICDFiles,
		DRMNodes:            env.DRMNodes,
		Implemented:         false,
		NativeDispatch:      false,
		LowLevelOps:         append([]string(nil), omnivoiceVulkanLowLevelOps...),
		MissingForOmniVoice: append([]string(nil), omnivoiceVulkanMissingPieces...),
	}
	res := probeVulkanRuntime()
	info.HardwareProbeAttempted = true
	info.HardwareAvailable = res.HardwareAvailable
	info.HardwareDeviceName = res.HardwareDeviceName
	info.SoftwareProbeAttempted = !res.HardwareAvailable
	info.SoftwareAvailable = res.SoftwareAvailable
	info.SoftwareDeviceName = res.SoftwareDeviceName
	info.Reason = unavailableVulkanReason(info)
	return info
}

func unavailableVulkanReason(info VulkanBackendInfo) string {
	switch {
	case info.HardwareAvailable:
		return fmt.Sprintf("hardware Vulkan detected on %q, but implemented=false for OmniVoice", info.HardwareDeviceName)
	case info.SoftwareAvailable:
		return fmt.Sprintf("only software Vulkan device %q detected", info.SoftwareDeviceName)
	}
	parts := make([]string, 0, 4)
	if !info.LoaderPresent {
		parts = append(parts, "no libvulkan loader found in common system paths")
	}
	if len(info.ICDFiles) == 0 {
		parts = append(parts, "no Vulkan ICD JSON files found")
	}
	if len(info.DRMNodes) == 0 {
		parts = append(parts, "no /dev/dri device nodes visible")
	}
	if len(parts) == 0 {
		parts = append(parts, "Vulkan loader did not expose a usable non-CPU device")
	}
	return strings.Join(parts, "; ")
}

func defaultVulkanEnvironmentProbe() vulkanEnvironment {
	return vulkanEnvironment{
		LoaderPaths: collectExistingPaths(16,
			"/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
			"/usr/lib/x86_64-linux-gnu/libvulkan.so",
			"/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
			"/usr/lib/aarch64-linux-gnu/libvulkan.so",
			"/usr/lib/riscv64-linux-gnu/libvulkan.so.1",
			"/usr/lib/riscv64-linux-gnu/libvulkan.so",
			"/usr/lib64/libvulkan.so.1",
			"/usr/lib64/libvulkan.so",
			"/usr/lib/libvulkan.so.1",
			"/usr/lib/libvulkan.so",
			"/lib/x86_64-linux-gnu/libvulkan.so.1",
			"/lib/aarch64-linux-gnu/libvulkan.so.1",
			"/lib/riscv64-linux-gnu/libvulkan.so.1",
		),
		ICDFiles: collectGlobMatches(32,
			"/usr/share/vulkan/icd.d/*.json",
			"/etc/vulkan/icd.d/*.json",
		),
		DRMNodes: collectGlobMatches(32, "/dev/dri/*"),
	}
}

var runtimeProbeOnce sync.Once
var runtimeProbeResult vulkanRuntimeProbeResult

func defaultVulkanRuntimeProbe() vulkanRuntimeProbeResult {
	runtimeProbeOnce.Do(func() { runtimeProbeResult = initVulkanRuntimeProbe() })
	return runtimeProbeResult
}
func initVulkanRuntimeProbe() vulkanRuntimeProbeResult {
	exe, err := os.Executable()
	if err != nil {
		return vulkanRuntimeProbeResult{}
	}
	// Only the CLI implements this private probe command. A generic library host
	// should supply its own capability policy; unsupported hosts return unknown.
	for _, software := range []bool{false, true} {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		env := append(os.Environ(), "GO_PHERENCE_VULKAN_ALLOW_CPU="+map[bool]string{false: "0", true: "1"}[software])
		raw, _, err := commandcapture.Run(ctx, exe, []string{"-mode", "vulkan-probe-internal"}, env, 64<<10)
		cancel()
		if err != nil {
			continue
		}
		var probe struct {
			Available bool   `json:"available"`
			Name      string `json:"name"`
		}
		if json.Unmarshal(raw, &probe) != nil || !probe.Available {
			continue
		}
		if isSoftwareVulkanDevice(probe.Name) {
			return vulkanRuntimeProbeResult{SoftwareAvailable: true, SoftwareDeviceName: probe.Name}
		}
		return vulkanRuntimeProbeResult{HardwareAvailable: true, HardwareDeviceName: probe.Name}
	}
	return vulkanRuntimeProbeResult{}
}

func isSoftwareVulkanDevice(name string) bool {
	n := strings.ToLower(strings.TrimSpace(name))
	return strings.Contains(n, "llvmpipe") ||
		strings.Contains(n, "lavapipe") ||
		strings.Contains(n, "swiftshader") ||
		strings.Contains(n, "software") ||
		strings.Contains(n, "cpu")
}

func collectExistingPaths(limit int, paths ...string) []string {
	seen := make(map[string]struct{}, len(paths))
	out := make([]string, 0, len(paths))
	for _, path := range paths {
		if path == "" {
			continue
		}
		if _, ok := seen[path]; ok {
			continue
		}
		if _, err := os.Stat(path); err == nil {
			seen[path] = struct{}{}
			out = append(out, path)
			if limit > 0 && len(out) >= limit {
				break
			}
		}
	}
	sort.Strings(out)
	return out
}

func collectGlobMatches(limit int, patterns ...string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, 8)
	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			continue
		}
		for _, match := range matches {
			if _, ok := seen[match]; ok {
				continue
			}
			seen[match] = struct{}{}
			out = append(out, match)
			if limit > 0 && len(out) >= limit {
				sort.Strings(out)
				return out
			}
		}
	}
	sort.Strings(out)
	return out
}
