//go:build linux

package vulkan

// Explicit one-shot diagnostic probes. Never run by the default test suite.
// No optimiser, model checkpoint/export writer, PTX or fallback is invoked.
import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"
)

const captureSlotPin = "dc6d1727e33b9ffea04c19a67066e595385062120cd581980b2038a97a9ea90d"
const captureTimePin = "8e5e583469c7f1f8d0045230cab2aa46cd5b53b2c057f361c284b5b1bf26764f"

type captureSlots struct {
	Resident, Tape         []int64
	Tensors, Stages, Plans int
	Operations             map[string]int
}
type capturePhase struct {
	Name                    string
	RSS, HeapAlloc, HeapSys uint64
	Native                  VulkanMemoryUsage
}
type captureRequirement struct {
	Requested, Size, Alignment uint64
	TypeBits                   uint32
}

func capturePinnedJSON(path, pin string, out any) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() > 8<<20 {
		return fmt.Errorf("capture input bound")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(raw)
	if hex.EncodeToString(sum[:]) != pin {
		return fmt.Errorf("capture input pin")
	}
	return json.Unmarshal(raw, out)
}
func (s captureSlots) validate() (uint64, error) {
	if len(s.Resident) != 661 || len(s.Tape) != 215 || s.Tensors != 1142 || s.Stages != 395 || s.Plans != 7 {
		return 0, fmt.Errorf("capture slot geometry")
	}
	var totals [2]uint64
	for i, group := range [][]int64{s.Resident, s.Tape} {
		for _, n := range group {
			if n <= 0 || n > 1<<30 || n%4 != 0 {
				return 0, fmt.Errorf("capture extent")
			}
			totals[i] += uint64(n)
		}
	}
	if totals != [2]uint64{1822745688, 317970992} {
		return 0, fmt.Errorf("capture payload")
	}
	stages := 0
	for _, n := range s.Operations {
		if n <= 0 || n > 395 {
			return 0, fmt.Errorf("capture stage count")
		}
		stages += n
	}
	if stages != 395 {
		return 0, fmt.Errorf("capture stages")
	}
	return totals[0] + totals[1], nil
}
func captureRequirementsTotal(rows []captureRequirement, cap uint64) (uint64, error) {
	var total uint64
	for _, r := range rows {
		if r.Requested == 0 || r.Size < r.Requested || r.Alignment == 0 || r.Alignment&(r.Alignment-1) != 0 || r.TypeBits == 0 || r.Size > ^uint64(0)-total {
			return 0, fmt.Errorf("capture invalid requirements")
		}
		total += r.Size
	}
	if total > cap {
		return total, fmt.Errorf("capture joint requirement bytes=%d cap=%d", total, cap)
	}
	return total, nil
}
func captureWrite(path string, data any) error {
	raw, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	_, we := f.Write(append(raw, '\n'))
	ce := f.Close()
	if we != nil {
		return we
	}
	return ce
}
func captureSample(name string) (capturePhase, error) {
	raw, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return capturePhase{}, err
	}
	var rss uint64
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			f := strings.Fields(line)
			if len(f) != 3 {
				return capturePhase{}, fmt.Errorf("RSS format")
			}
			rss, err = strconv.ParseUint(f[1], 10, 64)
			if err != nil {
				return capturePhase{}, err
			}
			rss *= 1024
		}
	}
	if rss == 0 {
		return capturePhase{}, fmt.Errorf("missing RSS")
	}
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return capturePhase{name, rss, m.HeapAlloc, m.HeapSys, VulkanMemoryStats()}, nil
}

// Caller holds the lane; create/query/destroy only, with no VkDeviceMemory.
func captureRequirementLocked(size uint64) (captureRequirement, error) {
	if err := vkCheckBufferLimitLocked(size); err != nil {
		return captureRequirement{}, err
	}
	info := struct {
		SType                      uint32
		Next                       uintptr
		Flags                      uint32
		Size                       uint64
		Usage, Sharing, QueueCount uint32
		Queues                     uintptr
	}{SType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, Size: size, Usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT}
	var b VkBuffer
	if r := vkCreateBuffer(vkDevice, unsafe.Pointer(&info), nil, &b); r != VK_SUCCESS {
		return captureRequirement{}, fmt.Errorf("capture vkCreateBuffer %d", r)
	}
	if b == 0 {
		return captureRequirement{}, fmt.Errorf("capture null buffer")
	}
	defer vkDestroyBuffer(vkDevice, b, nil)
	var req vkMemoryRequirements
	vkGetBufferMemoryRequirements(vkDevice, b, unsafe.Pointer(&req))
	out := captureRequirement{size, req.size, req.alignment, req.memoryTypeBits}
	_, err := captureRequirementsTotal([]captureRequirement{out}, ^uint64(0))
	return out, err
}
func captureInit(t *testing.T) {
	t.Helper()
	if os.Getenv("GO_PHERENCE_DISABLE_VULKAN") != "" {
		t.Fatal("selected Vulkan disabled")
	}
	if os.Getenv("VK_DRIVER_FILES") != "/usr/share/vulkan/icd.d/nvidia_icd.json" {
		t.Fatal("NVIDIA ICD pin required")
	}
	if !VulkanInit() {
		t.Fatal("Vulkan unavailable; no fallback")
	}
	if VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("device mismatch", VulkanDeviceName())
	}
	if u := VulkanMemoryStats(); u.Bytes != 0 || u.Allocations != 0 || u.InFlight || u.Uncertain || u.DeviceLost {
		t.Fatal("nonfresh Vulkan process", u)
	}
}

func TestVulkanOfflineCaptureValidation(t *testing.T) {
	s := captureSlots{Resident: make([]int64, 661), Tape: make([]int64, 215), Tensors: 1142, Stages: 395, Plans: 7, Operations: map[string]int{"vector": 395}}
	for i := range s.Resident {
		s.Resident[i] = 4
	}
	for i := range s.Tape {
		s.Tape[i] = 4
	}
	s.Resident[0] = 1073741824
	s.Resident[1] = 1822745688 - 1073741824 - 659*4
	s.Tape[0] = 317970992 - 214*4
	if n, e := s.validate(); e != nil || n != 2140716680 {
		t.Fatal(n, e)
	}
	s.Resident[3] = 0
	if _, e := s.validate(); e == nil {
		t.Fatal("extent")
	}
	s.Resident[3] = 4
	s.Operations["vector"] = 394
	if _, e := s.validate(); e == nil {
		t.Fatal("stages")
	}
	valid := captureRequirement{16, 64, 64, 1}
	if n, e := captureRequirementsTotal([]captureRequirement{valid}, 64); e != nil || n != 64 {
		t.Fatal(n, e)
	}
	if n, e := captureRequirementsTotal([]captureRequirement{valid}, 63); e == nil || n != 64 {
		t.Fatal("budget")
	}
	for _, bad := range []captureRequirement{{0, 64, 64, 1}, {16, 8, 64, 1}, {16, 64, 0, 1}, {16, 64, 3, 1}, {16, 64, 64, 0}} {
		if _, e := captureRequirementsTotal([]captureRequirement{bad}, 64); e == nil {
			t.Fatal("invalid", bad)
		}
	}
	if _, e := captureRequirementsTotal([]captureRequirement{{1, ^uint64(0), 1, 1}, valid}, ^uint64(0)); e == nil {
		t.Fatal("overflow")
	}
	p := t.TempDir() + "/input.json"
	if err := os.WriteFile(p, []byte(`{"Stages":395}`), 0600); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(p)
	sum := sha256.Sum256(raw)
	var got captureSlots
	if e := capturePinnedJSON(p, hex.EncodeToString(sum[:]), &got); e != nil || got.Stages != 395 {
		t.Fatal(e)
	}
	if e := capturePinnedJSON(p, captureSlotPin, &got); e == nil {
		t.Fatal("pin")
	}
	if e := captureWrite(t.TempDir()+"/output.json", got); e != nil {
		t.Fatal(e)
	}
}
func TestVulkanNativeResourceCapture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_VULKAN_NATIVE_RESOURCE_CAPTURE") != "1" {
		t.Skip("explicit bounded resource capture only")
	}
	var slots captureSlots
	if err := capturePinnedJSON(os.Getenv("GO_PHERENCE_VULKAN_CAPTURE_SLOTS"), captureSlotPin, &slots); err != nil {
		t.Fatal(err)
	}
	payload, err := slots.validate()
	if err != nil {
		t.Fatal(err)
	}
	output := os.Getenv("GO_PHERENCE_VULKAN_CAPTURE_OUTPUT")
	if output == "" {
		t.Fatal("exclusive output required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 75*time.Second)
	defer cancel()
	report := struct {
		Scope, Device, Status   string
		Payload, Requirements   uint64
		Limits                  VulkanDeviceLimits
		Rows                    []captureRequirement
		Phases                  []capturePhase
		MemoryFlags, HeapCounts map[uint32]int
		Sets, Bindings, Plans   int
	}{Scope: "Exact saved buffer extents; descriptor-interface stand-in plans NEVER submitted. Zero page touches are resource attribution only; no model arithmetic/update/export.", Payload: payload, MemoryFlags: map[uint32]int{}, HeapCounts: map[uint32]int{}}
	sample := func(name string) {
		p, e := captureSample(name)
		if e != nil {
			t.Fatal(e)
		}
		report.Phases = append(report.Phases, p)
		if p.RSS > 6<<30 {
			t.Fatal("host cap exceeded", p.RSS)
		}
	}
	captureInit(t)
	report.Device = VulkanDeviceName()
	report.Limits, err = VulkanLimits()
	if err != nil {
		t.Fatal(err)
	}
	sample("initialised")
	sizes := append(append([]int64(nil), slots.Resident...), slots.Tape...)
	for _, n := range sizes {
		if err = vkAcquire(ctx); err != nil {
			t.Fatal(err)
		}
		r, e := captureRequirementLocked(uint64(n))
		vkRelease()
		if e != nil {
			t.Fatal(e)
		}
		report.Rows = append(report.Rows, r)
	}
	report.Requirements, err = captureRequirementsTotal(report.Rows, 2<<30)
	sample("requirements-queried-no-memory")
	if err != nil {
		report.Status = "rejected-before-memory-allocation"
		if e := captureWrite(output, report); e != nil {
			t.Fatal(e)
		}
		t.Log(err)
		return
	}
	if err = VulkanSetMemoryBudget(VulkanMemoryBudget{MaxBytes: 2 << 30, MaxAllocations: 876}); err != nil {
		t.Fatal(err)
	}
	op, err := NewVkTrainingF32(ctx)
	if err != nil {
		t.Fatal(err)
	}
	sample("29-training-kernels")
	var arenas []*VkTensorArena
	var tensors []*VkTensorF32
	var plans []*VkF32Plan
	for i, n := range sizes {
		a, e := NewVkTensorArena(ctx, int(n))
		if e != nil {
			t.Fatal("no retry on allocation failure", i, e)
		}
		arenas = append(arenas, a)
		tensor, e := a.AllocF32(ctx, int(n/4))
		if e != nil {
			t.Fatal(e)
		}
		tensors = append(tensors, tensor)
		if err = vkAcquire(ctx); err != nil {
			t.Fatal(err)
		}
		buf := a.state.buffer
		clear(unsafe.Slice((*byte)(buf.mapped), int(n))) // one page touch, no upload/staging
		report.MemoryFlags[buf.memoryFlags]++
		report.HeapCounts[buf.heapIndex]++
		vkRelease()
		if i == 660 {
			sample("resident-pages-touched")
		}
	}
	sample("all-resident-and-tape-pages-touched")
	if u := VulkanMemoryStats(); u.Bytes != report.Requirements || u.Allocations != 876 {
		t.Fatal("native ledger mismatch", u)
	}
	kernels := map[string]*VkComputeKernel{"attention-backward": op.attention, "attention-forward": op.attentionForward, "embedding": op.embedding, "eos": op.eos, "gelu": op.gelu, "linear-backward": op.linearDual, "linear-forward": op.linearDualForward, "norm-backward": op.normDual, "norm-forward": op.normDualForward, "relu": op.relu, "rope": op.rope, "row": op.row, "vector": op.vector}
	names := make([]string, 0, len(slots.Operations))
	for name := range slots.Operations {
		names = append(names, name)
	}
	sort.Strings(names)
	var stages []VkF32Stage
	flush := func() {
		p, e := NewVkF32Plan(ctx, stages)
		if e != nil {
			t.Fatal("plan failure; no retry", e)
		}
		plans = append(plans, p)
		stages = nil
	}
	for _, name := range names {
		k := kernels[name]
		if k == nil {
			t.Fatal("unknown operation", name)
		}
		for i := 0; i < slots.Operations[name]; i++ {
			bindings := make([]vkBufferBinding, k.numBuffers)
			for j := range bindings {
				bindings[j] = vkBufferBinding{tensors[j].arena.buffer, 0, 16}
			}
			stages = append(stages, VkF32Stage{Kernel: k, Groups: [3]uint32{1, 1, 1}, bindings: bindings, PushWords: make([]uint32, k.pushSize/4)})
			report.Sets++
			report.Bindings += k.numBuffers
			if len(stages) == 64 {
				flush()
			}
		}
	}
	if len(stages) > 0 {
		flush()
	}
	report.Plans = len(plans)
	if report.Sets != 395 || report.Bindings != 1841 || report.Plans != 7 {
		t.Fatal("descriptor inventory")
	}
	sample("buffers-kernels-seven-interface-plans")
	for i := len(plans) - 1; i >= 0; i-- {
		if err = plans[i].Close(); err != nil {
			t.Fatal("close failure no retry", err)
		}
	}
	sample("plans-closed")
	for i := len(arenas) - 1; i >= 0; i-- {
		if err = arenas[i].Close(); err != nil {
			t.Fatal("close failure no retry", err)
		}
	}
	if err = op.Close(); err != nil {
		t.Fatal(err)
	}
	sample("all-explicit-owners-closed")
	if u := VulkanMemoryStats(); u.Bytes != 0 || u.Allocations != 0 {
		t.Fatal("retained allocations", u)
	}
	report.Status = "resource-probe-completed-training-unqualified"
	if err = captureWrite(output, report); err != nil {
		t.Fatal(err)
	}
	t.Logf("payload=%d requirements=%d padding=%d descriptors=%d plans=%d", payload, report.Requirements, report.Requirements-payload, report.Sets, report.Plans)
}

// Fixed-input native time arithmetic, independent of host AD/expected values.
type captureTimeRow struct {
	Branch                                                                string `json:"branch"`
	Row, Embedding, FrequencyIndex                                        int
	Time, TimeTangent, Frequency, Angle, Cos, Sin, CosTangent, SinTangent float32
}

func TestVulkanNativeTimeBoundaryCapture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_VULKAN_NATIVE_TIME_CAPTURE") != "1" {
		t.Skip("explicit bounded time capture only")
	}
	testVulkanTimeBoundaryCapture(t, spirvTrainingTimeEmbedding, "Native existing time shader, fixed saved inputs; excludes composed flow/trunk/AD/optimiser and fullstep qualification.")
}

// Shared capture plumbing only; callers select an explicitly pinned shader.
func testVulkanTimeBoundaryCapture(t *testing.T, code []byte, scope string) {
	var input struct{ Rows []captureTimeRow }
	if err := capturePinnedJSON(os.Getenv("GO_PHERENCE_VULKAN_CAPTURE_TIMES"), captureTimePin, &input); err != nil {
		t.Fatal(err)
	}
	if len(input.Rows) != 9984 {
		t.Fatal("time geometry")
	}
	for i, r := range input.Rows {
		if r.FrequencyIndex != i%128 || r.Time*r.Frequency != r.Angle || math.IsNaN(float64(r.Angle)) || math.Abs(float64(r.Angle)) > 1 || r.TimeTangent < 0 || r.TimeTangent > 1 {
			t.Fatal("input boundary", i)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
	defer cancel()
	captureInit(t)
	if err := VulkanSetMemoryBudget(VulkanMemoryBudget{MaxBytes: 8 << 20, MaxAllocations: 512}); err != nil {
		t.Fatal(err)
	}
	k, err := VkKernelCreate(code, 5, 8)
	if err != nil {
		t.Fatal(err)
	}
	op := &VkTrainingF32{timeEmbedding: k}
	type result struct{ Value, Tangent []float32 }
	var results []result
	var arenas []*VkTensorArena
	var stages []VkF32Stage
	var plans []*VkF32Plan
	var outputs [][2]*VkTensorF32
	makeTensor := func(data []float32) *VkTensorF32 {
		a, e := NewVkTensorArena(ctx, 4*len(data))
		if e != nil {
			t.Fatal(e)
		}
		arenas = append(arenas, a)
		x, e := a.AllocF32(ctx, len(data))
		if e != nil {
			t.Fatal(e)
		}
		if data != nil {
			if e = x.Upload(ctx, data); e != nil {
				t.Fatal(e)
			}
		}
		return x
	}
	flush := func() {
		p, e := NewVkF32Plan(ctx, stages)
		if e != nil {
			t.Fatal(e)
		}
		plans = append(plans, p)
		stages = nil
	}
	for offset := 0; offset < len(input.Rows); offset += 128 {
		block := input.Rows[offset : offset+128]
		freq := make([]float32, 128)
		for i, r := range block {
			if r.Row != block[0].Row || r.Embedding != block[0].Embedding || r.Branch != block[0].Branch || r.Time != block[0].Time || r.TimeTangent != block[0].TimeTangent {
				t.Fatal("block input")
			}
			freq[i] = r.Frequency
		}
		f, tm, zero := makeTensor(freq), makeTensor([]float32{block[0].Time, block[0].TimeTangent}), makeTensor(make([]float32, 256))
		var pair [2]*VkTensorF32
		for mode := uint32(0); mode < 2; mode++ {
			pair[mode] = makeTensor(make([]float32, 256))
			s, e := op.TimeEmbeddingStage(ctx, pair[mode], f, tm, zero, zero, mode)
			if e != nil {
				t.Fatal(e)
			}
			stages = append(stages, s)
			if len(stages) == 64 {
				flush()
			}
		}
		outputs = append(outputs, pair)
	}
	if len(stages) > 0 {
		flush()
	}
	before, err := captureSample("before-time-submission")
	if err != nil || before.RSS > 256<<20 {
		t.Fatal("time RSS cap", before, err)
	}
	for _, p := range plans {
		if err = p.Run(ctx); err != nil {
			t.Fatal("submission failed; retain owners, no retry", err)
		}
	}
	var maxValue, maxTangent float64
	var valueChanged, tangentChanged, valueOver, tangentOver int
	var sumValue, sumTangent float64
	for b, pair := range outputs {
		r := result{make([]float32, 256), make([]float32, 256)}
		if err = pair[0].Download(ctx, r.Value); err != nil {
			t.Fatal(err)
		}
		if err = pair[1].Download(ctx, r.Tangent); err != nil {
			t.Fatal(err)
		}
		for j, in := range input.Rows[b*128 : (b+1)*128] {
			for q := 0; q < 2; q++ {
				want, wantT := in.Cos, in.CosTangent
				if q == 1 {
					want, wantT = in.Sin, in.SinTangent
				}
				d, dt := math.Abs(float64(r.Value[128*q+j])-float64(want)), math.Abs(float64(r.Tangent[128*q+j])-float64(wantT))
				if math.IsNaN(d) || math.IsNaN(dt) || math.IsInf(d, 0) || math.IsInf(dt, 0) {
					t.Fatal("nonfinite capture")
				}
				maxValue = math.Max(maxValue, d)
				maxTangent = math.Max(maxTangent, dt)
				sumValue += d * d
				sumTangent += dt * dt
				if d != 0 {
					valueChanged++
				}
				if dt != 0 {
					tangentChanged++
				}
				if d > 6e-8 {
					valueOver++
				}
				if dt > 2e-5 {
					tangentOver++
				}
			}
		}
		results = append(results, r)
	}
	after, err := captureSample("after-time-downloads")
	if err != nil || after.RSS > 256<<20 {
		t.Fatal("time RSS cap", after, err)
	}
	for i := len(plans) - 1; i >= 0; i-- {
		if err = plans[i].Close(); err != nil {
			t.Fatal(err)
		}
	}
	for i := len(arenas) - 1; i >= 0; i-- {
		if err = arenas[i].Close(); err != nil {
			t.Fatal(err)
		}
	}
	if err = k.Close(); err != nil {
		t.Fatal(err)
	}
	closed, err := captureSample("time-owners-closed")
	if err != nil || closed.Native.Bytes != 0 || closed.Native.Allocations != 0 {
		t.Fatal(closed, err)
	}
	report := struct {
		Scope, Device                                                                                  string
		Blocks, ValueCount, TangentCount, ValueChanged, TangentChanged, ValueOverGate, TangentOverGate int
		MaxValue, MaxTangent, RMSValue, RMSTangent                                                     float64
		Phases                                                                                         []capturePhase
		Results                                                                                        []result
	}{scope, VulkanDeviceName(), len(results), 19968, 19968, valueChanged, tangentChanged, valueOver, tangentOver, maxValue, maxTangent, math.Sqrt(sumValue / 19968), math.Sqrt(sumTangent / 19968), []capturePhase{before, after, closed}, results}
	if err = captureWrite(os.Getenv("GO_PHERENCE_VULKAN_CAPTURE_OUTPUT"), report); err != nil {
		t.Fatal(err)
	}
	t.Logf("native time values=%d max=%g RMS=%g above6e-8=%d; tangents max=%g RMS=%g above2e-5=%d", 19968, maxValue, report.RMSValue, valueOver, maxTangent, report.RMSTangent, tangentOver)
	if valueOver != 0 || tangentOver != 0 {
		t.Fatal("unchanged local time gates failed; fullstep gate untouched")
	}
}
