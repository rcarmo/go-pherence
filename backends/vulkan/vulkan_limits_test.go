package vulkan

import (
	"context"
	"encoding/binary"
	"reflect"
	"strconv"
	"testing"
	"unsafe"
)

func offlineProperties() vkDeviceProperties {
	return vkDeviceProperties{apiVersion: 1<<22 | 3<<12, limits: vkPhysicalDeviceLimits{
		maxStorageBufferRange: 128 << 20, maxPushConstantsSize: 128, maxBoundDescriptorSets: 4, maxMemoryAllocationCount: 4096,
		maxPerStageDescriptorStorageBuffers: 16, maxPerStageResources: 128, maxDescriptorSetStorageBuffers: 16,
		maxComputeSharedMemorySize: 16384, maxComputeWorkGroupCount: [3]uint32{65535, 65535, 65535},
		maxComputeWorkGroupInvocations: 256, maxComputeWorkGroupSize: [3]uint32{256, 256, 64}}}
}
func offlineLimits() VulkanDeviceLimits {
	p := offlineProperties()
	l, err := vkLimitsFromProperties(p)
	if err != nil {
		panic(err)
	}
	return l
}

func TestVulkanOfflineLimitsABI(t *testing.T) {
	if !vkNative64() {
		t.Skip("current binding is64bit only")
	}
	var p vkDeviceProperties
	var l vkPhysicalDeviceLimits
	for _, c := range []struct {
		name      string
		got, want uintptr
	}{
		{"properties size", unsafe.Sizeof(p), 824}, {"properties align", unsafe.Alignof(p), 8},
		{"name", unsafe.Offsetof(p.deviceName), 20}, {"uuid", unsafe.Offsetof(p.pipelineCacheUUID), 276},
		{"limits", unsafe.Offsetof(p.limits), 296}, {"sparse", unsafe.Offsetof(p.sparseProperties), 800},
		{"limits size", unsafe.Sizeof(l), 504}, {"storage", unsafe.Offsetof(l.maxStorageBufferRange), 28},
		{"allocation count", unsafe.Offsetof(l.maxMemoryAllocationCount), 36}, {"push", unsafe.Offsetof(l.maxPushConstantsSize), 32}, {"sets", unsafe.Offsetof(l.maxBoundDescriptorSets), 64},
		{"stage storage", unsafe.Offsetof(l.maxPerStageDescriptorStorageBuffers), 76},
		{"stage resources", unsafe.Offsetof(l.maxPerStageResources), 92},
		{"set storage", unsafe.Offsetof(l.maxDescriptorSetStorageBuffers), 108},
		{"shared", unsafe.Offsetof(l.maxComputeSharedMemorySize), 216}, {"group count", unsafe.Offsetof(l.maxComputeWorkGroupCount), 220},
		{"invocations", unsafe.Offsetof(l.maxComputeWorkGroupInvocations), 232}, {"group size", unsafe.Offsetof(l.maxComputeWorkGroupSize), 236},
		{"storage align", unsafe.Offsetof(l.minStorageBufferOffsetAlignment), 328}, {"noncoherent", unsafe.Offsetof(l.nonCoherentAtomSize), 496},
	} {
		if c.got != c.want {
			t.Errorf("%s=%d want%d", c.name, c.got, c.want)
		}
	}
	// Independent raw-offset fixture: exact C64 wire locations, distinct values.
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&p)), 824)
	values := map[int]uint32{0: 1<<22 | 3<<12, 324: 101, 328: 102, 332: 115, 360: 103, 372: 104, 388: 105, 404: 106, 512: 107, 516: 108, 520: 109, 524: 110, 528: 111, 532: 112, 536: 113, 540: 114}
	for off, value := range values {
		binary.LittleEndian.PutUint32(bytes[off:off+4], value)
	}
	got, err := vkLimitsFromProperties(p)
	if err != nil {
		t.Fatal(err)
	}
	want := VulkanDeviceLimits{APIVersion: 1<<22 | 3<<12, StorageBufferRange: 101, MemoryAllocationCount: 115, PushConstantBytes: 102, BoundDescriptorSets: 103, PerStageStorageBuffers: 104, PerStageResources: 105, DescriptorSetStorageBuffers: 106, SharedMemoryBytes: 107, WorkgroupCount: [3]uint32{108, 109, 110}, WorkgroupInvocations: 111, WorkgroupSize: [3]uint32{112, 113, 114}}
	if got != want {
		t.Fatalf("driver decode %+v want%+v", got, want)
	}
}

func TestVulkanOfflineLimitsSnapshot(t *testing.T) {
	offlineVK(t)
	got, err := VulkanLimits()
	if err != nil || got != offlineLimits() {
		t.Fatal(got, err)
	}
	got.WorkgroupCount[0] = 1
	got.StorageBufferRange = 1
	again, err := VulkanLimits()
	if err != nil || again != offlineLimits() {
		t.Fatal("mutable snapshot", err)
	}
	vkPending = &vkPendingSubmission{}
	if _, err := VulkanLimits(); err != nil {
		t.Fatal("ordinary pending hides limits", err)
	}
	vkPending.uncertain = true
	_, err = VulkanLimits()
	expectErrorIs(t, err, ErrVulkanUncertain)
	vkPending = nil
	vkLost = true
	_, err = VulkanLimits()
	expectErrorIs(t, err, ErrVulkanDeviceLost)
	vkLost = false
	vkReady = false
	_, err = VulkanLimits()
	expectErrorIs(t, err, ErrVulkanLimit)
}

func TestVulkanOfflineLimitsRejectIncomplete(t *testing.T) {
	for _, c := range []struct {
		name   string
		change func(*VulkanDeviceLimits)
	}{
		{"api", func(l *VulkanDeviceLimits) { l.APIVersion = 1<<22 | 2<<12 }},
		{"variant", func(l *VulkanDeviceLimits) { l.APIVersion |= 1 << 29 }},
		{"range", func(l *VulkanDeviceLimits) { l.StorageBufferRange = 0 }},
		{"allocation count", func(l *VulkanDeviceLimits) { l.MemoryAllocationCount = 0 }},
		{"push", func(l *VulkanDeviceLimits) { l.PushConstantBytes = 0 }},
		{"sets", func(l *VulkanDeviceLimits) { l.BoundDescriptorSets = 0 }},
		{"stage storage", func(l *VulkanDeviceLimits) { l.PerStageStorageBuffers = 0 }},
		{"resources", func(l *VulkanDeviceLimits) { l.PerStageResources = 0 }},
		{"set storage", func(l *VulkanDeviceLimits) { l.DescriptorSetStorageBuffers = 0 }},
		{"groups", func(l *VulkanDeviceLimits) { l.WorkgroupCount[2] = 0 }},
		{"local", func(l *VulkanDeviceLimits) { l.WorkgroupSize[1] = 0 }},
		{"invocations", func(l *VulkanDeviceLimits) { l.WorkgroupInvocations = 0 }},
		{"shared", func(l *VulkanDeviceLimits) { l.SharedMemoryBytes = 0 }},
	} {
		t.Run(c.name, func(t *testing.T) { l := offlineLimits(); c.change(&l); expectErrorIs(t, l.validate(), ErrVulkanLimit) })
	}
}

func TestVulkanOfflineLimitsPipelineAdmission(t *testing.T) {
	for _, field := range []string{"stage", "resources", "set", "push"} {
		t.Run(field, func(t *testing.T) {
			offlineVK(t)
			calls := 0
			mockVK(t, &vkCreateShaderModule, func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkShaderModule) VkResult { calls++; return -1 })
			n, push := 2, 0
			switch field {
			case "stage":
				vkLimits.PerStageStorageBuffers = 1
			case "resources":
				vkLimits.PerStageResources = 1
			case "set":
				vkLimits.DescriptorSetStorageBuffers = 1
			case "push":
				vkLimits.PushConstantBytes = 4
				n = 1
				push = 8
			}
			_, err := VkKernelCreate(dummySPIRV(), n, push)
			expectErrorIs(t, err, ErrVulkanLimit)
			if field != "push" {
				_, err = LoadSPIRV(dummySPIRV(), n)
				expectErrorIs(t, err, ErrVulkanLimit)
			}
			if calls != 0 {
				t.Fatal("created shader before limit rejection")
			}
			if err := vkCheckPipelineLimitsLocked(1, 4); err != nil {
				t.Fatal("exact/under limits rejected", err)
			}
		})
	}
}

func TestVulkanOfflineLimitsBufferAdmission(t *testing.T) {
	offlineVK(t)
	vkLimits.StorageBufferRange = 16
	calls := 0
	mockVK(t, &vkCreateBuffer, func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkBuffer) VkResult { calls++; return -1 })
	_, err := VkBufAlloc(17)
	expectErrorIs(t, err, ErrVulkanLimit)
	_, err = VkBufAlloc(int(^uint(0) >> 1))
	expectErrorIs(t, err, ErrVulkanLimit)
	if calls != 0 {
		t.Fatal("created oversized buffer")
	}
	if err := vkCheckBufferLimitLocked(16); err != nil {
		t.Fatal(err)
	}
}

func TestVulkanOfflineLimitsDispatchAdmission(t *testing.T) {
	for _, axis := range []int{0, 1, 2, 3} {
		t.Run(strconv.Itoa(axis), func(t *testing.T) {
			m, k, b := newLifetimeMock(t)
			vkLimits.WorkgroupCount = [3]uint32{2, 3, 4}
			vkLimits.StorageBufferRange = 16
			groups := [3]uint32{2, 3, 4}
			if axis < 3 {
				groups[axis]++
			} else {
				b.size = 17
			}
			expectErrorIs(t, k.Dispatch(groups[0], groups[1], groups[2], []*VkBuf{b}, nil), ErrVulkanLimit)
			if len(m.events) != 0 || vkPending != nil {
				t.Fatal("dispatch mutated before limit check")
			}
			b.size = 16
			if err := k.Dispatch(2, 3, 4, []*VkBuf{b}, nil); err != nil {
				t.Fatal("exact group/range limit rejected", err)
			}
		})
	}
}

func TestVulkanOfflineLimitsExplicitDescriptorRange(t *testing.T) {
	_, k, b := newLifetimeMock(t)
	mockVK(t, &vkUpdateDescriptorSets, func(d VkDevice, n uint32, p unsafe.Pointer, c uint32, q unsafe.Pointer) {
		if d != 100 || n != 1 || c != 0 || q != nil {
			t.Fatal("descriptor ABI")
		}
		info := *(*unsafe.Pointer)(unsafe.Add(p, 48))
		if *(*VkBuffer)(info) != b.buf || *(*uint64)(unsafe.Add(info, 8)) != 0 || *(*uint64)(unsafe.Add(info, 16)) != 16 {
			t.Fatal("descriptor not exact bounded range")
		}
	})
	if err := k.Dispatch(1, 1, 1, []*VkBuf{b}, nil); err != nil {
		t.Fatal(err)
	}
}

func TestVulkanOfflineLimitsInitPublication(t *testing.T) {
	for _, mode := range []string{"valid", "skip-discrete", "none", "pool-failure"} {
		t.Run(mode, func(t *testing.T) {
			offlineInitState(t)
			loader, events := initMock(t, "")
			base := loader.bind
			loader.bind = func(target any, lib uintptr, name string) error {
				if err := base(target, lib, name); err != nil {
					return err
				}
				if name == "vkGetPhysicalDeviceProperties" {
					original := vkGetPhysicalDeviceProperties
					reflect.ValueOf(target).Elem().Set(reflect.ValueOf(func(d VkPhysicalDevice, p unsafe.Pointer) {
						original(d, p)
						v := (*vkDeviceProperties)(p)
						v.limits.maxStorageBufferRange = 100 + uint32(d)
						if mode == "none" || (mode == "skip-discrete" && d == 12) {
							v.apiVersion = 1<<22 | 2<<12
						}
					}))
				}
				if name == "vkCreateCommandPool" && mode == "pool-failure" {
					vkCreateCommandPool = func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkCommandPool) VkResult { return -1 }
				}
				return nil
			}
			success := runInitMock(loader)
			if mode == "none" || mode == "pool-failure" {
				if success || vkLimits != (VulkanDeviceLimits{}) {
					t.Fatal("partial limit publication")
				}
				if mode == "none" {
					for _, s := range *events {
						if s == "device" {
							t.Fatal("created with invalid limits")
						}
					}
				}
				return
			}
			if !success {
				t.Fatal("valid init rejected")
			}
			want := uint32(112)
			if mode == "skip-discrete" {
				want = 111
			}
			got, err := VulkanLimits()
			if err != nil || got.StorageBufferRange != want {
				t.Fatal("wrong device limits", got, err)
			}
		})
	}
}

// Keep cooperative admission state usable after a limit rejection.
func TestVulkanOfflineLimitsDoesNotQuarantine(t *testing.T) {
	_, k, b := newLifetimeMock(t)
	vkLimits.WorkgroupCount[0] = 1
	expectErrorIs(t, k.DispatchContext(context.Background(), 2, 1, 1, []*VkBuf{b}, nil), ErrVulkanLimit)
	if vkPending != nil || vkLost {
		t.Fatal("host rejection quarantined device")
	}
	if err := k.Dispatch(1, 1, 1, []*VkBuf{b}, nil); err != nil {
		t.Fatal(err)
	}
}
