package vulkan

import (
	"encoding/binary"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"unsafe"
)

// Offline tests replace every invoked native function; never call VulkanInit.
// Globals mean these tests MUST NOT use t.Parallel.
func mockVK[T any](t *testing.T, dst *T, value T) {
	t.Helper()
	old := *dst
	*dst = value
	t.Cleanup(func() { *dst = old })
}
func offlineVK(t *testing.T) {
	t.Helper()
	if !vkNative64() {
		t.Skip("current binding is64bit only")
	}
	mockVK(t, &vkPending, (*vkPendingSubmission)(nil))
	mockVK(t, &vkLost, false)
	mockVK(t, &vkReady, true)
	mockVK(t, &vkLimits, offlineLimits())
	mockVK(t, &vkDevice, VkDevice(100))
	mockVK(t, &vkPhysDev, VkPhysicalDevice(101))
	mockVK(t, &vkCmdPool, VkCommandPool(102))
	mockVK(t, &vkQueue, VkQueue(103))
}
func dummySPIRV() []byte {
	data := make([]byte, 20)
	binary.LittleEndian.PutUint32(data, 0x07230203)
	return data
}

func TestVulkanOfflineABILayouts(t *testing.T) {
	if !vkNative64() {
		t.Skip("64bit layout assertions")
	}
	var begin vkCommandBufferBeginInfo
	var req vkMemoryRequirements
	var heap vkMemoryHeap
	var props vkPhysicalDeviceMemoryProperties
	for _, c := range []struct {
		name      string
		got, want uintptr
	}{
		{"begin size", unsafe.Sizeof(begin), 32}, {"begin pNext", unsafe.Offsetof(begin.pNext), 8}, {"begin flags", unsafe.Offsetof(begin.flags), 16}, {"begin inheritance", unsafe.Offsetof(begin.pInheritanceInfo), 24},
		{"req size", unsafe.Sizeof(req), 24}, {"req typebits", unsafe.Offsetof(req.memoryTypeBits), 16}, {"heap size", unsafe.Sizeof(heap), 16}, {"heap flags", unsafe.Offsetof(heap.flags), 8},
		{"props size", unsafe.Sizeof(props), 520}, {"props typearray", unsafe.Offsetof(props.memoryTypes), 4}, {"props heapcount", unsafe.Offsetof(props.memoryHeapCount), 260}, {"props heaps", unsafe.Offsetof(props.memoryHeaps), 264},
	} {
		if c.got != c.want {
			t.Errorf("%s=%d want%d", c.name, c.got, c.want)
		}
	}
}

func TestVulkanOfflineKernelRollback(t *testing.T) {
	stages := []string{"shader", "setlayout", "pipeline-layout", "pipeline", "pool", "set", "command", "fence", "success", "partial-pipeline"}
	for _, failure := range stages {
		t.Run(failure, func(t *testing.T) {
			offlineVK(t)
			var calls, freed []string
			result := func(name string) VkResult {
				calls = append(calls, name)
				if name == failure || name == "pipeline" && failure == "partial-pipeline" {
					return -1
				}
				return VK_SUCCESS
			}
			device := func(d VkDevice) {
				if d != 100 {
					t.Fatal("owner device changed", d)
				}
			}
			mockVK(t, &vkCreateShaderModule, func(d VkDevice, p, allocator unsafe.Pointer, out *VkShaderModule) VkResult {
				device(d)
				code := *(*unsafe.Pointer)(unsafe.Add(p, 32))
				if uintptr(code)%4 != 0 || *(*uint32)(code) != 0x07230203 {
					t.Fatal("unaligned/invalid code")
				}
				*out = 1
				return result("shader")
			})
			mockVK(t, &vkCreateDescriptorSetLayout, func(d VkDevice, p, a unsafe.Pointer, out *VkDescriptorSetLayout) VkResult {
				device(d)
				*out = 2
				return result("setlayout")
			})
			mockVK(t, &vkCreatePipelineLayout, func(d VkDevice, p, a unsafe.Pointer, out *VkPipelineLayout) VkResult {
				device(d)
				*out = 3
				return result("pipeline-layout")
			})
			mockVK(t, &vkCreateComputePipelines, func(d VkDevice, cache uintptr, n uint32, p, a unsafe.Pointer, out *VkPipeline) VkResult {
				device(d)
				*out = 4
				r := result("pipeline")
				if failure == "pipeline" {
					*out = 0
				}
				return r
			})
			mockVK(t, &vkCreateDescriptorPool, func(d VkDevice, p, a unsafe.Pointer, out *VkDescriptorPool) VkResult {
				device(d)
				*out = 5
				return result("pool")
			})
			mockVK(t, &vkAllocateDescriptorSets, func(d VkDevice, p unsafe.Pointer, out *VkDescriptorSet) VkResult {
				device(d)
				*out = 6
				return result("set")
			})
			mockVK(t, &vkAllocateCommandBuffers, func(d VkDevice, p unsafe.Pointer, out *VkCommandBuffer) VkResult {
				device(d)
				if *(*VkCommandPool)(unsafe.Add(p, 16)) != 102 {
					t.Fatal("commandpool owner")
				}
				*out = 7
				return result("command")
			})
			mockVK(t, &vkCreateFence, func(d VkDevice, p, a unsafe.Pointer, out *VkFence) VkResult {
				device(d)
				*out = 8
				return result("fence")
			})
			mockVK(t, &vkDestroyShaderModule, func(d VkDevice, h VkShaderModule, p unsafe.Pointer) {
				device(d)
				if h != 1 {
					t.Fatal(h)
				}
				freed = append(freed, "shader")
			})
			mockVK(t, &vkDestroyDescriptorSetLayout, func(d VkDevice, h VkDescriptorSetLayout, p unsafe.Pointer) {
				device(d)
				freed = append(freed, "setlayout")
			})
			mockVK(t, &vkDestroyPipelineLayout, func(d VkDevice, h VkPipelineLayout, p unsafe.Pointer) {
				device(d)
				freed = append(freed, "pipeline-layout")
			})
			mockVK(t, &vkDestroyPipeline, func(d VkDevice, h VkPipeline, p unsafe.Pointer) { device(d); freed = append(freed, "pipeline") })
			mockVK(t, &vkDestroyDescriptorPool, func(d VkDevice, h VkDescriptorPool, p unsafe.Pointer) { device(d); freed = append(freed, "pool") })
			mockVK(t, &vkFreeCommandBuffers, func(d VkDevice, pool VkCommandPool, n uint32, h *VkCommandBuffer) {
				device(d)
				if pool != 102 || n != 1 || *h != 7 {
					t.Fatal("free command owner")
				}
				freed = append(freed, "command")
			})
			mockVK(t, &vkDestroyFence, func(d VkDevice, h VkFence, p unsafe.Pointer) { device(d); freed = append(freed, "fence") })
			// Intentionally unaligned input storage must be copied before passing pCode.
			input := append([]byte{0}, dummySPIRV()...)
			kernel, err := VkKernelCreate(input[1:], 2, 8)
			expected := map[string][]string{"shader": nil, "setlayout": {"shader"}, "pipeline-layout": {"setlayout", "shader"}, "pipeline": {"pipeline-layout", "setlayout", "shader"}, "partial-pipeline": {"pipeline", "pipeline-layout", "setlayout", "shader"}, "pool": {"pipeline", "pipeline-layout", "setlayout", "shader"}, "set": {"pool", "pipeline", "pipeline-layout", "setlayout", "shader"}, "command": {"pool", "pipeline", "pipeline-layout", "setlayout", "shader"}, "fence": {"command", "pool", "pipeline", "pipeline-layout", "setlayout", "shader"}, "success": {"shader"}}[failure]
			if failure == "success" {
				if err != nil || kernel == nil {
					t.Fatal(err)
				}
			} else if err == nil || kernel != nil {
				t.Fatal("failure accepted", failure, err)
			}
			if !reflect.DeepEqual(freed, expected) {
				t.Fatalf("rollback got%v want%v calls%v", freed, expected, calls)
			}
			if failure == "success" {
				if kernel.device != 100 || kernel.commandPool != 102 || kernel.queue != 103 {
					t.Fatal("missing captured owners")
				}
				if err := kernel.Close(); err != nil {
					t.Fatal(err)
				}
				if err := kernel.Close(); err != nil {
					t.Fatal(err)
				}
				want := []string{"shader", "fence", "command", "pool", "pipeline", "pipeline-layout", "setlayout"}
				if !reflect.DeepEqual(freed, want) {
					t.Fatal("successful kernel cleanup", freed)
				}
			}
		})
	}
}

func TestVulkanOfflineKernelPreflight(t *testing.T) {
	offlineVK(t)
	calls := 0
	mockVK(t, &vkCreateShaderModule, func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkShaderModule) VkResult { calls++; return -1 })
	mockVK(t, &vkDestroyShaderModule, (func(VkDevice, VkShaderModule, unsafe.Pointer))(nil))
	for _, c := range []struct {
		code    []byte
		n, push int
	}{{dummySPIRV(), 1, 0}, {dummySPIRV(), 17, 0}, {dummySPIRV(), 1, 129}, {dummySPIRV(), 1, 3}, {[]byte{0, 0, 0, 0}, 1, 0}} {
		if k, err := VkKernelCreate(c.code, c.n, c.push); err == nil || k != nil {
			t.Fatal("preflight accepted")
		}
	}
	if calls != 0 {
		t.Fatal("preflight allocated")
	}
}

func TestVulkanOfflineBufferRollback(t *testing.T) {
	for _, failure := range []string{"buffer", "requirements", "typecount", "heapcount", "heapindex", "no-type", "small-heap", "allocate", "bind", "map", "nil-map", "success"} {
		t.Run(failure, func(t *testing.T) {
			offlineVK(t)
			var freed []string
			storage := make([]byte, 16)
			fail := func(name string) VkResult {
				if failure == name {
					return -2
				}
				return VK_SUCCESS
			}
			mockVK(t, &vkCreateBuffer, func(d VkDevice, p, a unsafe.Pointer, out *VkBuffer) VkResult { *out = 11; return fail("buffer") })
			mockVK(t, &vkGetBufferMemoryRequirements, func(d VkDevice, b VkBuffer, p unsafe.Pointer) {
				r := (*vkMemoryRequirements)(p)
				*r = vkMemoryRequirements{64, 16, 1}
				if failure == "requirements" {
					r.size = 1
				}
			})
			mockVK(t, &vkGetPhysicalDeviceMemoryProperties, func(d VkPhysicalDevice, p unsafe.Pointer) {
				v := (*vkPhysicalDeviceMemoryProperties)(p)
				v.memoryTypeCount = 1
				v.memoryHeapCount = 1
				v.memoryTypes[0] = vkMemoryType{6, 0}
				v.memoryHeaps[0] = vkMemoryHeap{4096, 0}
				switch failure {
				case "typecount":
					v.memoryTypeCount = 33
				case "heapcount":
					v.memoryHeapCount = 17
				case "heapindex":
					v.memoryTypes[0].heapIndex = 1
				case "no-type":
					v.memoryTypes[0].propertyFlags = 1
				case "small-heap":
					v.memoryHeaps[0].size = 32
				}
			})
			mockVK(t, &vkAllocateMemory, func(d VkDevice, p, a unsafe.Pointer, out *VkDeviceMemory) VkResult {
				*out = 12
				return fail("allocate")
			})
			mockVK(t, &vkBindBufferMemory, func(VkDevice, VkBuffer, VkDeviceMemory, uint64) VkResult { return fail("bind") })
			mockVK(t, &vkMapMemory, func(d VkDevice, m VkDeviceMemory, o, n uint64, f uint32, out *unsafe.Pointer) VkResult {
				if failure != "nil-map" {
					*out = unsafe.Pointer(&storage[0])
				}
				return fail("map")
			})
			mockVK(t, &vkUnmapMemory, func(VkDevice, VkDeviceMemory) { freed = append(freed, "unmap") })
			mockVK(t, &vkDestroyBuffer, func(d VkDevice, h VkBuffer, p unsafe.Pointer) {
				if d != 100 || h != 11 {
					t.Fatal("buffer owner")
				}
				freed = append(freed, "buffer")
			})
			mockVK(t, &vkFreeMemory, func(d VkDevice, h VkDeviceMemory, p unsafe.Pointer) {
				if d != 100 || h != 12 {
					t.Fatal("memory owner")
				}
				freed = append(freed, "memory")
			})
			b, err := VkBufAlloc(16)
			if failure == "success" {
				if err != nil || b == nil || len(freed) != 0 {
					t.Fatal(err, freed)
				}
				if err := b.UploadChecked([]float32{1, 2, 3, 4}); err != nil {
					t.Fatal(err)
				}
				b.Free()
				b.Free()
				if !reflect.DeepEqual(freed, []string{"unmap", "buffer", "memory"}) {
					t.Fatal("successful free", freed)
				}
				return
			}
			if err == nil || b != nil {
				t.Fatal("failure accepted", failure)
			}
			expected := []string{"buffer"}
			if failure == "buffer" {
				expected = nil
			}
			if failure == "bind" || failure == "map" || failure == "nil-map" {
				expected = []string{"buffer", "memory"}
			}
			if !reflect.DeepEqual(freed, expected) {
				t.Fatal("buffer rollback", freed, expected)
			}
		})
	}
}

func TestVulkanOfflineDispatchPreflight(t *testing.T) {
	offlineVK(t)
	calls := 0
	mockVK(t, &vkUpdateDescriptorSets, func(VkDevice, uint32, unsafe.Pointer, uint32, unsafe.Pointer) { calls++ })
	mockVK(t, &vkCmdPushConstants, (func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer))(nil))
	kernel := &VkComputeKernel{device: 100, queue: 103, commandPool: 102, pipeline: 1, pipelineLayout: 2, descSet: 3, cmdBuf: 4, fence: 5, numBuffers: 1, pushSize: 4}
	bufs := []*VkBuf{{device: 100, buf: 1, mem: 2, size: 4}}
	if err := kernel.Dispatch(1, 1, 1, bufs, nil); err == nil || !strings.Contains(err.Error(), "push constants") {
		t.Fatal("missingpush", err)
	}
	var value uint32
	if err := kernel.Dispatch(1, 1, 1, bufs, unsafe.Pointer(&value)); err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Fatal("nilpushfunction", err)
	}
	if calls != 0 {
		t.Fatal("dispatch wrote before validation")
	}
}

func TestVulkanOfflineDispatchBeginAndFenceErrors(t *testing.T) {
	for _, failure := range []string{"begin", "end", "reset", "submit", "wait", "success"} {
		t.Run(failure, func(t *testing.T) {
			offlineVK(t)
			var calls []string
			step := func(name string) VkResult {
				calls = append(calls, name)
				if failure == name {
					return -3
				}
				return VK_SUCCESS
			}
			mockVK(t, &vkUpdateDescriptorSets, func(VkDevice, uint32, unsafe.Pointer, uint32, unsafe.Pointer) { calls = append(calls, "descriptors") })
			mockVK(t, &vkBeginCommandBuffer, func(c VkCommandBuffer, p unsafe.Pointer) VkResult {
				v := (*vkCommandBufferBeginInfo)(p)
				if v.sType != 42 || v.pNext != 0 || v.flags != 1 || v.pInheritanceInfo != nil {
					t.Fatal("begin ABI", v)
				}
				return step("begin")
			})
			mockVK(t, &vkCmdBindPipeline, func(VkCommandBuffer, uint32, VkPipeline) {})
			mockVK(t, &vkCmdBindDescriptorSets, func(VkCommandBuffer, uint32, VkPipelineLayout, uint32, uint32, *VkDescriptorSet, uint32, unsafe.Pointer) {
			})
			mockVK(t, &vkCmdPushConstants, func(cmd VkCommandBuffer, layout VkPipelineLayout, stages, offset, size uint32, p unsafe.Pointer) {
				if cmd != 4 || layout != 2 || stages != 0x20 || offset != 0 || size != 4 || *(*uint32)(p) != 17 {
					t.Fatal("push constant ABI")
				}
				calls = append(calls, "push")
			})
			mockVK(t, &vkCmdDispatch, func(VkCommandBuffer, uint32, uint32, uint32) { calls = append(calls, "dispatch") })
			mockVK(t, &vkCmdPipelineBarrier, func(VkCommandBuffer, uint32, uint32, uint32, uint32, unsafe.Pointer, uint32, unsafe.Pointer, uint32, unsafe.Pointer) {
				calls = append(calls, "barrier")
			})
			mockVK(t, &vkEndCommandBuffer, func(VkCommandBuffer) VkResult { return step("end") })
			mockVK(t, &vkResetFences, func(VkDevice, uint32, *VkFence) VkResult { return step("reset") })
			mockVK(t, &vkQueueSubmit, func(VkQueue, uint32, unsafe.Pointer, VkFence) VkResult { return step("submit") })
			mockVK(t, &vkWaitForFences, func(VkDevice, uint32, *VkFence, uint32, uint64) VkResult { return step("wait") })
			kernel := &VkComputeKernel{device: 100, queue: 103, commandPool: 102, pipeline: 1, pipelineLayout: 2, descSet: 3, cmdBuf: 4, fence: 5, numBuffers: 1, pushSize: 4}
			push := uint32(17)
			err := kernel.Dispatch(1, 1, 1, []*VkBuf{{device: 100, buf: 1, mem: 2, size: 4}}, unsafe.Pointer(&push))
			if failure == "success" {
				if err != nil {
					t.Fatal(err)
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), "-3") {
					t.Fatal("driver cause", err)
				}
				if calls[len(calls)-1] != failure {
					t.Fatal("continued after failure", fmt.Sprint(calls))
				}
			}
			barriers := 0
			for _, call := range calls {
				if call == "barrier" {
					barriers++
				}
			}
			want := 2
			if failure == "begin" {
				want = 0
			}
			if barriers != want {
				t.Fatalf("barriers=%d want%d after%s", barriers, want, failure)
			}
		})
	}
}
