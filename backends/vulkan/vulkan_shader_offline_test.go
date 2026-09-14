package vulkan

import (
	"reflect"
	"testing"
	"unsafe"
)

func TestVulkanOfflineLegacyShaderRollbackAndClose(t *testing.T) {
	for _, failure := range []string{"shader", "setlayout", "layout", "pipeline", "partial-pipeline", "success"} {
		t.Run(failure, func(t *testing.T) {
			offlineVK(t)
			var freed []string
			result := func(name string) VkResult {
				if name == failure || (name == "pipeline" && failure == "partial-pipeline") {
					return -1
				}
				return VK_SUCCESS
			}
			owner := func(d VkDevice) {
				if d != 100 {
					t.Error("shader owner")
				}
			}
			mockVK(t, &vkCreateShaderModule, func(d VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
				owner(d)
				code := *(*unsafe.Pointer)(unsafe.Add(p, 32))
				if uintptr(code)%4 != 0 || *(*uint32)(code) != 0x07230203 {
					t.Fatal("SPIRV alignment")
				}
				*out = 1
				return result("shader")
			})
			mockVK(t, &vkCreateDescriptorSetLayout, func(d VkDevice, p, a unsafe.Pointer, out *VkDescriptorSetLayout) VkResult {
				owner(d)
				if *(*uint32)(unsafe.Add(p, 20)) != 2 {
					t.Error("binding count")
				}
				*out = 2
				return result("setlayout")
			})
			mockVK(t, &vkCreatePipelineLayout, func(d VkDevice, p, a unsafe.Pointer, out *VkPipelineLayout) VkResult {
				owner(d)
				if *(*uint32)(unsafe.Add(p, 20)) != 1 || *(*uint32)(unsafe.Add(p, 32)) != 0 || *(*uintptr)(unsafe.Add(p, 40)) != 0 {
					t.Error("pipeline layout ABI")
				}
				*out = 3
				return result("layout")
			})
			mockVK(t, &vkCreateComputePipelines, func(d VkDevice, cache uintptr, n uint32, p, a unsafe.Pointer, out *VkPipeline) VkResult {
				owner(d)
				// VkComputePipelineCreateInfo: stage offset24/size48, layout offset72.
				if n != 1 || *(*uint32)(p) != 29 || *(*uint32)(unsafe.Add(p, 24)) != 18 || *(*uint32)(unsafe.Add(p, 44)) != 0x20 || *(*VkShaderModule)(unsafe.Add(p, 48)) != 1 || *(*VkPipelineLayout)(unsafe.Add(p, 72)) != 3 {
					t.Error("compute stage ABI")
				}
				name := *(*unsafe.Pointer)(unsafe.Add(p, 56))
				if string(unsafe.Slice((*byte)(name), 5)) != "main\x00" {
					t.Error("entrypoint")
				}
				*out = 4
				if failure == "pipeline" {
					*out = 0
				}
				return result("pipeline")
			})
			mockVK(t, &vkDestroyShaderModule, func(d VkDevice, h VkShaderModule, p unsafe.Pointer) {
				owner(d)
				if h != 1 {
					t.Error("shader handle")
				}
				freed = append(freed, "shader")
			})
			mockVK(t, &vkDestroyDescriptorSetLayout, func(d VkDevice, h VkDescriptorSetLayout, p unsafe.Pointer) {
				owner(d)
				if h != 2 {
					t.Error("descriptor handle")
				}
				freed = append(freed, "setlayout")
			})
			mockVK(t, &vkDestroyPipelineLayout, func(d VkDevice, h VkPipelineLayout, p unsafe.Pointer) {
				owner(d)
				if h != 3 {
					t.Error("layout handle")
				}
				freed = append(freed, "layout")
			})
			mockVK(t, &vkDestroyPipeline, func(d VkDevice, h VkPipeline, p unsafe.Pointer) {
				owner(d)
				if h != 4 {
					t.Error("pipeline handle")
				}
				freed = append(freed, "pipeline")
			})
			input := append([]byte{0}, dummySPIRV()...)
			s, err := LoadSPIRV(input[1:], 2)
			expected := map[string][]string{"shader": nil, "setlayout": {"shader"}, "layout": {"setlayout", "shader"}, "pipeline": {"layout", "setlayout", "shader"}, "partial-pipeline": {"pipeline", "layout", "setlayout", "shader"}, "success": {"shader"}}[failure]
			if !reflect.DeepEqual(freed, expected) {
				t.Fatalf("rollback got%v want%v", freed, expected)
			}
			if failure != "success" {
				if err == nil || s != nil {
					t.Fatal("failure accepted")
				}
				return
			}
			if err != nil || s == nil || s.device != 100 {
				t.Fatal("successful owner", err)
			}
			s.device = 999
			if err := s.Close(); err == nil {
				t.Fatal("stale owner accepted")
			}
			s.device = 100
			vkLost = true
			expectErrorIs(t, s.Close(), ErrVulkanDeviceLost)
			vkLost = false
			vkPending = &vkPendingSubmission{uncertain: true}
			expectErrorIs(t, s.Close(), ErrVulkanUncertain)
			vkPending = nil
			if len(freed) != 1 {
				t.Fatal("quarantine destroyed shader")
			}
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			if !s.closed || s.pipeline != 0 || s.pipelineLayout != 0 || s.descSetLayout != 0 || !reflect.DeepEqual(freed, []string{"shader", "pipeline", "layout", "setlayout"}) {
				t.Fatal("close", freed)
			}
		})
	}
}
func TestVulkanOfflineLegacyShaderGuards(t *testing.T) {
	offlineVK(t)
	calls := 0
	mockVK(t, &vkCreateShaderModule, func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkShaderModule) VkResult { calls++; return -1 })
	mockVK(t, &vkDestroyShaderModule, (func(VkDevice, VkShaderModule, unsafe.Pointer))(nil))
	for _, c := range []struct {
		code []byte
		n    int
	}{{nil, 1}, {[]byte{1, 2, 3}, 1}, {make([]byte, 20), 1}, {dummySPIRV(), 0}, {dummySPIRV(), 17}, {dummySPIRV(), 1}} {
		if s, err := LoadSPIRV(c.code, c.n); err == nil || s != nil {
			t.Fatal("invalid shader accepted")
		}
	}
	if calls != 0 {
		t.Fatal("allocated before preflight")
	}
	if err := (*VkComputeShader)(nil).Close(); err != nil {
		t.Fatal(err)
	}
}
