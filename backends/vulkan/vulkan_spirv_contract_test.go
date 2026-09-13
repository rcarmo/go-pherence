package vulkan

import (
	"encoding/binary"
	"errors"
	"testing"
	"unsafe"
)

func spirvTestBytes(words []uint32) []byte {
	b := make([]byte, 4*len(words))
	for i, w := range words {
		binary.LittleEndian.PutUint32(b[4*i:], w)
	}
	return b
}
func spirvTestWords(b []byte) []uint32 {
	w := make([]uint32, len(b)/4)
	for i := range w {
		w[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return w
}
func contractShaders() map[string][]byte {
	return map[string][]byte{
		"linear_regtile": spirv_linear_f32_regtile, "linear_f16_weight": spirv_linear_f16_weight_f32, "conv1d3": spirv_conv1d3_f32, "conv2d_chw": spirv_conv2d_chw_f32, "channel_affine": spirv_channel_affine_relu_f32, "attention_f32": spirv_attention_f32, "gelu_erf": spirv_gelu_erf_f32, "linear": spirv_linear_f32, "lstm_cell": spirv_lstm_cell_f32, "lstm_sequence": spirv_lstm_sequence_f32, "layernorm": spirv_layer_norm_f32, "attention": spirv_attention_score, "gemv_f32": spirv_gemv_f32, "gemv_bf16": spirv_gemv_bf16_mixed,
		"rms_f32": spirv_rms_norm_f32, "rms_bf16": spirv_rms_norm_bf16, "rms_no_scale": spirv_rms_norm_no_scale_f32,
		"gelu": spirv_gelu_tanh_mul_f32, "rope": spirv_rope_partial_f32, "silu": spirv_silu_mul_f32,
		"add_f32": spirv_vec_add_f32, "add_bf16": spirv_vec_add_bf16}
}
func TestVulkanOfflineShaderContractEmbedded(t *testing.T) {
	for name, b := range contractShaders() {
		t.Run(name, func(t *testing.T) {
			got, err := InspectVulkanShader(b)
			if err != nil {
				t.Fatal(err)
			}
			want := VulkanShaderContract{LocalSize: [3]uint32{256, 1, 1}}
			switch name {
			case "layernorm", "attention", "gemv_f32", "gemv_bf16", "rms_f32", "rms_bf16", "rms_no_scale":
				want.SharedBytes = 1024
			}
			if name == "attention_f32" {
				want.LocalSize = [3]uint32{16, 16, 1}
				want.SharedBytes = 9408
			}
			if name == "linear_regtile" {
				want.LocalSize = [3]uint32{16, 16, 1}
				want.SharedBytes = 8192
			}
			if name == "linear_f16_weight" || name == "linear" || name == "conv1d3" || name == "conv2d_chw" {
				want.LocalSize = [3]uint32{16, 16, 1}
				want.SharedBytes = 2048
			}
			if name == "lstm_sequence" {
				want.SharedBytes = 1024
			}
			interfaces := map[string][2]uint32{
				"linear_regtile": {15, 12}, "linear_f16_weight": {15, 12}, "conv1d3": {15, 24}, "conv2d_chw": {7, 36}, "channel_affine": {15, 12}, "attention_f32": {15, 20}, "gelu_erf": {3, 4}, "linear": {15, 12}, "lstm_cell": {15, 4}, "lstm_sequence": {255, 24}, "layernorm": {15, 12}, "attention": {7, 20}, "gemv_f32": {7, 8}, "gemv_bf16": {7, 8},
				"rms_f32": {3, 8}, "rms_bf16": {3, 8}, "rms_no_scale": {3, 8},
				"gelu": {3, 4}, "rope": {3, 16}, "silu": {7, 4}, "add_f32": {7, 4}, "add_bf16": {7, 4},
			}
			want.StorageBindings, want.PushBytes = interfaces[name][0], interfaces[name][1]
			if got != want {
				t.Fatalf("got%+v want%+v", got, want)
			}
			if err := vkCheckShaderLimits(got, offlineLimits()); err != nil {
				t.Fatal(err)
			}
			// Result has no borrowed slices; unaligned source uses the same contract.
			unaligned := append([]byte{0}, b...)
			again, err := InspectVulkanShader(unaligned[1:])
			if err != nil || again != want {
				t.Fatal(again, err)
			}
		})
	}
}
func editSPIRV(code []byte, op uint16, fn func([]uint32)) []byte {
	w := spirvTestWords(code)
	for i := 5; i < len(w); {
		n := int(w[i] >> 16)
		if uint16(w[i]) == op {
			fn(w[i : i+n])
			return spirvTestBytes(w)
		}
		i += n
	}
	panic("missing op")
}
func TestVulkanOfflineShaderContractRejects(t *testing.T) {
	base := dummySPIRV()
	word := func(index int, value uint32) []byte {
		w := spirvTestWords(base)
		w[index] = value
		return spirvTestBytes(w)
	}
	appendWords := func(words ...uint32) []byte { return append(append([]byte(nil), base...), spirvTestBytes(words)...) }
	cases := map[string][]byte{
		"empty": nil, "tail": append(base, 0), "oversized": make([]byte, (1<<20)+4), "bound": word(3, 65537), "zero_bound": word(3, 0), "version": word(1, 0x10400), "schema": word(4, 1),
		"zero_word_count": word(5, 17), "truncated": word(5, 65535<<16|17), "unknown_op": word(5, 2<<16|65535),
		"optional_cap": editSPIRV(base, 17, func(a []uint32) { a[1] = 9 }), "duplicate_cap": appendWords(2<<16|17, 1),
		"extension": appendWords(3<<16|10, 0x00414243, 0), "specconst": appendWords(4<<16|50, 1, 2, 7), "modeid": appendWords(6<<16|331, 3, 38, 1, 1, 1),
		"noncompute": editSPIRV(base, 15, func(a []uint32) { a[1] = 0 }), "wrong_entry": editSPIRV(base, 15, func(a []uint32) { a[3] = 0x006f6f66 }),
		"bad_string_padding": editSPIRV(base, 15, func(a []uint32) { a[4] = 256 }), "mode_target": editSPIRV(base, 16, func(a []uint32) { a[1] = 2 }),
		"zero_local": editSPIRV(base, 16, func(a []uint32) { a[3] = 0 }), "duplicate_mode": appendWords(6<<16|16, 3, 17, 1, 1, 1),
		"memorymodel": editSPIRV(base, 14, func(a []uint32) { a[2] = 3 }), "duplicate_id": appendWords(2<<16|19, 1),
		"result_bound": editSPIRV(base, 54, func(a []uint32) { a[2] = 5 }), "unterminated_func": base[:len(base)-4],
		"float16":        editSPIRV(spirv_vec_add_f32, 22, func(a []uint32) { a[2] = 16 }),
		"int64":          editSPIRV(spirv_vec_add_f32, 21, func(a []uint32) { a[2] = 64 }),
		"override_local": editSPIRV(spirv_vec_add_f32, 16, func(a []uint32) { a[3] = 128 }),
		"legacy_broken":  buildSPIRVVecAdd(),
	}
	for name, code := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := InspectVulkanShader(code)
			if !errors.Is(err, ErrVulkanShaderContract) {
				t.Fatal("not rejected", err)
			}
		})
	}
}
func TestVulkanOfflineShaderContractLocalLimits(t *testing.T) {
	c := VulkanShaderContract{LocalSize: [3]uint32{16, 4, 4}, SharedBytes: 1024}
	limits := offlineLimits()
	limits.SharedMemoryBytes = 1024
	if err := vkCheckShaderLimits(c, limits); err != nil {
		t.Fatal("exactlimit", err)
	}
	for i := 0; i < 3; i++ {
		l := limits
		l.WorkgroupSize[i] = c.LocalSize[i] - 1
		expectErrorIs(t, vkCheckShaderLimits(c, l), ErrVulkanLimit)
	}
	l := limits
	l.WorkgroupInvocations = 255
	expectErrorIs(t, vkCheckShaderLimits(c, l), ErrVulkanLimit)
	l = limits
	l.SharedMemoryBytes = 1023
	expectErrorIs(t, vkCheckShaderLimits(c, l), ErrVulkanLimit)
	c.LocalSize = [3]uint32{^uint32(0), ^uint32(0), ^uint32(0)}
	l.WorkgroupSize = c.LocalSize
	l.WorkgroupInvocations = ^uint32(0)
	expectErrorIs(t, vkCheckShaderLimits(c, l), ErrVulkanLimit)
}
func TestVulkanOfflineShaderContractNativePreflight(t *testing.T) {
	offlineVK(t)
	calls := 0
	mockVK(t, &vkCreateShaderModule, func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkShaderModule) VkResult { calls++; return -1 })
	invalid := editSPIRV(dummySPIRV(), 17, func(a []uint32) { a[1] = 9 })
	for _, legacy := range []bool{false, true} {
		var err error
		if legacy {
			_, err = LoadSPIRV(invalid, 1)
		} else {
			_, err = VkKernelCreate(invalid, 1, 0)
		}
		expectErrorIs(t, err, ErrVulkanShaderContract)
		vkLimits.WorkgroupSize[0] = 128
		if legacy {
			_, err = LoadSPIRV(spirv_vec_add_f32, 3)
		} else {
			_, err = VkKernelCreate(spirv_vec_add_f32, 3, 4)
		}
		expectErrorIs(t, err, ErrVulkanLimit)
		vkLimits = offlineLimits()
		vkLimits.SharedMemoryBytes = 1023
		if legacy {
			_, err = LoadSPIRV(spirv_gemv_f32, 3)
		} else {
			_, err = VkKernelCreate(spirv_gemv_f32, 3, 8)
		}
		expectErrorIs(t, err, ErrVulkanLimit)
		vkLimits = offlineLimits()
	}
	if calls != 0 {
		t.Fatal("invalid contract reached native")
	}
}

func FuzzVulkanShaderContract(f *testing.F) {
	f.Add(dummySPIRV())
	f.Add(layoutTestModule())
	for _, b := range contractShaders() {
		f.Add(b)
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		c, err := InspectVulkanShader(b)
		if err == nil {
			for _, n := range c.LocalSize {
				if n == 0 {
					t.Fatal("zero admitted")
				}
			}
			if c.StorageBindings > 0xffff || c.PushBytes > 128 || c.PushBytes%4 != 0 {
				t.Fatal("unbounded shader layout", c)
			}
			if err := vkCheckShaderInterface(c, 16, 128); err != nil {
				t.Fatal("admitted layout cannot fit envelope", err)
			}
			_ = vkCheckShaderLimits(c, offlineLimits())
		}
	})
}

// Build declarations independently of glslang to exercise size accounting.
func sharedTestModule(length uint32) []byte {
	w := spirvTestWords(dummySPIRV())
	w[3] = 20
	var at int
	for at = 5; at < len(w); at += int(w[at] >> 16) {
		if uint16(w[at]) == 54 {
			break
		}
	}
	declarations := []uint32{4<<16 | 21, 5, 32, 0, 3<<16 | 22, 6, 32, 4<<16 | 43, 5, 7, length,
		4<<16 | 28, 8, 6, 7, 4<<16 | 32, 9, 4, 8, 4<<16 | 59, 9, 10, 4}
	out := append([]uint32(nil), w[:at]...)
	out = append(out, declarations...)
	out = append(out, w[at:]...)
	return spirvTestBytes(out)
}
func TestVulkanOfflineShaderContractSharedMemory(t *testing.T) {
	base := sharedTestModule(256)
	if c, err := InspectVulkanShader(base); err != nil || c.SharedBytes != 1024 {
		t.Fatal(c, err)
	}
	appendInstruction := func(b []byte, words ...uint32) []byte {
		return append(append([]byte(nil), b...), spirvTestBytes(words)...)
	}
	for name, code := range map[string][]byte{
		"zero": sharedTestModule(0), "overflow": sharedTestModule(^uint32(0)),
		"float_length":        editSPIRV(base, 43, func(a []uint32) { a[1] = 6 }),
		"unknown_length":      editSPIRV(base, 28, func(a []uint32) { a[3] = 19 }),
		"runtime_array":       editSPIRV(base, 28, func(a []uint32) { a[0] = 4<<16 | 29 }),
		"decorated_array":     appendInstruction(base, 4<<16|71, 8, 6, 16),
		"decorated_variable":  appendInstruction(base, 3<<16|71, 10, 24),
		"pointer_disagrees":   editSPIRV(base, 59, func(a []uint32) { a[3] = 6 }),
		"pointer_wrong_class": editSPIRV(base, 32, func(a []uint32) { a[2] = 7 }),
		"undefined_pointer":   editSPIRV(base, 59, func(a []uint32) { a[1] = 19 }),
		"spec_length":         editSPIRV(base, 43, func(a []uint32) { a[0] = 4<<16 | 50 }),
	} {
		t.Run(name, func(t *testing.T) {
			_, err := InspectVulkanShader(code)
			expectErrorIs(t, err, ErrVulkanShaderContract)
		})
	}
	// Two distinct shared variables of same type each consume storage.
	w := spirvTestWords(base)
	var at int
	for at = 5; at < len(w); at += int(w[at] >> 16) {
		if uint16(w[at]) == 54 {
			break
		}
	}
	out := append([]uint32(nil), w[:at]...)
	out = append(out, 4<<16|59, 9, 11, 4)
	out = append(out, w[at:]...)
	got, err := InspectVulkanShader(spirvTestBytes(out))
	if err != nil || got.SharedBytes != 2048 {
		t.Fatal(got, err)
	}
}

func TestVulkanOfflineShaderContractOwnsNativeCode(t *testing.T) {
	for _, legacy := range []bool{false, true} {
		t.Run(map[bool]string{false: "kernel", true: "legacy"}[legacy], func(t *testing.T) {
			// Existing mock constructors exercise rollback. Stop at shader creation,
			// mutate caller storage inside the mock and verify the native view is owned.
			offlineVK(t)
			mockVK(t, &vkCreateDescriptorSetLayout, func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkDescriptorSetLayout) VkResult { return -1 })
			mockVK(t, &vkCreatePipelineLayout, func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkPipelineLayout) VkResult { return -1 })
			mockVK(t, &vkCreateComputePipelines, func(VkDevice, uintptr, uint32, unsafe.Pointer, unsafe.Pointer, *VkPipeline) VkResult { return -1 })
			mockVK(t, &vkCreateDescriptorPool, func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkDescriptorPool) VkResult { return -1 })
			mockVK(t, &vkAllocateDescriptorSets, func(VkDevice, unsafe.Pointer, *VkDescriptorSet) VkResult { return -1 })
			mockVK(t, &vkAllocateCommandBuffers, func(VkDevice, unsafe.Pointer, *VkCommandBuffer) VkResult { return -1 })
			mockVK(t, &vkCreateFence, func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkFence) VkResult { return -1 })
			mockVK(t, &vkDestroyShaderModule, func(VkDevice, VkShaderModule, unsafe.Pointer) {})
			mockVK(t, &vkDestroyPipeline, func(VkDevice, VkPipeline, unsafe.Pointer) {})
			mockVK(t, &vkDestroyPipelineLayout, func(VkDevice, VkPipelineLayout, unsafe.Pointer) {})
			mockVK(t, &vkDestroyDescriptorSetLayout, func(VkDevice, VkDescriptorSetLayout, unsafe.Pointer) {})
			mockVK(t, &vkDestroyDescriptorPool, func(VkDevice, VkDescriptorPool, unsafe.Pointer) {})
			mockVK(t, &vkFreeCommandBuffers, func(VkDevice, VkCommandPool, uint32, *VkCommandBuffer) {})
			mockVK(t, &vkDestroyFence, func(VkDevice, VkFence, unsafe.Pointer) {})
			code := dummySPIRV()
			called := false
			mockVK(t, &vkCreateShaderModule, func(d VkDevice, p, a unsafe.Pointer, out *VkShaderModule) VkResult {
				called = true
				n := *(*uint64)(unsafe.Add(p, 24))
				native := *(*unsafe.Pointer)(unsafe.Add(p, 32))
				if n != uint64(len(code)) || uintptr(native)%4 != 0 {
					t.Fatal("native code layout")
				}
				clear(code)
				words := unsafe.Slice((*uint32)(native), int(n)/4)
				if _, err := vkInspectSPIRV(words); err != nil {
					t.Fatal("native code aliased caller", err)
				}
				return -1
			})
			var err error
			if legacy {
				_, err = LoadSPIRV(code, 1)
			} else {
				_, err = VkKernelCreate(code, 1, 0)
			}
			if err == nil || !called {
				t.Fatal("native ownership path not reached", err)
			}
		})
	}
}
