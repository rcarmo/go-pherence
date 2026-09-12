package vulkan

import (
	"encoding/binary"
	"fmt"
	"reflect"
	"testing"
	"unsafe"
)

// Independent declaration-only fixtures around the existing no-op compute
// body; never submitted to a driver. IDs5..11 are types/resources, others free.
func layoutTestModule() []byte {
	w := spirvTestWords(dummySPIRV())
	w[3] = 40
	at := 5
	for uint16(w[at]) != 54 {
		at += int(w[at] >> 16)
	}
	declarations := []uint32{
		3<<16 | 22, 5, 32, // float32
		3<<16 | 29, 6, 5, // runtime array
		3<<16 | 30, 7, 6, // storage struct
		4<<16 | 32, 8, 2, 7, 4<<16 | 59, 8, 9, 2,
		4<<16 | 71, 6, 6, 4, 3<<16 | 71, 7, 3, 5<<16 | 72, 7, 0, 35, 0,
		4<<16 | 71, 9, 34, 0, 4<<16 | 71, 9, 33, 0,
		4<<16 | 30, 10, 5, 5, // push block2scalars
		4<<16 | 32, 11, 9, 10, 4<<16 | 59, 11, 12, 9,
		3<<16 | 71, 10, 2, 5<<16 | 72, 10, 0, 35, 0, 5<<16 | 72, 10, 1, 35, 4,
	}
	out := append([]uint32(nil), w[:at]...)
	out = append(out, declarations...)
	out = append(out, w[at:]...)
	return spirvTestBytes(out)
}
func changeInstruction(b []byte, predicate func([]uint32) bool, change func([]uint32) []uint32) []byte {
	w := spirvTestWords(b)
	for i := 5; i < len(w); {
		n := int(w[i] >> 16)
		a := w[i : i+n]
		if predicate(a) {
			out := append([]uint32(nil), w[:i]...)
			out = append(out, change(append([]uint32(nil), a...))...)
			out = append(out, w[i+n:]...)
			return spirvTestBytes(out)
		}
		i += n
	}
	panic("instruction absent")
}
func changeDecoration(b []byte, id, kind, value uint32) []byte {
	return changeInstruction(b, func(a []uint32) bool { return uint16(a[0]) == 71 && a[1] == id && a[2] == kind }, func(a []uint32) []uint32 { a[3] = value; return a })
}
func changeOffset(b []byte, member, value uint32) []byte {
	return changeInstruction(b, func(a []uint32) bool { return uint16(a[0]) == 72 && a[1] == 10 && a[2] == member }, func(a []uint32) []uint32 { a[4] = value; return a })
}
func addDeclaration(b []byte, extra ...uint32) []byte {
	return changeInstruction(b, func(a []uint32) bool { return uint16(a[0]) == 54 }, func(a []uint32) []uint32 { return append(append([]uint32(nil), extra...), a...) })
}
func TestVulkanOfflineShaderLayoutReflection(t *testing.T) {
	base := layoutTestModule()
	c, err := InspectVulkanShader(base)
	want := VulkanShaderContract{LocalSize: [3]uint32{1, 1, 1}, StorageBindings: 1, PushBytes: 8}
	if err != nil || c != want {
		t.Fatal(c, err)
	}
	// Sparse bindings legal, must fit caller slots. Push gaps/end accepted.
	sparse := changeDecoration(base, 9, 33, 15)
	sparse = changeOffset(sparse, 1, 124)
	c, err = InspectVulkanShader(sparse)
	if err != nil || c.StorageBindings != 1<<15 || c.PushBytes != 128 {
		t.Fatal(c, err)
	}
	if err := vkCheckShaderInterface(c, 16, 128); err != nil {
		t.Fatal(err)
	}
	expectErrorIs(t, vkCheckShaderInterface(c, 15, 128), ErrVulkanShaderContract)
	expectErrorIs(t, vkCheckShaderInterface(c, 16, 124), ErrVulkanShaderContract)
	// Reversed offsets are non-overlapping and accepted, not sorted by members.
	reversed := changeOffset(changeOffset(base, 0, 4), 1, 0)
	if c, err := InspectVulkanShader(reversed); err != nil || c.PushBytes != 8 {
		t.Fatal(c, err)
	}
	// StorageBuffer+Block in extension-free SPIRV1.3 also supported.
	modern := changeInstruction(base, func(a []uint32) bool { return uint16(a[0]) == 32 && a[1] == 8 }, func(a []uint32) []uint32 { a[2] = 12; return a })
	modern = changeInstruction(modern, func(a []uint32) bool { return uint16(a[0]) == 59 && a[2] == 9 }, func(a []uint32) []uint32 { a[3] = 12; return a })
	modern = changeInstruction(modern, func(a []uint32) bool { return uint16(a[0]) == 71 && a[1] == 7 }, func(a []uint32) []uint32 { a[2] = 2; return a })
	binary.LittleEndian.PutUint32(modern[4:], 0x10300)
	if c, err := InspectVulkanShader(modern); err != nil || c != want {
		t.Fatal(c, err)
	}
	binary.LittleEndian.PutUint32(modern[4:], 0x10000)
	_, err = InspectVulkanShader(modern)
	expectErrorIs(t, err, ErrVulkanShaderContract)
	// Caller supersets remain compatible, including synthetic no-resource kernels.
	c, err = InspectVulkanShader(base)
	if err != nil {
		t.Fatal(err)
	}
	if err := vkCheckShaderInterface(c, 4, 32); err != nil {
		t.Fatal(err)
	}
	if err := vkCheckShaderInterface(VulkanShaderContract{}, 1, 0); err != nil {
		t.Fatal(err)
	}
}
func TestVulkanOfflineShaderLayoutRejections(t *testing.T) {
	b := layoutTestModule()
	mutate := func(op uint16, id uint32, fn func([]uint32)) []byte {
		return changeInstruction(b, func(a []uint32) bool { return uint16(a[0]) == op && a[1] == id }, func(a []uint32) []uint32 { fn(a); return a })
	}
	drop := func(id, kind uint32) []byte {
		return changeInstruction(b, func(a []uint32) bool { return uint16(a[0]) == 71 && a[1] == id && a[2] == kind }, func([]uint32) []uint32 { return nil })
	}
	cases := map[string][]byte{
		"set1": changeDecoration(b, 9, 34, 1), "binding16": changeDecoration(b, 9, 33, 16), "maxbinding": changeDecoration(b, 9, 33, ^uint32(0)),
		"missing-binding": drop(9, 33), "missing-set": drop(9, 34), "missing-block": drop(7, 3), "missing-stride": drop(6, 6),
		"stride8": changeDecoration(b, 6, 6, 8), "ubo": mutate(71, 7, func(a []uint32) { a[2] = 2 }),
		"array-floatmissing":  mutate(29, 6, func(a []uint32) { a[2] = 39 }),
		"block-not-struct":    mutate(32, 8, func(a []uint32) { a[3] = 6 }),
		"nested-push":         mutate(30, 10, func(a []uint32) { a[2] = 7 }),
		"missing-push-offset": changeInstruction(b, func(a []uint32) bool { return uint16(a[0]) == 72 && a[1] == 10 && a[2] == 1 }, func([]uint32) []uint32 { return nil }),
		"overlap":             changeOffset(b, 1, 0), "unaligned": changeOffset(b, 1, 5), "push-overflow": changeOffset(b, 1, ^uint32(0)-3),
		"past128":                changeOffset(b, 1, 128),
		"member-outofbounds":     addDeclaration(b, 5<<16|72, 10, 2, 35, 8),
		"member-on-scalar":       addDeclaration(b, 5<<16|72, 5, 0, 35, 0),
		"block-on-variable":      addDeclaration(b, 3<<16|71, 9, 2),
		"arraystride-on-scalar":  addDeclaration(b, 4<<16|71, 5, 6, 4),
		"binding-on-type":        addDeclaration(b, 4<<16|71, 8, 33, 0),
		"conflicting-block":      addDeclaration(b, 3<<16|71, 7, 2),
		"duplicate-binding":      addDeclaration(b, 4<<16|59, 8, 13, 2, 4<<16|71, 13, 34, 0, 4<<16|71, 13, 33, 0),
		"two-push":               addDeclaration(b, 4<<16|59, 11, 13, 9),
		"descriptor-array":       addDeclaration(mutate(32, 8, func(a []uint32) { a[3] = 13 }), 4<<16|28, 13, 7, 14),
		"storage-nonzero-offset": changeInstruction(b, func(a []uint32) bool { return uint16(a[0]) == 72 && a[1] == 7 }, func(a []uint32) []uint32 { a[4] = 4; return a }),
		"push-initializer":       changeInstruction(b, func(a []uint32) bool { return uint16(a[0]) == 59 && a[2] == 12 }, func(a []uint32) []uint32 { a[0] = 5<<16 | 59; return append(a, 13) }),
	}
	for name, code := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := InspectVulkanShader(code)
			expectErrorIs(t, err, ErrVulkanShaderContract)
		})
	}
}
func TestVulkanOfflineShaderLayoutNativeAdmission(t *testing.T) {
	offlineVK(t)
	calls := 0
	mockVK(t, &vkCreateShaderModule, func(VkDevice, unsafe.Pointer, unsafe.Pointer, *VkShaderModule) VkResult { calls++; return -1 })
	for _, c := range []struct {
		code    []byte
		n, push int
	}{
		{spirv_vec_add_f32, 2, 4}, {spirv_vec_add_f32, 3, 0}, {spirv_gemv_f32, 3, 4},
		{spirv_rms_norm_no_scale_f32, 1, 8}, {spirv_rope_partial_f32, 2, 16},
	} {
		_, err := VkKernelCreate(c.code, c.n, c.push)
		expectErrorIs(t, err, ErrVulkanShaderContract)
	}
	_, err := LoadSPIRV(spirv_vec_add_f32, 3)
	expectErrorIs(t, err, ErrVulkanShaderContract)
	if calls != 0 {
		t.Fatal("mismatched pipeline reached native")
	}
}
func TestVulkanOfflineShaderLayoutCacheContracts(t *testing.T) {
	// Actual wrapper cache requests. Mismatches are rejected, never auto-filled:
	// no-scale needs a separate output and RoPE uses a different frequency layout.
	for _, c := range []struct {
		name          string
		code          []byte
		buffers, push int
		accept        bool
	}{
		{"addF32", spirv_vec_add_f32, 3, 4, true}, {"addBF16", spirv_vec_add_bf16, 3, 4, true},
		{"rmsF32", spirv_rms_norm_f32, 2, 8, true}, {"rmsBF16", spirv_rms_norm_bf16, 2, 8, true},
		{"rmsNoScale", spirv_rms_norm_no_scale_f32, 1, 8, false}, {"gemvF32", spirv_gemv_f32, 3, 8, true},
		{"gemvBF16", spirv_gemv_bf16_mixed, 3, 8, true}, {"silu", spirv_silu_mul_f32, 3, 4, true},
		{"gelu", spirv_gelu_tanh_mul_f32, 2, 4, true}, {"rope", spirv_rope_partial_f32, 2, 16, false},
		{"attention", spirv_attention_score, 3, 20, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			contract, err := InspectVulkanShader(c.code)
			if err != nil {
				t.Fatal(err)
			}
			err = vkCheckShaderInterface(contract, c.buffers, c.push)
			if (err == nil) != c.accept {
				t.Fatal("cache contract", err)
			}
		})
	}
}
func TestVulkanOfflineShaderLayoutBoundedResult(t *testing.T) {
	got, err := InspectVulkanShader(layoutTestModule())
	if err != nil {
		t.Fatal(err)
	}
	copy := got
	copy.StorageBindings = 0
	copy.PushBytes = 0
	again, err := InspectVulkanShader(layoutTestModule())
	if err != nil || !reflect.DeepEqual(again, got) {
		t.Fatal("mutated contract")
	}
	for _, c := range [][2]int{{0, 0}, {17, 0}, {1, -1}, {1, 129}, {1, 3}} {
		expectErrorIs(t, vkCheckShaderInterface(got, c[0], c[1]), ErrVulkanShaderContract)
	}
	if fmt.Sprint(copy) == fmt.Sprint(got) {
		t.Fatal("test mutation ineffective")
	}
}
