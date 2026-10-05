package vulkan

import (
	"errors"
	"os"
	"testing"
)

// Explicit integer-dot admission accepts only tightly strided uvec2/uvec4
// storage views; default admission keeps scalar-only storage arrays.
func TestVulkanOfflineIntegerDotWideStorageViews(t *testing.T) {
	layout := func(count, stride, signed uint32) (vkSPIRVLayout, map[uint32]vkSPIRVType) {
		types := map[uint32]vkSPIRVType{1: {21, 32, signed}, 2: {23, 1, count}, 3: {29, 2, 0}, 4: {30, 0, 0}, 5: {32, 12, 4}}
		l := vkSPIRVLayout{
			structs:     map[uint32][]uint32{4: {3}},
			variables:   map[uint32]vkSPIRVLayoutVariable{6: {pointer: 5, storage: 12}},
			decorations: map[[2]uint32]uint32{{4, 2}: 0, {6, 34}: 0, {6, 33}: 0, {3, 6}: stride},
			offsets:     map[[2]uint32]uint32{{4, 0}: 0},
		}
		return l, types
	}
	for _, c := range []struct {
		name                  string
		count, stride, signed uint32
		integerDot, ok        bool
	}{
		{"uvec4", 4, 16, 0, true, true}, {"uvec2", 2, 8, 0, true, true},
		{"default-mode", 4, 16, 0, false, false}, {"loose-stride", 4, 32, 0, true, false},
		{"uvec3", 3, 12, 0, true, false}, {"signed", 4, 16, 1, true, false},
	} {
		l, types := layout(c.count, c.stride, c.signed)
		bindings, _, err := l.inspect(types, 0x10300, c.integerDot)
		if (err == nil) != c.ok || (c.ok && bindings != 1) {
			t.Fatalf("%s: bindings=%d err=%v", c.name, bindings, err)
		}
	}
	code, err := os.ReadFile("testdata/integer-dot/linear-mmq-stripped.spv")
	if err != nil {
		t.Fatal(err)
	}
	if c, err := vkInspectSPIRVMode(spirvTestWords(code), true); err != nil || !c.IntegerDot || c.LocalSize != [3]uint32{128, 1, 1} {
		t.Fatal("MMQ admission", c, err)
	}
	if _, err := InspectVulkanShader(code); !errors.Is(err, ErrVulkanShaderContract) {
		t.Fatal("baseline admission", err)
	}
	words := spirvTestWords(code)
	for _, arm := range []string{"stride", "float-vector"} {
		w := append([]uint32(nil), words...)
		var floatType uint32
		for i := 5; i < len(w); i += int(w[i] >> 16) {
			if w[i]&65535 == 22 {
				floatType = w[i+1]
			}
		}
		edited := false
		for i := 5; i < len(w) && !edited; i += int(w[i] >> 16) {
			switch op := w[i] & 65535; {
			case arm == "stride" && op == 71 && w[i+2] == 6 && w[i+3] == 16:
				w[i+3], edited = 32, true
			case arm == "float-vector" && op == 23 && w[i+3] == 4:
				w[i+2], edited = floatType, true
			}
		}
		if !edited {
			t.Fatal("no edit", arm)
		}
		if _, err := vkInspectSPIRVMode(w, true); !errors.Is(err, ErrVulkanShaderContract) {
			t.Fatal("not rejected", arm, err)
		}
	}
}

func TestVulkanOfflineArenaAlignedAllocRejectsBadAlignment(t *testing.T) {
	var a *VkTensorArena
	if _, err := a.AllocF32Aligned(nil, 24, 4); err == nil {
		t.Fatal("accepted non-power-of-two alignment")
	}
}
