package vulkan

import (
	"bytes"
	"testing"
)

func TestLegacyAssembleSPIRVHasNoOperationPlaceholders(t *testing.T) {
	first, err := assembleSPIRV("vec_add")
	if err != nil || len(first) == 0 {
		t.Fatal("legacy vector shader", err)
	}
	first[0] ^= 0xff
	second, err := assembleSPIRV("vec_add")
	if err != nil || bytes.Equal(first, second) {
		t.Fatal("legacy shader storage escaped")
	}
	for _, name := range []string{"gemv_f32", "vec_add_bf16", "attention"} {
		if code, err := assembleSPIRV(name); err == nil || code != nil {
			t.Fatalf("placeholder operation %q accepted", name)
		}
	}
	contract, err := InspectVulkanShader(spirv_gemv_f32)
	if err != nil || contract.StorageBindings != 7 || contract.PushBytes != 8 {
		t.Fatalf("generated GEMV contract=%+v err=%v", contract, err)
	}
	if bytes.Equal(spirv_gemv_f32, spirv_vec_add_f32) {
		t.Fatal("generated GEMV aliases vector add")
	}
}

func TestLoadSPIRVRejectsMalformedInputsBeforeRuntime(t *testing.T) {
	if _, err := LoadSPIRV(nil, 1); err == nil {
		t.Fatal("LoadSPIRV accepted nil bytecode")
	}
	if _, err := LoadSPIRV([]byte{1, 2, 3}, 1); err == nil {
		t.Fatal("LoadSPIRV accepted misaligned bytecode")
	}
	if _, err := LoadSPIRV([]byte{0, 0, 0, 0}, 0); err == nil {
		t.Fatal("LoadSPIRV accepted zero descriptor buffers")
	}
}
