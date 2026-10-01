package vulkan

import (
	"encoding/binary"
	"testing"
)

func TestVulkanOfflineQ5IntegerDotMMQContract(t *testing.T) {
	for _, code := range [][]byte{spirv_linear_q5_integer_dot_mmq, spirv_quant_q8_integer_dot_coop} {
		words := make([]uint32, len(code)/4)
		for i := range words {
			words[i] = binary.LittleEndian.Uint32(code[i*4:])
		}
		if len(code)%4 != 0 || len(words) < 5 {
			t.Fatal("framing")
		}
		if _, err := vkInspectSPIRVMode(words, true); err != nil {
			t.Fatal(err)
		}
	}
}
