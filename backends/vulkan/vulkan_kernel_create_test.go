package vulkan

import (
	"strings"
	"testing"
)

func TestVkKernelCreateRejectsMalformedInputs(t *testing.T) {
	oldReady := vkReady
	vkReady = true
	t.Cleanup(func() { vkReady = oldReady })
	cases := []struct {
		name     string
		spirv    []byte
		buffers  int
		pushSize int
		want     string
	}{
		{"empty spirv", nil, 1, 0, "SPIR-V"},
		{"odd spirv", []byte{1, 2, 3}, 1, 0, "SPIR-V"},
		{"zero buffers", []byte{0, 0, 0, 0}, 0, 0, "buffer count"},
		{"negative push", []byte{0, 0, 0, 0}, 1, -1, "push constant"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := VkKernelCreate(tc.spirv, tc.buffers, tc.pushSize)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want substring %q", err, tc.want)
			}
		})
	}
}
