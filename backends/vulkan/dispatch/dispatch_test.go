package dispatch

import (
	"strings"
	"testing"
)

func TestVulkanDispatchRejectsInvalidKernelAndInputs(t *testing.T) {
	validBuf := &VkBuf{buf: 1, mem: 1, size: 4}
	cases := []struct {
		name string
		k    *VkComputeKernel
		gx   uint32
		bufs []*VkBuf
		want string
	}{
		{"nil kernel", nil, 1, []*VkBuf{validBuf}, "uninitialized kernel"},
		{"zero pipeline", &VkComputeKernel{pipelineLayout: 1, descSet: 1, cmdBuf: 1, fence: 1, numBuffers: 1}, 1, []*VkBuf{validBuf}, "uninitialized kernel"},
		{"zero workgroups", &VkComputeKernel{pipeline: 1, pipelineLayout: 1, descSet: 1, cmdBuf: 1, fence: 1, numBuffers: 1}, 0, []*VkBuf{validBuf}, "zero workgroups"},
		{"wrong buffer count", &VkComputeKernel{pipeline: 1, pipelineLayout: 1, descSet: 1, cmdBuf: 1, fence: 1, numBuffers: 2}, 1, []*VkBuf{validBuf}, "buffer count"},
		{"nil buffer", &VkComputeKernel{pipeline: 1, pipelineLayout: 1, descSet: 1, cmdBuf: 1, fence: 1, numBuffers: 1}, 1, []*VkBuf{nil}, "buffer 0"},
		{"zero buffer handle", &VkComputeKernel{pipeline: 1, pipelineLayout: 1, descSet: 1, cmdBuf: 1, fence: 1, numBuffers: 1}, 1, []*VkBuf{{mem: 1, size: 4}}, "buffer 0"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.k.Dispatch(tc.gx, 1, 1, tc.bufs, nil)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want substring %q", err, tc.want)
			}
		})
	}
}
