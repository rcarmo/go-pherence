package vulkan

import (
	"context"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"
)

func TestVulkanOfflineBarrierLayout(t *testing.T) {
	if !vkNative64() {
		t.Skip("current binding is64bit only")
	}
	var b vkMemoryBarrier
	for _, c := range []struct {
		name      string
		got, want uintptr
	}{
		{"size", unsafe.Sizeof(b), 24}, {"align", unsafe.Alignof(b), 8},
		{"type", unsafe.Offsetof(b.sType), 0}, {"next", unsafe.Offsetof(b.pNext), 8},
		{"source", unsafe.Offsetof(b.srcAccessMask), 16}, {"destination", unsafe.Offsetof(b.dstAccessMask), 20},
	} {
		if c.got != c.want {
			t.Errorf("%s=%d want%d", c.name, c.got, c.want)
		}
	}
}

func TestVulkanOfflineBarrierPreflight(t *testing.T) {
	m, k, b := newLifetimeMock(t)
	mockVK(t, &vkCmdPipelineBarrier, (func(VkCommandBuffer, uint32, uint32, uint32, uint32, unsafe.Pointer, uint32, unsafe.Pointer, uint32, unsafe.Pointer))(nil))
	err := k.Dispatch(1, 1, 1, []*VkBuf{b}, nil)
	if err == nil || !strings.Contains(err.Error(), "functions unavailable") {
		t.Fatal("missing barrier accepted", err)
	}
	if len(m.events) != 0 || vkPending != nil {
		t.Fatal("missing barrier mutated descriptors/submitted", m.events)
	}
}

// Check exact dependency placement, command-buffer ownership and global-memory
// scope with no buffer/image barriers. The same masks are used for aliases and
// for new/old compute kernels. This validates recording only, not GPU visibility.
func TestVulkanOfflineBarrierRecording(t *testing.T) {
	for _, n := range []int{1, 3, 16} {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			m, k, b := newLifetimeMock(t)
			k.numBuffers = n
			bufs := make([]*VkBuf, n)
			for i := range bufs {
				bufs[i] = b
			}
			for round := 0; round < 3; round++ {
				k.cmdBuf = VkCommandBuffer(50 + round)
				start := len(m.events)
				barriers := 0
				base := vkCmdPipelineBarrier
				// Restore immediately between iterations (mockVK cleanup remains test-wide).
				vkCmdPipelineBarrier = func(cmd VkCommandBuffer, src, dst, flags, count uint32, p unsafe.Pointer, bc uint32, bp unsafe.Pointer, ic uint32, ip unsafe.Pointer) {
					if cmd != VkCommandBuffer(50+round) {
						t.Error("wrong command buffer")
					}
					if count != 1 || bc != 0 || ic != 0 || bp != nil || ip != nil {
						t.Error("not a global memory dependency")
					}
					if barriers == 0 {
						if !reflect.DeepEqual(m.events[start:], []string{"descriptors", "begin"}) {
							t.Error("acquire is not first command", m.events[start:])
						}
					} else if barriers == 1 {
						if m.events[len(m.events)-1] != "dispatch" {
							t.Error("release not directly after dispatch")
						}
					} else {
						t.Error("too many barriers")
					}
					barriers++
					base(cmd, src, dst, flags, count, p, bc, bp, ic, ip)
				}
				err := k.Dispatch(1, 1, 1, bufs, nil)
				vkCmdPipelineBarrier = base
				if err != nil {
					t.Fatal(err)
				}
				expected := []string{"descriptors", "begin", "acquire", "pipeline", "bindings", "dispatch", "release", "end", "reset", "submit", "wait"}
				if barriers != 2 || !reflect.DeepEqual(m.events[start:], expected) {
					t.Fatal("dispatch dependencies", m.events[start:])
				}
			}
		})
	}
}

func TestVulkanOfflineBarrierPendingDrainDoesNotRecord(t *testing.T) {
	m, k, b := newLifetimeMock(t)
	cancelAfterSubmit(t, m, k, b)
	if eventCount(m, "acquire") != 1 || eventCount(m, "release") != 1 || eventCount(m, "wait") != 0 {
		t.Fatal("submit recording", m.events)
	}
	n := len(m.events)
	expectErrorIs(t, k.Dispatch(1, 1, 1, []*VkBuf{b}, nil), ErrVulkanInFlight)
	expectErrorIs(t, b.UploadChecked([]float32{1}), ErrVulkanInFlight)
	expectErrorIs(t, b.DownloadChecked([]float32{0}), ErrVulkanInFlight)
	if len(m.events) != n {
		t.Fatal("pending work recorded again")
	}
	m.wait = VK_SUCCESS
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m.events[n:], []string{"wait"}) {
		t.Fatal("drain recorded commands", m.events[n:])
	}
	if err := b.UploadChecked([]float32{2}); err != nil {
		t.Fatal(err)
	}
	if err := k.Dispatch(1, 1, 1, []*VkBuf{b}, nil); err != nil {
		t.Fatal(err)
	}
	if eventCount(m, "acquire") != 2 || eventCount(m, "release") != 2 {
		t.Fatal("reuse missing dependencies")
	}
}
