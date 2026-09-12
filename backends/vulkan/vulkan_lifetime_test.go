package vulkan

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"
)

// All native entry points used here are mocks. Never initialise a real driver.
type lifetimeMock struct {
	events       []string
	submit, wait VkResult
	hook         func(string)
}

func newLifetimeMock(t *testing.T) (*lifetimeMock, *VkComputeKernel, *VkBuf) {
	t.Helper()
	offlineVK(t)
	m := &lifetimeMock{}
	step := func(s string) {
		m.events = append(m.events, s)
		if m.hook != nil {
			m.hook(s)
		}
	}
	owner := func(d VkDevice) {
		if d != 100 {
			t.Errorf("wrong device: %d", d)
		}
	}
	mockVK(t, &vkResetCommandBuffer, func(VkCommandBuffer, uint32) VkResult { step("command-reset"); return VK_SUCCESS })
	mockVK(t, &vkUpdateDescriptorSets, func(d VkDevice, n uint32, p unsafe.Pointer, c uint32, q unsafe.Pointer) {
		owner(d)
		step("descriptors")
	})
	mockVK(t, &vkBeginCommandBuffer, func(VkCommandBuffer, unsafe.Pointer) VkResult { step("begin"); return VK_SUCCESS })
	mockVK(t, &vkCmdBindPipeline, func(VkCommandBuffer, uint32, VkPipeline) { step("pipeline") })
	mockVK(t, &vkCmdBindDescriptorSets, func(VkCommandBuffer, uint32, VkPipelineLayout, uint32, uint32, *VkDescriptorSet, uint32, unsafe.Pointer) {
		step("bindings")
	})
	mockVK(t, &vkCmdDispatch, func(VkCommandBuffer, uint32, uint32, uint32) { step("dispatch") })
	mockVK(t, &vkCmdPipelineBarrier, func(cmd VkCommandBuffer, src, dst, flags, n uint32, p unsafe.Pointer, nb uint32, pb unsafe.Pointer, ni uint32, pi unsafe.Pointer) {
		if cmd == 0 || flags != 0 || n != 1 || p == nil || nb != 0 || pb != nil || ni != 0 || pi != nil {
			t.Error("invalid barrier call")
			return
		}
		v := (*vkMemoryBarrier)(p)
		if v.sType != 46 || v.pNext != 0 {
			t.Error("barrier header")
		}
		if dst == 0x800 {
			if src != 0x4800 || v.srcAccessMask != 0x4040 || v.dstAccessMask != 0x60 {
				t.Error("acquire barrier masks")
			}
			step("acquire")
		} else {
			if src != 0x800 || dst != 0x4000 || v.srcAccessMask != 0x40 || v.dstAccessMask != 0x6000 {
				t.Error("release barrier masks")
			}
			step("release")
		}
	})
	mockVK(t, &vkEndCommandBuffer, func(VkCommandBuffer) VkResult { step("end"); return VK_SUCCESS })
	mockVK(t, &vkResetFences, func(d VkDevice, n uint32, f *VkFence) VkResult { owner(d); step("fence-reset"); return VK_SUCCESS })
	mockVK(t, &vkQueueSubmit, func(q VkQueue, n uint32, p unsafe.Pointer, f VkFence) VkResult {
		if q != 103 {
			t.Error("queue owner")
		}
		step("submit")
		return m.submit
	})
	mockVK(t, &vkWaitForFences, func(d VkDevice, n uint32, f *VkFence, all uint32, ns uint64) VkResult {
		owner(d)
		if n != 1 || all != 1 || ns == 0 || ns > uint64(10*time.Millisecond) {
			t.Errorf("invalid wait %d/%d/%d", n, all, ns)
		}
		step("wait")
		return m.wait
	})
	mockVK(t, &vkDestroyFence, func(d VkDevice, f VkFence, p unsafe.Pointer) { owner(d); step("free-fence") })
	mockVK(t, &vkFreeCommandBuffers, func(d VkDevice, p VkCommandPool, n uint32, c *VkCommandBuffer) {
		owner(d)
		if p != 102 || n != 1 {
			t.Error("pool owner")
		}
		step("free-command")
	})
	mockVK(t, &vkDestroyDescriptorPool, func(d VkDevice, h VkDescriptorPool, p unsafe.Pointer) { owner(d); step("free-pool") })
	mockVK(t, &vkDestroyPipeline, func(d VkDevice, h VkPipeline, p unsafe.Pointer) { owner(d); step("free-pipeline") })
	mockVK(t, &vkDestroyPipelineLayout, func(d VkDevice, h VkPipelineLayout, p unsafe.Pointer) { owner(d); step("free-layout") })
	mockVK(t, &vkDestroyDescriptorSetLayout, func(d VkDevice, h VkDescriptorSetLayout, p unsafe.Pointer) { owner(d); step("free-setlayout") })
	mockVK(t, &vkUnmapMemory, func(d VkDevice, h VkDeviceMemory) { owner(d); step("unmap") })
	mockVK(t, &vkDestroyBuffer, func(d VkDevice, h VkBuffer, p unsafe.Pointer) { owner(d); step("free-buffer") })
	mockVK(t, &vkFreeMemory, func(d VkDevice, h VkDeviceMemory, p unsafe.Pointer) { owner(d); step("free-memory") })
	k := &VkComputeKernel{device: 100, queue: 103, commandPool: 102, pipeline: 1, pipelineLayout: 2, descSetLayout: 3, descPool: 4, descSet: 5, cmdBuf: 6, fence: 7, numBuffers: 1}
	storage := new([4]float32)
	b := &VkBuf{device: 100, buf: 8, mem: 9, size: 16, mapped: unsafe.Pointer(&storage[0])}
	return m, k, b
}
func expectErrorIs(t *testing.T, err, errorKind error) {
	t.Helper()
	if !errors.Is(err, errorKind) {
		t.Fatalf("error %v want %v", err, errorKind)
	}
}
func eventCount(m *lifetimeMock, s string) int {
	n := 0
	for _, e := range m.events {
		if e == s {
			n++
		}
	}
	return n
}
func cancelAfterSubmit(t *testing.T, m *lifetimeMock, k *VkComputeKernel, b *VkBuf) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.hook = func(s string) {
		if s == "submit" {
			cancel()
		}
	}
	err := k.DispatchContext(ctx, 1, 1, 1, []*VkBuf{b}, nil)
	expectErrorIs(t, err, context.Canceled)
	expectErrorIs(t, err, ErrVulkanInFlight)
	m.hook = nil
}

func TestVulkanOfflineLifetimeTimeoutDrain(t *testing.T) {
	m, k, b := newLifetimeMock(t)
	m.wait = VK_TIMEOUT
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Millisecond)
	defer cancel()
	err := k.DispatchContext(ctx, 1, 1, 1, []*VkBuf{b}, nil)
	expectErrorIs(t, err, ErrVulkanInFlight)
	expectErrorIs(t, err, context.DeadlineExceeded)
	if eventCount(m, "submit") != 1 || eventCount(m, "wait") == 0 || vkPending == nil {
		t.Fatal("missing pending work", m.events)
	}
	if VulkanReady() || VulkanBF16Ready() {
		t.Fatal("pending work reported ready")
	}
	n := len(m.events)
	expectErrorIs(t, k.Dispatch(1, 1, 1, []*VkBuf{b}, nil), ErrVulkanInFlight)
	expectErrorIs(t, b.UploadChecked([]float32{99}), ErrVulkanInFlight)
	out := []float32{77}
	expectErrorIs(t, b.DownloadChecked(out), ErrVulkanInFlight)
	expectErrorIs(t, b.UploadChecked(nil), ErrVulkanInFlight)
	expectErrorIs(t, b.FreeChecked(), ErrVulkanInFlight)
	expectErrorIs(t, k.Close(), ErrVulkanInFlight)
	if b.deferredFree || len(m.events) != n || out[0] != 77 {
		t.Fatal("unsafe pending mutation")
	}
	b.Free()
	b.Free()
	if !b.deferredFree || b.closed || k.closed || len(m.events) != n {
		t.Fatal("unsafe legacy free")
	}
	expectErrorIs(t, VulkanDrain(context.Background(), 2*time.Millisecond), ErrVulkanInFlight)
	if b.closed || vkPending == nil {
		t.Fatal("drain timeout released resource")
	}
	m.wait = VK_SUCCESS
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	if !b.closed || vkPending != nil {
		t.Fatal("confirmed fence did not release")
	}
	if !VulkanReady() || !VulkanBF16Ready() {
		t.Fatal("drained lane not ready")
	}
	if eventCount(m, "unmap") != 1 || eventCount(m, "free-buffer") != 1 || eventCount(m, "free-memory") != 1 {
		t.Fatal(m.events)
	}
	if !reflect.DeepEqual(m.events[len(m.events)-3:], []string{"unmap", "free-buffer", "free-memory"}) {
		t.Fatal("free order", m.events)
	}
	n = len(m.events)
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	b.Free()
	if err := b.FreeChecked(); err != nil {
		t.Fatal(err)
	}
	if n != len(m.events) {
		t.Fatal("double free/wait")
	}
	if err := k.Close(); err != nil {
		t.Fatal(err)
	}
	expectErrorIs(t, k.Dispatch(1, 1, 1, []*VkBuf{b}, nil), ErrVulkanClosed)
}

func TestVulkanOfflineLifetimeCancellationBoundaries(t *testing.T) {
	for _, boundary := range []string{"preflight", "command-reset", "descriptors", "begin", "acquire", "release", "end", "fence-reset", "submit", "wait"} {
		t.Run(boundary, func(t *testing.T) {
			m, k, b := newLifetimeMock(t)
			m.wait = VK_TIMEOUT
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if boundary == "preflight" {
				cancel()
			}
			m.hook = func(s string) {
				if s == boundary {
					cancel()
				}
			}
			err := k.DispatchContext(ctx, 1, 1, 1, []*VkBuf{b}, nil)
			expectErrorIs(t, err, context.Canceled)
			submitted := boundary == "submit" || boundary == "wait"
			if submitted != (vkPending != nil) || eventCount(m, "submit") != map[bool]int{true: 1, false: 0}[submitted] {
				t.Fatal("submission boundary", m.events)
			}
			if submitted {
				expectErrorIs(t, err, ErrVulkanInFlight)
				n := len(m.events)
				expectErrorIs(t, VulkanDrain(ctx, time.Second), context.Canceled)
				if len(m.events) != n {
					t.Fatal("cancelled drain called native")
				}
				m.hook = nil
				m.wait = VK_SUCCESS
				if err := VulkanDrain(context.Background(), time.Second); err != nil {
					t.Fatal(err)
				}
			}
			if err := b.FreeChecked(); err != nil {
				t.Fatal(err)
			}
			if err := k.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestVulkanOfflineLifetimeCompletionRacesCancel(t *testing.T) {
	for _, draining := range []bool{false, true} {
		t.Run(map[bool]string{false: "dispatch", true: "drain"}[draining], func(t *testing.T) {
			m, k, b := newLifetimeMock(t)
			if draining {
				cancelAfterSubmit(t, m, k, b)
				b.Free()
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			m.hook = func(s string) {
				if s == "wait" {
					cancel()
				}
			}
			var err error
			if draining {
				err = VulkanDrain(ctx, time.Second)
			} else {
				err = k.DispatchContext(ctx, 1, 1, 1, []*VkBuf{b}, nil)
			}
			expectErrorIs(t, err, context.Canceled)
			if errors.Is(err, ErrVulkanInFlight) || vkPending != nil || b.closed != draining {
				t.Fatal("completion not recognised", err)
			}
			if err := k.Close(); err != nil {
				t.Fatal(err)
			}
			if err := b.FreeChecked(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestVulkanOfflineLifetimeUncertainAndDeviceLost(t *testing.T) {
	for _, op := range []string{"submit", "wait"} {
		for _, status := range []VkResult{-1, VK_ERROR_DEVICE_LOST} {
			t.Run(op+"/"+(&VulkanCallError{Result: status}).Error(), func(t *testing.T) {
				m, k, b := newLifetimeMock(t)
				if op == "submit" {
					m.submit = status
				} else {
					m.wait = status
				}
				err := k.Dispatch(1, 1, 1, []*VkBuf{b}, nil)
				expectErrorIs(t, err, ErrVulkanInFlight)
				expectErrorIs(t, err, ErrVulkanUncertain)
				var native *VulkanCallError
				if !errors.As(err, &native) || native.Result != status {
					t.Fatal("native cause missing", err)
				}
				if status == VK_ERROR_DEVICE_LOST {
					expectErrorIs(t, err, ErrVulkanDeviceLost)
				}
				if vkPending == nil || !vkPending.uncertain || vkPending.kernel != k || vkPending.buffers[0] != b {
					t.Fatal("missing quarantine")
				}
				n := len(m.events)
				m.wait = VK_SUCCESS
				want := ErrVulkanUncertain
				if status == VK_ERROR_DEVICE_LOST {
					want = ErrVulkanDeviceLost
				}
				expectErrorIs(t, VulkanDrain(context.Background(), time.Second), want)
				expectErrorIs(t, k.Dispatch(1, 1, 1, []*VkBuf{b}, nil), want)
				b.Free()
				expectErrorIs(t, b.FreeChecked(), want)
				if VulkanReady() || VulkanBF16Ready() {
					t.Fatal("quarantined device reported ready")
				}
				// Unrelated resources are also protected once the device is quarantined.
				otherK := &VkComputeKernel{device: 100, commandPool: 102, pipeline: 90}
				otherB := &VkBuf{device: 100, buf: 91, mem: 92, size: 4, mapped: b.mapped}
				expectErrorIs(t, otherK.Close(), want)
				expectErrorIs(t, otherB.FreeChecked(), want)
				otherB.Free()
				expectErrorIs(t, otherB.UploadChecked([]float32{1}), want)
				expectErrorIs(t, otherB.DownloadChecked([]float32{0}), want)
				_, err = VkBufAlloc(16)
				expectErrorIs(t, err, want)
				_, err = VkKernelCreate(dummySPIRV(), 1, 0)
				expectErrorIs(t, err, want)
				if VulkanInit() {
					t.Fatal("quarantined init accepted")
				}
				if len(m.events) != n {
					t.Fatal("quarantine called driver", m.events[n:])
				}
			})
		}
	}
}

func TestVulkanOfflineLifetimeSuccessfulReuseAndClose(t *testing.T) {
	m, k, b := newLifetimeMock(t)
	for i := 0; i < 3; i++ {
		if err := b.UploadChecked([]float32{float32(i)}); err != nil {
			t.Fatal(err)
		}
		if err := k.Dispatch(1, 1, 1, []*VkBuf{b}, nil); err != nil {
			t.Fatal(err)
		}
		out := []float32{-1}
		if err := b.DownloadChecked(out); err != nil || out[0] != float32(i) {
			t.Fatal(out, err)
		}
	}
	if vkPending != nil || eventCount(m, "submit") != 3 || eventCount(m, "command-reset") != 3 || eventCount(m, "fence-reset") != 3 {
		t.Fatal("reuse failed")
	}
	n := len(m.events)
	if err := k.Close(); err != nil {
		t.Fatal(err)
	}
	if err := k.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m.events[n:], []string{"free-fence", "free-command", "free-pool", "free-pipeline", "free-layout", "free-setlayout"}) {
		t.Fatal("kernel teardown", m.events[n:])
	}
	if k.pipeline != 0 || k.fence != 0 || k.descSet != 0 || k.cmdBuf != 0 || !k.closed {
		t.Fatal("live handles after close")
	}
	if err := b.FreeChecked(); err != nil {
		t.Fatal(err)
	}
	expectErrorIs(t, b.UploadChecked([]float32{1}), ErrVulkanClosed)
	expectErrorIs(t, b.DownloadChecked(nil), ErrVulkanClosed)
}

func TestVulkanOfflineLifetimeDeduplicatesRetainedBuffers(t *testing.T) {
	m, k, b := newLifetimeMock(t)
	k.numBuffers = 3
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.hook = func(s string) {
		if s == "submit" {
			cancel()
		}
	}
	bufs := []*VkBuf{b, b, b}
	expectErrorIs(t, k.DispatchContext(ctx, 1, 1, 1, bufs, nil), ErrVulkanInFlight)
	if len(vkPending.buffers) != 1 || vkPending.buffers[0] != b {
		t.Fatal("duplicate refs")
	}
	bufs[0] = nil
	bufs[1] = nil
	bufs[2] = nil
	b.Free()
	m.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	if eventCount(m, "free-buffer") != 1 {
		t.Fatal("duplicate frees")
	}
}

func TestVulkanOfflineLifetimeSerializedAdmission(t *testing.T) {
	m, k, b := newLifetimeMock(t)
	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	m.hook = func(s string) {
		if s == "wait" {
			close(entered)
			<-release
		}
	}
	done := make(chan error, 1)
	go func() { done <- k.Dispatch(1, 1, 1, []*VkBuf{b}, nil) }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("no wait entered")
	}
	// The lane is held by the first caller, so this must cancel at admission
	// without descriptor/fence/queue activity or access to mutable handles.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	expectErrorIs(t, k.DispatchContext(ctx, 1, 1, 1, []*VkBuf{b}, nil), context.DeadlineExceeded)
	expectErrorIs(t, VulkanDrain(context.Background(), 5*time.Millisecond), context.DeadlineExceeded)
	freed := make(chan error, 1)
	go func() { freed <- b.FreeChecked() }()
	select {
	case err := <-freed:
		t.Fatal("freed during native wait", err)
	case <-time.After(5 * time.Millisecond):
	}
	unblock()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if err := <-freed; err != nil {
		t.Fatal(err)
	}
	if eventCount(m, "submit") != 1 || eventCount(m, "descriptors") != 1 || !b.closed {
		t.Fatal("interleaved lane", m.events)
	}
}

func TestVulkanOfflineLifetimeParallelDispatch(t *testing.T) {
	m, k, b := newLifetimeMock(t)
	const workers = 16
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := 0; i < workers; i++ {
		// Exercise both shared and distinct kernels on the same queue/pool.
		kernel := k
		if i%2 != 0 {
			kernel = &VkComputeKernel{device: 100, queue: 103, commandPool: 102,
				pipeline: VkPipeline(100 + i), pipelineLayout: 2, descSet: VkDescriptorSet(200 + i),
				cmdBuf: VkCommandBuffer(300 + i), fence: VkFence(400 + i), numBuffers: 1}
		}
		wg.Add(1)
		go func() { defer wg.Done(); errs <- kernel.Dispatch(1, 1, 1, []*VkBuf{b}, nil) }()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	unit := []string{"command-reset", "descriptors", "begin", "acquire", "pipeline", "bindings", "dispatch", "release", "end", "fence-reset", "submit", "wait"}
	if len(m.events) != workers*len(unit) {
		t.Fatal("event count", len(m.events))
	}
	for i := 0; i < workers; i++ {
		if !reflect.DeepEqual(m.events[i*len(unit):(i+1)*len(unit)], unit) {
			t.Fatal("dispatch interleaving", m.events)
		}
	}
}

func TestVulkanOfflineLifetimeOwnerAndCleanupGuards(t *testing.T) {
	for _, which := range []string{"buffer-owner", "kernel-owner", "pool-owner", "cleanup-functions"} {
		t.Run(which, func(t *testing.T) {
			m, k, b := newLifetimeMock(t)
			switch which {
			case "buffer-owner":
				b.device = 999
			case "kernel-owner":
				k.device = 999
			case "pool-owner":
				k.commandPool = 999
			case "cleanup-functions":
				mockVK(t, &vkDestroyFence, (func(VkDevice, VkFence, unsafe.Pointer))(nil))
				mockVK(t, &vkFreeMemory, (func(VkDevice, VkDeviceMemory, unsafe.Pointer))(nil))
			}
			if which == "buffer-owner" || which == "cleanup-functions" {
				if err := b.FreeChecked(); err == nil {
					t.Fatal("bad buffer free accepted")
				}
			}
			if which != "buffer-owner" {
				if err := k.Close(); err == nil {
					t.Fatal("bad kernel close accepted")
				}
			}
			if len(m.events) != 0 {
				t.Fatal("guard called native")
			}
			if which != "cleanup-functions" {
				if err := k.Dispatch(1, 1, 1, []*VkBuf{b}, nil); err == nil || !strings.Contains(err.Error(), "owner mismatch") {
					t.Fatal("dispatch owner", err)
				}
			}
		})
	}
}

func TestVulkanOfflineLifetimeCacheAdmission(t *testing.T) {
	m, k, b := newLifetimeMock(t)
	// Save the one-shot for other tests without copying its internal atomics.
	// Admission errors must not consume it; observe with a marker closure.
	// This test is serial, and no cache call has been left running.
	vkKernelOnce = sync.Once{}
	t.Cleanup(func() { vkKernelOnce = sync.Once{} })
	cancelAfterSubmit(t, m, k, b)
	expectErrorIs(t, initVkKernels(), ErrVulkanInFlight)
	expectErrorIs(t, VkVecAddF32(b, b, b, 1), ErrVulkanInFlight)
	used := false
	vkKernelOnce.Do(func() { used = true })
	if !used {
		t.Fatal("pending admission consumed cache Once")
	}
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	vkKernelOnce = sync.Once{}
	mockVK(t, &vkReady, false)
	if err := initVkKernels(); err == nil {
		t.Fatal("uninitialised cache accepted")
	}
	used = false
	vkKernelOnce.Do(func() { used = true })
	if !used {
		t.Fatal("not-ready consumed cache Once")
	}
}

func TestVulkanOfflineLifetimeDrainPreflight(t *testing.T) {
	m, _, _ := newLifetimeMock(t)
	for _, budget := range []time.Duration{0, -1, 31 * time.Second} {
		if err := VulkanDrain(context.Background(), budget); err == nil {
			t.Fatal("bad budget accepted")
		}
	}
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	if len(m.events) != 0 {
		t.Fatal("empty drain called driver")
	}
	if err := (*VkComputeKernel)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	if err := (*VkBuf)(nil).FreeChecked(); err != nil {
		t.Fatal(err)
	}
}
