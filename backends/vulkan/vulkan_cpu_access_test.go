package vulkan

import (
	"context"
	"errors"
	"testing"
	"unsafe"
)

func sharedMockArena(t *testing.T) (*VkTensorArena, *memoryMock) {
	t.Helper()
	offlineVK(t)
	m := mockMemory(t)
	m.props.memoryTypes[0].propertyFlags = 14
	a, err := NewVkSharedTensorArena(context.Background(), 64)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := a.Close(); err != nil {
			t.Error(err)
		}
	})
	return a, m
}

func TestVulkanOfflineSharedMemorySelection(t *testing.T) {
	offlineVK(t)
	m := mockMemory(t)
	// An uncached first choice must not defeat the explicit shared contract.
	m.props.memoryTypeCount = 2
	m.props.memoryTypes[1] = vkMemoryType{propertyFlags: 14}
	a, err := NewVkSharedTensorArena(context.Background(), 64)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.selected) != 1 || m.selected[0] != 1 || a.state.buffer.memoryFlags != 14 {
		t.Fatal("did not require cached storage")
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	m.props.memoryTypes[1].propertyFlags = 6
	before := m.allocates
	if _, err := NewVkSharedTensorArena(context.Background(), 64); err == nil {
		t.Fatal("silently fell back to uncached")
	}
	if m.allocates != before || m.destroys != 2 || m.frees != 1 {
		t.Fatal("failed selection leaked/mutated allocation")
	}
	legacy, err := NewVkTensorArena(context.Background(), 64)
	if err != nil {
		t.Fatal(err)
	}
	tensor, _ := legacy.AllocF32(context.Background(), 4)
	if err := tensor.WithCPURead(context.Background(), func([]float32) error { t.Fatal("borrowed uncached"); return nil }); err == nil {
		t.Fatal("uncached borrow admitted")
	}
	if err := legacy.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestVulkanOfflineCPUAccessIdentityRangesAndPrefix(t *testing.T) {
	a, m := sharedMockArena(t)
	ctx := context.Background()
	padding, _ := a.AllocF32(ctx, 1)
	tensor, _ := a.AllocF32(ctx, 2, 2)
	if err := padding.Upload(ctx, []float32{77}); err != nil {
		t.Fatal(err)
	}
	b := a.state.buffer
	memory := unsafe.Slice((*float32)(b.mapped), 16)
	memory[3], memory[8] = 88, 99
	ptr := (*float32)(unsafe.Add(b.mapped, uintptr(tensor.offset)))
	if err := tensor.WithCPUWrite(ctx, func(v []float32) error {
		if len(v) != 4 || cap(v) != 4 || &v[0] != ptr {
			t.Fatal("copy, wrong offset or unbounded view")
		}
		for i := range v {
			v[i] = float32(i + 1)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	prefix, err := tensor.PrefixRows(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := prefix.WithCPURead(ctx, func(v []float32) error {
		if len(v) != 2 || &v[0] != ptr || v[0] != 1 || v[1] != 2 {
			t.Fatal("prefix storage identity")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if memory[0] != 77 || memory[3] != 88 || memory[8] != 99 || m.allocates != 1 {
		t.Fatal("canary or allocation changed")
	}
	owned := make([]float32, 4)
	if err := tensor.Download(ctx, owned); err != nil {
		t.Fatal(err)
	}
	for i, v := range owned {
		if v != float32(i+1) {
			t.Fatal("producer not visible in GPU-bound storage")
		}
	}
}

func TestVulkanOfflineCPUAccessFailureAndPanicRelease(t *testing.T) {
	a, _ := sharedMockArena(t)
	ctx, cancel := context.WithCancel(context.Background())
	tensor, _ := a.AllocF32(ctx, 2)
	sentinel := errors.New("callback failed")
	if err := tensor.WithCPUWrite(ctx, func(v []float32) error { v[0] = 3; return sentinel }); !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
	if err := tensor.WithCPURead(ctx, func(v []float32) error {
		if v[0] != 3 {
			t.Fatal("writes incorrectly rolled back")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	func() {
		defer func() {
			if recover() != "panic" {
				t.Fatal("lost panic")
			}
		}()
		_ = tensor.WithCPUWrite(ctx, func([]float32) error { panic("panic") })
	}()
	if len(vkLane) != 0 {
		t.Fatal("panic leaked lane")
	}
	if err := tensor.WithCPURead(ctx, nil); err == nil {
		t.Fatal("nil callback")
	}
	if err := tensor.WithCPURead(nil, func([]float32) error { return nil }); err == nil {
		t.Fatal("nil context")
	}
	if err := tensor.WithCPURead(ctx, func([]float32) error { cancel(); return nil }); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := tensor.WithCPUWrite(ctx, func([]float32) error { t.Fatal("called cancelled producer"); return nil }); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if err := tensor.WithCPURead(context.Background(), func([]float32) error { return nil }); !errors.Is(err, ErrVulkanClosed) {
		t.Fatal(err)
	}
}

func TestVulkanOfflineCPUAccessExcludesClose(t *testing.T) {
	a, m := sharedMockArena(t)
	tensor, _ := a.AllocF32(context.Background(), 2)
	started := make(chan struct{})
	done := make(chan error, 1)
	err := tensor.WithCPUWrite(context.Background(), func(v []float32) error {
		go func() { close(started); done <- a.Close() }()
		<-started
		if m.frees != 0 || len(vkLane) != 1 {
			t.Fatal("closed live CPU lease")
		}
		// Cancellation at lane admission is safe, even from inside a lease.
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if err := tensor.Upload(ctx, []float32{1, 2}); !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
		v[0] = 7
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if m.frees != 1 {
		t.Fatal("close did not release after lease")
	}
}

func TestVulkanOfflineCPUAccessRejectsInvalidOwners(t *testing.T) {
	a, _ := sharedMockArena(t)
	tensor, _ := a.AllocF32(context.Background(), 2)
	callback := func([]float32) error { t.Fatal("borrow admitted invalid storage"); return nil }
	var nilTensor *VkTensorF32
	if err := nilTensor.WithCPURead(context.Background(), callback); err == nil {
		t.Fatal("nil tensor")
	}
	for _, tc := range []struct {
		name   string
		mutate func(*VkTensorF32)
	}{
		{"offset", func(v *VkTensorF32) { v.offset = 1 }},
		{"size", func(v *VkTensorF32) { v.size = 3 }},
		{"range", func(v *VkTensorF32) { v.offset = 64 }},
		{"owner", func(v *VkTensorF32) { v.arena = nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			view := *tensor
			tc.mutate(&view)
			if err := view.WithCPURead(context.Background(), callback); err == nil {
				t.Fatal("admitted invalid range")
			}
		})
	}
	for _, mode := range []string{"pending", "uncertain", "lost"} {
		t.Run(mode, func(t *testing.T) {
			vkPending = &vkPendingSubmission{buffers: []*VkBuf{a.state.buffer}, uncertain: mode == "uncertain"}
			vkLost = mode == "lost"
			err := tensor.WithCPURead(context.Background(), callback)
			vkPending = nil
			vkLost = false
			want := ErrVulkanInFlight
			if mode == "lost" {
				want = ErrVulkanDeviceLost
			}
			if !errors.Is(err, want) {
				t.Fatal(err)
			}
		})
	}
}

func TestVulkanOfflineCPUAccessFenceAndBarriers(t *testing.T) {
	_, k, _ := newLifetimeMock(t)
	m := mockMemory(t)
	m.props.memoryTypes[0].propertyFlags = 14
	ctx := context.Background()
	a, err := NewVkSharedTensorArena(ctx, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	tensor, _ := a.AllocF32(ctx, 4)
	k.numBuffers = 1
	if err := tensor.WithCPUWrite(ctx, func(v []float32) error { v[0] = 6; return nil }); err != nil {
		t.Fatal(err)
	}
	barriers := 0
	mockVK(t, &vkCmdPipelineBarrier, func(c VkCommandBuffer, src, dst, dep, count uint32, p unsafe.Pointer, buffers uint32, bp unsafe.Pointer, images uint32, ip unsafe.Pointer) {
		barriers++
		access := unsafe.Slice((*uint32)(p), 6)
		if barriers == 1 {
			if src != (vkPipelineStageHost|vkPipelineStageComputeShader) || dst != vkPipelineStageComputeShader || access[4] != (vkAccessHostWrite|vkAccessShaderWrite) {
				t.Error("host acquire barrier")
			}
		} else if src != vkPipelineStageComputeShader || dst != vkPipelineStageHost || access[4] != vkAccessShaderWrite || access[5] != (vkAccessHostRead|vkAccessHostWrite) {
			t.Error("host release barrier")
		}
	})
	afterSubmit, cancel := context.WithCancel(ctx)
	mockVK(t, &vkQueueSubmit, func(VkQueue, uint32, unsafe.Pointer, VkFence) VkResult { cancel(); return VK_SUCCESS })
	if err := k.DispatchF32Context(afterSubmit, 1, 1, 1, []*VkTensorF32{tensor}, nil); !errors.Is(err, ErrVulkanInFlight) {
		t.Fatal(err)
	}
	if err := tensor.WithCPURead(ctx, func([]float32) error { t.Fatal("read before fence"); return nil }); !errors.Is(err, ErrVulkanInFlight) {
		t.Fatal(err)
	}
	if err := VulkanDrain(ctx, 1e9); err != nil {
		t.Fatal(err)
	}
	if err := tensor.WithCPURead(ctx, func(v []float32) error {
		if v[0] != 6 {
			t.Fatal("storage changed")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if barriers != 2 {
		t.Fatal("missing host barriers")
	}
}

func TestVulkanOfflineCPUAccessNoSteadyStateAllocations(t *testing.T) {
	a, _ := sharedMockArena(t)
	tensor, _ := a.AllocF32(context.Background(), 4)
	ctx := context.Background()
	callback := func(v []float32) error { v[0] = 1; return nil }
	if got := testing.AllocsPerRun(100, func() {
		if err := tensor.WithCPUWrite(ctx, callback); err != nil {
			panic(err)
		}
	}); got != 0 {
		t.Fatalf("CPU lease allocations=%g want0", got)
	}
}
