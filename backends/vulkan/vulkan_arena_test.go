package vulkan

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"
	"unsafe"
)

func mustArena(t *testing.T, bytes int) *VkTensorArena {
	t.Helper()
	a, err := NewVkTensorArena(context.Background(), bytes)
	if err != nil {
		t.Fatal(err)
	}
	return a
}
func mustTensor(t *testing.T, a *VkTensorArena, shape ...int) *VkTensorF32 {
	t.Helper()
	v, err := a.AllocF32(context.Background(), shape...)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestVulkanOfflineArenaPackingAndTransfer(t *testing.T) {
	offlineVK(t)
	m := mockMemory(t)
	a := mustArena(t, 64)
	shape := []int{3}
	x := mustTensor(t, a, shape...)
	shape[0] = 99
	y := mustTensor(t, a, 2, 2)
	if x.offset != 0 || x.size != 12 || y.offset != 16 || y.size != 16 {
		t.Fatal("bad packed layout", x, y)
	}
	if !reflect.DeepEqual(x.Shape(), []int{3}) || !reflect.DeepEqual(y.Shape(), []int{2, 2}) || y.Elements() != 4 {
		t.Fatal("shape")
	}
	s := x.Shape()
	s[0] = 9
	if x.Shape()[0] != 3 {
		t.Fatal("borrowed shape")
	}
	storage := m.storage[a.state.buffer.mem]
	for i := range storage {
		storage[i] = 0x5a
	}
	if err := x.Upload(context.Background(), []float32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if err := y.Upload(context.Background(), []float32{4, 5, 6, 7}); err != nil {
		t.Fatal(err)
	}
	for _, i := range []int{12, 13, 14, 15, 32, 63} {
		if storage[i] != 0x5a {
			t.Fatal("padding/outside range clobbered", i)
		}
	}
	out := make([]float32, 3)
	if err := x.Download(context.Background(), out); err != nil || !reflect.DeepEqual(out, []float32{1, 2, 3}) {
		t.Fatal(out, err)
	}
	before := *storage
	if err := x.Upload(context.Background(), []float32{9}); err == nil {
		t.Fatal("short upload")
	}
	if err := x.Upload(context.Background(), []float32{9, 8, 7, 6}); err == nil {
		t.Fatal("long upload")
	}
	if *storage != before {
		t.Fatal("rejected transfer wrote")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	expectErrorIs(t, x.Upload(ctx, []float32{9, 9, 9}), context.Canceled)
	if *storage != before {
		t.Fatal("cancelled upload wrote")
	}
	if a.Stats() != (VkTensorArenaStats{CapacityBytes: 64, UsedBytes: 32, Tensors: 2}) {
		t.Fatal(a.Stats())
	}
	if m.creates != 1 || m.allocates != 1 {
		t.Fatal("view allocated native")
	}
	assertMemory(t, 1, 64)
	// Copies refer to the same state; one close invalidates every view and copy.
	copyA := *a
	copyT := *x
	if err := copyA.Close(); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	expectErrorIs(t, copyT.Download(context.Background(), out), ErrVulkanClosed)
	_, err := a.AllocF32(context.Background(), 1)
	expectErrorIs(t, err, ErrVulkanClosed)
	if m.frees != 1 {
		t.Fatal("copy doublefreed")
	}
	assertMemory(t, 0, 0)
}
func TestVulkanOfflineArenaAdmission(t *testing.T) {
	offlineVK(t)
	m := mockMemory(t)
	a := mustArena(t, 64)
	before := a.Stats()
	for _, shape := range [][]int{nil, {0}, {-1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {int(^uint(0) >> 1), 2}, {17}} {
		if v, err := a.AllocF32(context.Background(), shape...); err == nil || v != nil {
			t.Fatal("invalid shape admitted", shape)
		}
		if a.Stats() != before {
			t.Fatal("failed shape consumed capacity")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := a.AllocF32(ctx, 1)
	expectErrorIs(t, err, context.Canceled)
	if a.Stats() != before {
		t.Fatal("cancel consumed capacity")
	}
	// Alignment padding is capacity, not user tensor storage.
	mustTensor(t, a, 13) //52bytes, next aligned offset64, no room
	before = a.Stats()
	if _, err := a.AllocF32(context.Background(), 1); err == nil {
		t.Fatal("alignment overshoot")
	}
	if a.Stats() != before {
		t.Fatal("alignment fail mutated")
	}
	a.Close()
	_, err = NewVkTensorArena(ctx, 64)
	expectErrorIs(t, err, context.Canceled)
	if m.creates != 1 {
		t.Fatal("cancelled constructor called native")
	}
	if err := (*VkTensorArena)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := (*VkTensorArena)(nil).AllocF32(context.Background(), 1); err == nil {
		t.Fatal("nil arena")
	}
	if err := (*VkTensorF32)(nil).Upload(context.Background(), []float32{1}); err == nil {
		t.Fatal("nil tensor")
	}
}
func TestVulkanOfflineArenaRangeDispatch(t *testing.T) {
	lane, k, _ := newLifetimeMock(t)
	m := mockMemory(t)
	a := mustArena(t, 64)
	x := mustTensor(t, a, 3)
	y := mustTensor(t, a, 2)
	k.numBuffers = 3
	calls := 0
	mockVK(t, &vkUpdateDescriptorSets, func(d VkDevice, n uint32, p unsafe.Pointer, c uint32, q unsafe.Pointer) {
		calls++
		if d != 100 || n != 3 || c != 0 || q != nil {
			t.Fatal("write header")
		}
		for i, want := range [][2]uint64{{0, 12}, {16, 8}, {0, 12}} {
			write := unsafe.Add(p, i*64)
			if *(*uint32)(unsafe.Add(write, 24)) != uint32(i) {
				t.Fatal("binding index")
			}
			info := *(*unsafe.Pointer)(unsafe.Add(write, 48))
			if *(*VkBuffer)(info) != a.state.buffer.buf || *(*uint64)(unsafe.Add(info, 8)) != want[0] || *(*uint64)(unsafe.Add(info, 16)) != want[1] {
				t.Fatal("tensor descriptor", i)
			}
		}
	})
	for i := 0; i < 3; i++ {
		if err := k.DispatchF32Context(context.Background(), 1, 1, 1, []*VkTensorF32{x, y, x}, nil); err != nil {
			t.Fatal(err)
		}
	}
	if calls != 3 || eventCount(lane, "submit") != 3 || m.allocates != 1 {
		t.Fatal("resident reuse")
	}
	ctx, cancel := context.WithCancel(context.Background())
	lane.hook = func(s string) {
		if s == "submit" {
			cancel()
		}
	}
	err := k.DispatchF32Context(ctx, 1, 1, 1, []*VkTensorF32{x, y, x}, nil)
	expectErrorIs(t, err, ErrVulkanInFlight)
	if vkPending == nil || len(vkPending.buffers) != 1 || vkPending.buffers[0] != a.state.buffer {
		t.Fatal("view rather than owner retained")
	}
	expectErrorIs(t, y.Upload(context.Background(), []float32{1, 2}), ErrVulkanInFlight)
	_, err = a.AllocF32(context.Background(), 1)
	expectErrorIs(t, err, ErrVulkanInFlight)
	expectErrorIs(t, a.Close(), ErrVulkanInFlight)
	assertMemory(t, 1, 64)
	if !a.Stats().InFlight || m.frees != 0 {
		t.Fatal("pending arena freed")
	}
	lane.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	if err := y.Upload(context.Background(), []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	assertMemory(t, 0, 0)
	expectErrorIs(t, k.DispatchF32Context(context.Background(), 1, 1, 1, []*VkTensorF32{x, y, x}, nil), ErrVulkanClosed)
}
func TestVulkanOfflineArenaQuarantine(t *testing.T) {
	for _, status := range []VkResult{-1, VK_ERROR_DEVICE_LOST} {
		t.Run((&VulkanCallError{Result: status}).Error(), func(t *testing.T) {
			lane, k, _ := newLifetimeMock(t)
			m := mockMemory(t)
			a := mustArena(t, 64)
			x := mustTensor(t, a, 1)
			lane.submit = status
			err := k.DispatchF32Context(context.Background(), 1, 1, 1, []*VkTensorF32{x}, nil)
			want := ErrVulkanUncertain
			if status == VK_ERROR_DEVICE_LOST {
				want = ErrVulkanDeviceLost
			}
			expectErrorIs(t, err, want)
			expectErrorIs(t, a.Close(), want)
			expectErrorIs(t, x.Upload(context.Background(), []float32{0}), want)
			if m.frees != 0 {
				t.Fatal("quarantine freed arena")
			}
			assertMemory(t, 1, 64)
		})
	}
}
func TestVulkanOfflineArenaConcurrentReservations(t *testing.T) {
	offlineVK(t)
	mockMemory(t)
	a := mustArena(t, 64)
	const n = 16
	var wg sync.WaitGroup
	out := make(chan *VkTensorF32, n)
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			v, err := a.AllocF32(context.Background(), 1)
			if err != nil {
				errs <- err
			} else {
				out <- v
			}
		}()
	}
	wg.Wait()
	close(out)
	close(errs)
	if len(out) != 4 || len(errs) != 12 {
		t.Fatal("concurrent bounds", len(out), len(errs))
	}
	offsets := []int{}
	for v := range out {
		offsets = append(offsets, int(v.offset))
	}
	sort.Ints(offsets)
	if !reflect.DeepEqual(offsets, []int{0, 16, 32, 48}) {
		t.Fatal("overlapping ranges", offsets)
	}
	if a.Stats().Tensors != 4 || a.Stats().UsedBytes != 52 {
		t.Fatal(a.Stats())
	}
	a.Close()
	assertMemory(t, 0, 0)
}
func TestVulkanOfflineArenaRangeAndOwnerGuards(t *testing.T) {
	lane, k, _ := newLifetimeMock(t)
	mockMemory(t)
	a := mustArena(t, 64)
	x := mustTensor(t, a, 4)
	for _, change := range []func(*VkTensorF32){func(v *VkTensorF32) { v.offset = 4 }, func(v *VkTensorF32) { v.offset = 64 }, func(v *VkTensorF32) { v.size = ^uint64(0) }, func(v *VkTensorF32) { v.size = 0 }} {
		v := *x
		change(&v)
		if err := k.DispatchF32Context(context.Background(), 1, 1, 1, []*VkTensorF32{&v}, nil); err == nil {
			t.Fatal("invalid range accepted")
		}
	}
	if len(lane.events) != 0 {
		t.Fatal("invalid range reached native")
	}
	a.state.buffer.device = 999
	if err := x.Upload(context.Background(), []float32{1, 2, 3, 4}); err == nil {
		t.Fatal("owner mismatch")
	}
	if err := a.Close(); err == nil {
		t.Fatal("stale close")
	}
	assertMemory(t, 1, 64)
	a.state.buffer.device = 100
	a.Close()
	assertMemory(t, 0, 0)
}
func TestVulkanOfflineArenaAllocationRollback(t *testing.T) {
	offlineVK(t)
	m := mockMemory(t)
	ctx, cancel := context.WithCancel(context.Background())
	m.hook = func(s string) {
		if s == "map" {
			cancel()
		}
	}
	a, err := NewVkTensorArena(ctx, 64)
	expectErrorIs(t, err, context.Canceled)
	if a != nil || m.frees != 1 {
		t.Fatal("cancelled construction leaked")
	}
	assertMemory(t, 0, 0)
	m.hook = nil
	if err := VulkanSetMemoryBudget(VulkanMemoryBudget{MaxBytes: 32}); err != nil {
		t.Fatal(err)
	}
	a, err = NewVkTensorArena(context.Background(), 64)
	if !errors.Is(err, ErrVulkanMemoryBudget) || a != nil {
		t.Fatal("budget", err)
	}
	assertMemory(t, 0, 0)
}

func TestVulkanOfflineArenaAlignmentAndCountBounds(t *testing.T) {
	for _, alignment := range []uint64{1, 4, 16, 64, 1 << 63} {
		t.Run(fmt.Sprint(alignment), func(t *testing.T) {
			offlineVK(t)
			mockMemory(t)
			vkLimits.StorageBufferOffsetAlignment = alignment
			a := mustArena(t, 64)
			mustTensor(t, a, 1)
			v, err := a.AllocF32(context.Background(), 1)
			if alignment >= 64 {
				if err == nil {
					t.Fatal("alignment exceeded capacity")
				}
			} else if err != nil || v.offset != max(uint64(4), alignment) {
				t.Fatal(v, err)
			}
			a.state.tensors = 65536
			before := a.Stats()
			if _, err := a.AllocF32(context.Background(), 1); err == nil {
				t.Fatal("unbounded view count")
			}
			if a.Stats() != before {
				t.Fatal("count failure mutated")
			}
			a.Close()
		})
	}
}
func TestVulkanOfflineArenaMultiOwnerRetention(t *testing.T) {
	lane, k, _ := newLifetimeMock(t)
	m := mockMemory(t)
	a, b := mustArena(t, 32), mustArena(t, 32)
	x, y := mustTensor(t, a, 1), mustTensor(t, b, 1)
	k.numBuffers = 3
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	lane.hook = func(s string) {
		if s == "submit" {
			cancel()
		}
	}
	expectErrorIs(t, k.DispatchF32Context(ctx, 1, 1, 1, []*VkTensorF32{x, y, x}, nil), ErrVulkanInFlight)
	if len(vkPending.buffers) != 2 || vkPending.buffers[0] != a.state.buffer || vkPending.buffers[1] != b.state.buffer {
		t.Fatal("owner retention")
	}
	expectErrorIs(t, a.Close(), ErrVulkanInFlight)
	expectErrorIs(t, b.Close(), ErrVulkanInFlight)
	assertMemory(t, 2, 128)
	lane.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	a.Close()
	b.Close()
	assertMemory(t, 0, 0)
	if m.frees != 2 {
		t.Fatal("owner cleanup count")
	}
}
func FuzzVulkanF32Shape(f *testing.F) {
	f.Add(int64(3), int64(4), int64(5))
	f.Add(int64(-1), int64(0), int64(1))
	f.Add(int64(1<<60), int64(16), int64(1))
	f.Fuzz(func(t *testing.T, a, b, c int64) {
		dims := []int{int(a), int(b), int(c)}
		n, err := vkF32ShapeBytes(dims)
		if err == nil {
			if a <= 0 || b <= 0 || c <= 0 || n == 0 || n%4 != 0 || n > uint64(int(^uint(0)>>1)) {
				t.Fatal("invalid successful shape")
			}
		}
	})
}
