package vulkan

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
	"unsafe"
)

// Installed after offlineVK/newLifetimeMock: distinct Go-backed mock handles,
// no library loading or driver access. Can model two types sharing one heap.
type memoryMock struct {
	creates, allocates, frees, destroys int
	requirement                         uint64
	props                               vkPhysicalDeviceMemoryProperties
	selected                            []uint32
	storage                             map[VkDeviceMemory]*[64]byte
	hook                                func(string)
}

func mockMemory(t *testing.T) *memoryMock {
	t.Helper()
	m := &memoryMock{requirement: 64, storage: map[VkDeviceMemory]*[64]byte{}}
	m.props.memoryTypeCount = 1
	m.props.memoryHeapCount = 1
	m.props.memoryTypes[0] = vkMemoryType{propertyFlags: 6, heapIndex: 0}
	m.props.memoryHeaps[0].size = 4096
	step := func(s string) {
		if m.hook != nil {
			m.hook(s)
		}
	}
	mockVK(t, &vkCreateBuffer, func(d VkDevice, p, a unsafe.Pointer, out *VkBuffer) VkResult {
		m.creates++
		*out = VkBuffer(100 + m.creates)
		step("create")
		return VK_SUCCESS
	})
	mockVK(t, &vkGetBufferMemoryRequirements, func(d VkDevice, b VkBuffer, p unsafe.Pointer) {
		*(*vkMemoryRequirements)(p) = vkMemoryRequirements{size: m.requirement, alignment: 16, memoryTypeBits: 7}
	})
	mockVK(t, &vkGetPhysicalDeviceMemoryProperties, func(d VkPhysicalDevice, p unsafe.Pointer) { *(*vkPhysicalDeviceMemoryProperties)(p) = m.props })
	mockVK(t, &vkAllocateMemory, func(d VkDevice, p, a unsafe.Pointer, out *VkDeviceMemory) VkResult {
		if d != 100 || *(*uint64)(unsafe.Add(p, 16)) != m.requirement {
			t.Error("allocation owner/requirements")
		}
		m.selected = append(m.selected, *(*uint32)(unsafe.Add(p, 24)))
		m.allocates++
		*out = VkDeviceMemory(200 + m.allocates)
		m.storage[*out] = new([64]byte)
		step("allocate")
		return VK_SUCCESS
	})
	mockVK(t, &vkBindBufferMemory, func(VkDevice, VkBuffer, VkDeviceMemory, uint64) VkResult { step("bind"); return VK_SUCCESS })
	mockVK(t, &vkMapMemory, func(d VkDevice, h VkDeviceMemory, o, n uint64, f uint32, out *unsafe.Pointer) VkResult {
		if n > 64 {
			t.Fatal("mock requested too much")
		}
		*out = unsafe.Pointer(m.storage[h])
		step("map")
		return VK_SUCCESS
	})
	mockVK(t, &vkUnmapMemory, func(VkDevice, VkDeviceMemory) { step("unmap") })
	mockVK(t, &vkDestroyBuffer, func(VkDevice, VkBuffer, unsafe.Pointer) { m.destroys++; step("destroy") })
	mockVK(t, &vkFreeMemory, func(d VkDevice, h VkDeviceMemory, p unsafe.Pointer) { m.frees++; step("free"); delete(m.storage, h) })
	return m
}
func mustMemoryBuffer(t *testing.T) *VkBuf {
	t.Helper()
	b, err := VkBufAlloc(16)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func assertMemory(t *testing.T, count uint32, bytes uint64) {
	t.Helper()
	got := VulkanMemoryStats()
	if got.Allocations != count || got.Bytes != bytes {
		t.Fatalf("usage=%+v want%d/%d", got, count, bytes)
	}
}

func TestVulkanOfflineMemoryBudgetBoundaries(t *testing.T) {
	offlineVK(t)
	m := mockMemory(t)
	if err := VulkanSetMemoryBudget(VulkanMemoryBudget{MaxBytes: 64, MaxAllocations: 1}); err != nil {
		t.Fatal(err)
	}
	b := mustMemoryBuffer(t)
	assertMemory(t, 1, 64)
	if b.size != 16 || b.allocationBytes != 64 {
		t.Fatal("charged requested rather than padded bytes")
	}
	_, err := VkBufAlloc(1)
	expectErrorIs(t, err, ErrVulkanMemoryBudget)
	if m.creates != 1 || m.allocates != 1 {
		t.Fatal("budget failure called native")
	}
	expectErrorIs(t, VulkanSetMemoryBudget(VulkanMemoryBudget{MaxBytes: 63}), ErrVulkanMemoryBudget)
	if got := VulkanMemoryStats(); got.Budget.MaxBytes != 64 || got.Budget.MaxAllocations != 1 {
		t.Fatal("partial budget update")
	}
	m.hook = func(s string) {
		if s == "free" && vkMemoryUsed.bytes != 64 {
			t.Error("uncharged before free returned")
		}
	}
	if err := b.FreeChecked(); err != nil {
		t.Fatal(err)
	}
	b.Free()
	assertMemory(t, 0, 0)
	if m.frees != 1 {
		t.Fatal("double free")
	}
	if err := VulkanSetMemoryBudget(VulkanMemoryBudget{MaxBytes: 63}); err != nil {
		t.Fatal(err)
	}
	_, err = VkBufAlloc(16)
	expectErrorIs(t, err, ErrVulkanMemoryBudget)
	if m.creates != 2 || m.allocates != 1 || m.destroys != 2 {
		t.Fatal("padded-requirement rollback", m)
	}
	assertMemory(t, 0, 0)
	m.hook = nil
	if err := VulkanSetMemoryBudget(VulkanMemoryBudget{}); err != nil {
		t.Fatal(err)
	}
	b = mustMemoryBuffer(t)
	if err := b.FreeChecked(); err != nil {
		t.Fatal(err)
	}
}
func TestVulkanOfflineMemoryDeviceCountAndHeaps(t *testing.T) {
	offlineVK(t)
	m := mockMemory(t)
	vkLimits.MemoryAllocationCount = 1
	a := mustMemoryBuffer(t)
	_, err := VkBufAlloc(16)
	expectErrorIs(t, err, ErrVulkanMemoryBudget)
	if m.creates != 1 {
		t.Fatal("device count not preflighted")
	}
	// Clearing operator budget cannot bypass native count.
	if err := VulkanSetMemoryBudget(VulkanMemoryBudget{}); err != nil {
		t.Fatal(err)
	}
	_, err = VkBufAlloc(16)
	expectErrorIs(t, err, ErrVulkanMemoryBudget)
	if err := a.FreeChecked(); err != nil {
		t.Fatal(err)
	}
	vkLimits.MemoryAllocationCount = 4
	m.props.memoryTypeCount = 3
	m.props.memoryHeapCount = 2
	m.props.memoryTypes[1] = vkMemoryType{propertyFlags: 6, heapIndex: 0}
	m.props.memoryTypes[2] = vkMemoryType{propertyFlags: 6, heapIndex: 1}
	m.props.memoryHeaps[0].size = 64
	m.props.memoryHeaps[1].size = 64
	a = mustMemoryBuffer(t)
	b := mustMemoryBuffer(t)
	if a.heapIndex != 0 || b.heapIndex != 1 || m.selected[len(m.selected)-1] != 2 {
		t.Fatal("heap fallback/alias accounting", m.selected)
	}
	_, err = VkBufAlloc(16)
	if err == nil {
		t.Fatal("overfull heaps accepted")
	}
	assertMemory(t, 2, 128)
	if VulkanMemoryStats().HeapBytes[0] != 64 || VulkanMemoryStats().HeapBytes[1] != 64 {
		t.Fatal("perheap usage")
	}
	if err := a.FreeChecked(); err != nil {
		t.Fatal(err)
	}
	c := mustMemoryBuffer(t)
	if c.heapIndex != 0 {
		t.Fatal("freed heap not reused")
	}
	b.Free()
	c.Free()
	assertMemory(t, 0, 0)
}
func TestVulkanOfflineMemoryStatsAndUninitialisedBudget(t *testing.T) {
	offlineVK(t)
	vkReady = false
	vkLimits = VulkanDeviceLimits{}
	budget := VulkanMemoryBudget{MaxBytes: 128, MaxAllocations: 2}
	if err := VulkanSetMemoryBudget(budget); err != nil {
		t.Fatal(err)
	}
	stats := VulkanMemoryStats()
	if stats.Budget != budget || stats.Bytes != 0 || stats.DeviceAllocationLimit != 0 {
		t.Fatal(stats)
	}
	stats.HeapBytes[0] = 99
	stats.Budget.MaxBytes = 1
	if VulkanMemoryStats().HeapBytes[0] != 0 || VulkanMemoryStats().Budget != budget {
		t.Fatal("borrowed stats")
	}
	vkReady = true
	vkLimits = offlineLimits()
	m := mockMemory(t)
	b := mustMemoryBuffer(t)
	if m.allocates != 1 {
		t.Fatal("allocation missed")
	}
	assertMemory(t, 1, 64)
	b.Free()
	assertMemory(t, 0, 0)
}
func TestVulkanOfflineMemoryDeferredAndQuarantine(t *testing.T) {
	for _, mode := range []string{"drain", "lost", "uncertain"} {
		t.Run(mode, func(t *testing.T) {
			lost := mode == "lost"
			quarantined := mode != "drain"
			lane, k, _ := newLifetimeMock(t)
			m := mockMemory(t)
			b := mustMemoryBuffer(t)
			if lost {
				lane.submit = VK_ERROR_DEVICE_LOST
				expectErrorIs(t, k.Dispatch(1, 1, 1, []*VkBuf{b}, nil), ErrVulkanDeviceLost)
			} else if quarantined {
				lane.submit = -1
				expectErrorIs(t, k.Dispatch(1, 1, 1, []*VkBuf{b}, nil), ErrVulkanUncertain)
			} else {
				cancelAfterSubmit(t, lane, k, b)
			}
			b.Free()
			stats := VulkanMemoryStats()
			if stats.Bytes != 64 || stats.Allocations != 1 || !stats.InFlight || stats.DeviceLost != lost || stats.Uncertain != quarantined {
				t.Fatal("pending not charged", stats)
			}
			if m.frees != 0 || !b.deferredFree {
				t.Fatal("pending buffer freed")
			}
			if err := VulkanSetMemoryBudget(VulkanMemoryBudget{}); err == nil {
				t.Fatal("pending budget changed")
			}
			if quarantined {
				want := ErrVulkanUncertain
				if lost {
					want = ErrVulkanDeviceLost
				}
				expectErrorIs(t, b.FreeChecked(), want)
				expectErrorIs(t, VulkanDrain(context.Background(), time.Millisecond), want)
				assertMemory(t, 1, 64)
				return
			}
			if err := VulkanDrain(context.Background(), time.Second); err != nil {
				t.Fatal(err)
			}
			assertMemory(t, 0, 0)
			if !b.closed || m.frees != 1 {
				t.Fatal("deferred charge not released")
			}
		})
	}
}
func TestVulkanOfflineMemoryCleanupFailureRetainsCharge(t *testing.T) {
	offlineVK(t)
	m := mockMemory(t)
	b := mustMemoryBuffer(t)
	free := vkFreeMemory
	vkFreeMemory = nil
	if err := b.FreeChecked(); err == nil {
		t.Fatal("missing cleanup accepted")
	}
	assertMemory(t, 1, 64)
	if m.frees != 0 || m.destroys != 0 {
		t.Fatal("partial cleanup")
	}
	vkFreeMemory = free
	b.device = 999
	if err := b.FreeChecked(); err == nil {
		t.Fatal("owner mismatch accepted")
	}
	assertMemory(t, 1, 64)
	b.device = 100
	b.Free()
	assertMemory(t, 0, 0)
}
func TestVulkanOfflineMemoryConcurrentAdmission(t *testing.T) {
	offlineVK(t)
	mockMemory(t)
	if err := VulkanSetMemoryBudget(VulkanMemoryBudget{MaxBytes: 128, MaxAllocations: 2}); err != nil {
		t.Fatal(err)
	}
	const n = 16
	var wg sync.WaitGroup
	bufs := make(chan *VkBuf, n)
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			b, err := VkBufAlloc(16)
			if err != nil {
				errs <- err
			} else {
				bufs <- b
			}
		}()
	}
	wg.Wait()
	close(bufs)
	close(errs)
	if len(bufs) != 2 || len(errs) != 14 {
		t.Fatal("concurrent budget admission", len(bufs), len(errs))
	}
	for err := range errs {
		expectErrorIs(t, err, ErrVulkanMemoryBudget)
	}
	assertMemory(t, 2, 128)
	for b := range bufs {
		b.Free()
	}
	assertMemory(t, 0, 0)
}
func TestVulkanOfflineMemoryArithmeticAndGuards(t *testing.T) {
	offlineVK(t)
	vkMemoryUsed.bytes = ^uint64(0) - 3
	vkMemoryUsed.allocations = 1
	expectErrorIs(t, vkCheckMemoryBudgetLocked(4), ErrVulkanMemoryBudget)
	vkMemoryUsed.bytes = 0
	vkMemoryUsed.allocations = vkLimits.MemoryAllocationCount
	expectErrorIs(t, vkCheckMemoryBudgetLocked(1), ErrVulkanMemoryBudget)
	vkMemoryUsed = vkMemoryLedger{}
	vkLimits.MemoryAllocationCount = 0
	expectErrorIs(t, vkCheckMemoryBudgetLocked(1), ErrVulkanLimit)
	vkMemoryUsed.heaps[0] = 65
	if vkHeapFitsLocked(0, 64, 1) || vkHeapFitsLocked(16, 100, 1) {
		t.Fatal("heap underflow/invalid index")
	}
	if err := vkCheckChargeLocked(0, 1); err == nil {
		t.Fatal("invalid uncharge")
	}
}
func TestVulkanOfflineMemoryRollbackChargeOrder(t *testing.T) {
	offlineVK(t)
	m := mockMemory(t)
	mockVK(t, &vkMapMemory, func(VkDevice, VkDeviceMemory, uint64, uint64, uint32, *unsafe.Pointer) VkResult { return -1 })
	m.hook = func(step string) {
		if step == "bind" || step == "destroy" || step == "free" {
			if vkMemoryUsed.bytes != 64 || vkMemoryUsed.allocations != 1 {
				t.Error("rollback lost live charge", step)
			}
		}
	}
	_, err := VkBufAlloc(16)
	if err == nil || errors.Is(err, ErrVulkanMemoryBudget) {
		t.Fatal("native error lost", err)
	}
	assertMemory(t, 0, 0)
	if m.frees != 1 || m.destroys != 1 {
		t.Fatal("rollback cleanup counts")
	}
}

func TestVulkanOfflineMemoryMissingDeferredCleanupRetainsCharge(t *testing.T) {
	lane, k, _ := newLifetimeMock(t)
	m := mockMemory(t)
	b := mustMemoryBuffer(t)
	cancelAfterSubmit(t, lane, k, b)
	b.Free()
	free := vkFreeMemory
	vkFreeMemory = nil
	// Fence completes but native teardown cannot occur. Accounting must continue
	// to report the leak, rather than make its bytes available to new requests.
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	assertMemory(t, 1, 64)
	if b.closed || m.frees != 0 || vkPending != nil {
		t.Fatal("deferred cleanup state")
	}
	vkFreeMemory = free
	b.Free()
	assertMemory(t, 0, 0)
}
