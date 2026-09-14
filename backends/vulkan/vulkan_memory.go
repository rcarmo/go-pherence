package vulkan

import (
	"context"
	"errors"
	"fmt"
)

// ErrVulkanMemoryBudget reports a package-owned allocation budget rejection.
// It is distinct from native OOM and queried descriptor/dispatch limit failures.
var ErrVulkanMemoryBudget = errors.New("Vulkan allocation budget exceeded")

// VulkanMemoryBudget limits explicit VkDeviceMemory allocations owned by this
// package. Zero values mean no additional operator cap. The queried device
// allocation count and reported per-heap size always apply. Bytes are native
// memory requirements including padding, not just the requested buffer size.
// This does not cover opaque driver pipeline/descriptor allocations, other
// processes, system RAM, or physical free-memory availability.
type VulkanMemoryBudget struct {
	MaxBytes       uint64
	MaxAllocations uint32
}

// VulkanMemoryUsage is an independent snapshot. Bytes and Allocations include
// pending, deferred-free and quarantined buffers until vkFreeMemory returns.
// HeapBytes uses VkMemoryHeap indices (at most16). No driver query is performed.
// DeviceLost/InFlight/Uncertain explain retained usage; inspecting it is allowed
// even when new work and destruction are blocked.
type VulkanMemoryUsage struct {
	Bytes                           uint64
	Allocations                     uint32
	HeapBytes                       [16]uint64
	Budget                          VulkanMemoryBudget
	DeviceAllocationLimit           uint32
	InFlight, Uncertain, DeviceLost bool
}

type vkMemoryLedger struct {
	bytes       uint64
	allocations uint32
	heaps       [16]uint64
}

var vkMemoryBudget VulkanMemoryBudget
var vkMemoryUsed vkMemoryLedger

// VulkanSetMemoryBudget changes admission only; it never evicts resources.
// It can be used before initialisation. A cap below current usage, or a pending/
// quarantined submission, rejects the whole update. Zero clears operator caps;
// this cannot bypass queried limits. No environment/default behaviour changes.
func VulkanSetMemoryBudget(budget VulkanMemoryBudget) error {
	if err := vkAcquire(context.Background()); err != nil {
		return err
	}
	defer vkRelease()
	if err := vkStatusLocked(); err != nil {
		return err
	}
	if (budget.MaxBytes != 0 && budget.MaxBytes < vkMemoryUsed.bytes) || (budget.MaxAllocations != 0 && budget.MaxAllocations < vkMemoryUsed.allocations) {
		return fmt.Errorf("%w: new cap below live usage", ErrVulkanMemoryBudget)
	}
	vkMemoryBudget = budget
	return nil
}
func VulkanMemoryStats() VulkanMemoryUsage {
	_ = vkAcquire(context.Background())
	defer vkRelease()
	out := VulkanMemoryUsage{Bytes: vkMemoryUsed.bytes, Allocations: vkMemoryUsed.allocations, HeapBytes: vkMemoryUsed.heaps, Budget: vkMemoryBudget, DeviceAllocationLimit: vkLimits.MemoryAllocationCount, InFlight: vkPending != nil, DeviceLost: vkLost}
	if vkPending != nil {
		out.Uncertain = vkPending.uncertain
	}
	return out
}

// Caller owns vkLane throughout check -> allocate -> charge. No other package
// caller can consume allowance between checks; failed construction charges only
// successfully returned memory until its rollback free completes.
func vkCheckMemoryBudgetLocked(bytes uint64) error {
	countLimit := vkLimits.MemoryAllocationCount
	if countLimit == 0 {
		return fmt.Errorf("%w: missing memory allocation count", ErrVulkanLimit)
	}
	if vkMemoryUsed.allocations >= countLimit {
		return fmt.Errorf("%w: device allocation count=%d", ErrVulkanMemoryBudget, countLimit)
	}
	if vkMemoryBudget.MaxAllocations != 0 && vkMemoryUsed.allocations >= vkMemoryBudget.MaxAllocations {
		return fmt.Errorf("%w: operator allocation count=%d", ErrVulkanMemoryBudget, vkMemoryBudget.MaxAllocations)
	}
	if bytes == 0 || bytes > ^uint64(0)-vkMemoryUsed.bytes {
		return fmt.Errorf("%w: allocation byte overflow", ErrVulkanMemoryBudget)
	}
	if vkMemoryBudget.MaxBytes != 0 && (vkMemoryUsed.bytes > vkMemoryBudget.MaxBytes || bytes > vkMemoryBudget.MaxBytes-vkMemoryUsed.bytes) {
		return fmt.Errorf("%w: bytes=%d live=%d cap=%d", ErrVulkanMemoryBudget, bytes, vkMemoryUsed.bytes, vkMemoryBudget.MaxBytes)
	}
	return nil
}
func vkHeapFitsLocked(heap uint32, size, bytes uint64) bool {
	return heap < 16 && vkMemoryUsed.heaps[heap] <= size && bytes <= size-vkMemoryUsed.heaps[heap]
}
func vkChargeMemoryLocked(heap uint32, bytes uint64) {
	// Bounds and overflow already checked, and the lane is still owned.
	vkMemoryUsed.allocations++
	vkMemoryUsed.bytes += bytes
	vkMemoryUsed.heaps[heap] += bytes
}
func vkUnchargeMemoryLocked(heap uint32, bytes uint64) {
	vkMemoryUsed.allocations--
	vkMemoryUsed.bytes -= bytes
	vkMemoryUsed.heaps[heap] -= bytes
}
func vkCheckChargeLocked(heap uint32, bytes uint64) error {
	if heap >= 16 || bytes == 0 || vkMemoryUsed.allocations == 0 || bytes > vkMemoryUsed.bytes || bytes > vkMemoryUsed.heaps[heap] {
		return fmt.Errorf("Vulkan allocation accounting mismatch")
	}
	return nil
}
