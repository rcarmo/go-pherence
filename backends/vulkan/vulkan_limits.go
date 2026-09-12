package vulkan

import (
	"context"
	"errors"
	"fmt"
)

// ErrVulkanLimit indicates a request exceeds a queried core limit or no valid
// limit snapshot is available. It does not mean a shader has been validated.
var ErrVulkanLimit = errors.New("Vulkan device limit admission failed")

// VulkanDeviceLimits is an owned snapshot of queried core compute limits.
// WorkgroupSize/Invocations and SharedMemoryBytes are informational until a
// shader contract is checked; Dispatch checks only WorkgroupCount. No optional
// features are enabled or promised (including float16/int8/subgroups).
type VulkanDeviceLimits struct {
	APIVersion                  uint32
	StorageBufferRange          uint32
	PushConstantBytes           uint32
	BoundDescriptorSets         uint32
	PerStageStorageBuffers      uint32
	PerStageResources           uint32
	DescriptorSetStorageBuffers uint32
	WorkgroupCount              [3]uint32
	WorkgroupSize               [3]uint32
	WorkgroupInvocations        uint32
	SharedMemoryBytes           uint32
}

var vkLimits VulkanDeviceLimits

// VulkanLimits returns a value copy, with no live driver call. Valid limits may
// be inspected during an ordinary retained submission. Lost/uncertain devices
// return their quarantine error. Availability is separate from VulkanReady.
func VulkanLimits() (VulkanDeviceLimits, error) {
	if err := vkAcquire(context.Background()); err != nil {
		return VulkanDeviceLimits{}, err
	}
	defer vkRelease()
	if err := vkQuarantineLocked(); err != nil {
		return VulkanDeviceLimits{}, err
	}
	if !vkReady {
		return VulkanDeviceLimits{}, fmt.Errorf("%w: device not initialized", ErrVulkanLimit)
	}
	if err := vkLimits.validate(); err != nil {
		return VulkanDeviceLimits{}, err
	}
	return vkLimits, nil
}

func vkLimitsFromProperties(p vkDeviceProperties) (VulkanDeviceLimits, error) {
	l := p.limits
	out := VulkanDeviceLimits{APIVersion: p.apiVersion, StorageBufferRange: l.maxStorageBufferRange,
		PushConstantBytes: l.maxPushConstantsSize, BoundDescriptorSets: l.maxBoundDescriptorSets,
		PerStageStorageBuffers: l.maxPerStageDescriptorStorageBuffers, PerStageResources: l.maxPerStageResources,
		DescriptorSetStorageBuffers: l.maxDescriptorSetStorageBuffers, WorkgroupCount: l.maxComputeWorkGroupCount,
		WorkgroupSize: l.maxComputeWorkGroupSize, WorkgroupInvocations: l.maxComputeWorkGroupInvocations,
		SharedMemoryBytes: l.maxComputeSharedMemorySize}
	return out, out.validate()
}
func (l VulkanDeviceLimits) validate() error {
	// Match the existing Vulkan1.3 application request, not an inferred feature
	// set. Reject other API variants and incomplete/malformed driver snapshots.
	major := (l.APIVersion >> 22) & 0x7f
	minor := (l.APIVersion >> 12) & 0x3ff
	if l.APIVersion>>29 != 0 || major < 1 || (major == 1 && minor < 3) {
		return fmt.Errorf("%w: Vulkan1.3 required", ErrVulkanLimit)
	}
	if l.StorageBufferRange == 0 || l.PushConstantBytes == 0 || l.BoundDescriptorSets == 0 || l.PerStageStorageBuffers == 0 || l.PerStageResources == 0 || l.DescriptorSetStorageBuffers == 0 || l.WorkgroupInvocations == 0 || l.SharedMemoryBytes == 0 {
		return fmt.Errorf("%w: incomplete core limits", ErrVulkanLimit)
	}
	for i := 0; i < 3; i++ {
		if l.WorkgroupCount[i] == 0 || l.WorkgroupSize[i] == 0 {
			return fmt.Errorf("%w: zero compute dimension", ErrVulkanLimit)
		}
	}
	return nil
}
func vkCheckPipelineLimitsLocked(buffers, pushBytes int) error {
	if err := vkLimits.validate(); err != nil {
		return err
	}
	limit := min(vkLimits.PerStageStorageBuffers, vkLimits.PerStageResources, vkLimits.DescriptorSetStorageBuffers)
	if buffers < 1 || uint64(buffers) > uint64(limit) {
		return fmt.Errorf("%w: storage descriptors=%d max=%d", ErrVulkanLimit, buffers, limit)
	}
	if pushBytes < 0 || uint64(pushBytes) > uint64(vkLimits.PushConstantBytes) {
		return fmt.Errorf("%w: push bytes=%d max=%d", ErrVulkanLimit, pushBytes, vkLimits.PushConstantBytes)
	}
	return nil
}
func vkCheckBufferLimitLocked(size uint64) error {
	if err := vkLimits.validate(); err != nil {
		return err
	}
	if size == 0 || size > uint64(vkLimits.StorageBufferRange) {
		return fmt.Errorf("%w: storage bytes=%d max=%d", ErrVulkanLimit, size, vkLimits.StorageBufferRange)
	}
	return nil
}
func vkCheckDispatchLimitsLocked(k *VkComputeKernel, x, y, z uint32, bufs []*VkBuf) error {
	if err := vkCheckPipelineLimitsLocked(k.numBuffers, k.pushSize); err != nil {
		return err
	}
	for i, n := range [3]uint32{x, y, z} {
		if n == 0 || n > vkLimits.WorkgroupCount[i] {
			return fmt.Errorf("%w: workgroup axis%d=%d max=%d", ErrVulkanLimit, i, n, vkLimits.WorkgroupCount[i])
		}
	}
	for _, b := range bufs {
		if err := vkCheckBufferLimitLocked(b.size); err != nil {
			return err
		}
	}
	return nil
}
