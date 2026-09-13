package vulkan

import (
	"context"
	"errors"
	"fmt"
	"time"
)

var (
	ErrVulkanInFlight   = errors.New("Vulkan submission still in flight")
	ErrVulkanDeviceLost = errors.New("Vulkan device lost; process-level recovery required")
	ErrVulkanUncertain  = errors.New("Vulkan submission state uncertain; process-level recovery required")
	ErrVulkanClosed     = errors.New("Vulkan resource closed")
)

// VulkanCallError retains the native status. It never implies that a timeout
// or failed submit made its resources idle. errors.As exposes Result.
type VulkanCallError struct {
	Operation string
	Result    VkResult
}

func (e *VulkanCallError) Error() string { return fmt.Sprintf("%s: %d", e.Operation, e.Result) }

// All package API mutation/native calls use this one lane, including command
// pool allocation/free and host mapped access. No per-buffer lock ordering.
// Holding the lane during finite fence waits is deliberate; callbacks must not
// reenter package APIs. Driver calls themselves cannot be pre-empted.
var vkLane = make(chan struct{}, 1)

func vkAcquire(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("Vulkan: nil context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case vkLane <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-vkLane
			return err
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func vkRelease() { <-vkLane }

// Single retained submission bounds quarantine growth. Even after callers
// drop all references, these resources remain reachable and cannot be freed.
type vkPendingSubmission struct {
	kernel    *VkComputeKernel // single-operation owner (nil for a plan)
	plan      *vkF32PlanState
	buffers   []*VkBuf
	uncertain bool
}

var vkPending *vkPendingSubmission
var vkLost bool

func vkQuarantineLocked() error {
	if vkLost {
		return ErrVulkanDeviceLost
	}
	if vkPending != nil && vkPending.uncertain {
		return errors.Join(ErrVulkanInFlight, ErrVulkanUncertain)
	}
	return nil
}

func vkStatusLocked() error {
	if err := vkQuarantineLocked(); err != nil {
		return err
	}
	if vkPending != nil {
		return ErrVulkanInFlight
	}
	return nil
}
func vkDriverError(operation string, result VkResult) error {
	native := &VulkanCallError{operation, result}
	if result == VK_ERROR_DEVICE_LOST {
		vkLost = true
		return errors.Join(ErrVulkanDeviceLost, native)
	}
	return native
}
func vkUsesBufferLocked(b *VkBuf) bool {
	if vkPending == nil {
		return false
	}
	for _, buf := range vkPending.buffers {
		if buf == b {
			return true
		}
	}
	return false
}
func vkUsesKernelLocked(k *VkComputeKernel) bool {
	if vkPending == nil {
		return false
	}
	if vkPending.kernel == k {
		return true
	}
	if p := vkPending.plan; p != nil {
		for _, stage := range p.stages {
			if stage.kernel == k {
				return true
			}
		}
	}
	return false
}
func vkFinishPendingLocked() {
	completed := vkPending
	vkPending = nil
	if completed == nil {
		return
	}
	for _, buf := range completed.buffers {
		if buf.deferredFree {
			_ = buf.freeLocked()
		}
	}
}

// VulkanDrain waits only for the currently retained successful submission's
// fence. It does NOT resubmit commands, reset a fence, use vkQueueWaitIdle or
// destroy pending resources on timeout. budget (including lane admission) must
// be >0 and <=30s; context may bound it further. Lost/uncertain submissions remain quarantined until
// process teardown; a general device-recreation mechanism is not implemented.
// An unconfirmed fence retains resources. A confirmed fence releases pending
// state and services deferred legacy Free requests, even when cancellation races
// completion (Drain then returns the context error). No background worker.
func VulkanDrain(ctx context.Context, budget time.Duration) error {
	if ctx == nil {
		return fmt.Errorf("Vulkan drain: nil context")
	}
	if budget <= 0 || budget > 30*time.Second {
		return fmt.Errorf("invalid Vulkan drain budget")
	}
	ctx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	if vkLost {
		return ErrVulkanDeviceLost
	}
	if vkPending == nil {
		return nil
	}
	if vkPending.uncertain {
		return errors.Join(ErrVulkanInFlight, ErrVulkanUncertain)
	}
	return vkWaitPendingLocked(ctx, budget)
}

func vkWaitPendingLocked(ctx context.Context, budget time.Duration) error {
	end := time.Now().Add(budget)
	if deadline, ok := ctx.Deadline(); ok && deadline.Before(end) {
		end = deadline
	}
	for {
		if err := ctx.Err(); err != nil {
			return errors.Join(ErrVulkanInFlight, err)
		}
		remaining := time.Until(end)
		if remaining <= 0 {
			return errors.Join(ErrVulkanInFlight, context.DeadlineExceeded)
		}
		wait := min(remaining, 10*time.Millisecond)
		p := vkPending
		if p == nil {
			return nil
		}
		if vkWaitForFences == nil {
			return errors.Join(ErrVulkanInFlight, fmt.Errorf("Vulkan wait function unavailable"))
		}
		var device VkDevice
		var fence VkFence
		if p.plan != nil {
			device = p.plan.device
			fence = p.plan.fence
		} else {
			device = p.kernel.device
			fence = p.kernel.fence
		}
		result := vkWaitForFences(device, 1, &fence, 1, uint64(wait))
		if result == VK_SUCCESS {
			vkFinishPendingLocked()
			// If completion raced cancellation resources are safe, but output is not
			// returned as a successful cancelled request.
			return ctx.Err()
		}
		if result != VK_TIMEOUT {
			// Conservative policy for all unexpected wait errors. Do not infer
			// whether a driver error left a usable fence/device.
			p.uncertain = true
			return errors.Join(ErrVulkanInFlight, ErrVulkanUncertain, vkDriverError("vkWaitForFences", result))
		}
		// Avoid a busy spin from a driver/mock returning TIMEOUT before its budget.
		delay := min(time.Millisecond, time.Until(end))
		if delay > 0 {
			timer := time.NewTimer(delay)
			select {
			case <-timer.C:
			case <-ctx.Done():
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
				return errors.Join(ErrVulkanInFlight, ctx.Err())
			}
		}
	}
}

// Close destroys a completed, owned kernel once. Never copies the kernel.
// Pending or lost/uncertain state returns an error without native destruction.
// Successful closure invalidates cached operation pointers: wrappers return
// errors; cache recreation/device restart remains a separate future API.
func (k *VkComputeKernel) Close() error {
	if k == nil {
		return nil
	}
	if err := vkAcquire(context.Background()); err != nil {
		return err
	}
	defer vkRelease()
	if k.closed {
		return nil
	}
	if err := vkQuarantineLocked(); err != nil {
		return err
	}
	if vkUsesKernelLocked(k) {
		return ErrVulkanInFlight
	}
	if k.device == 0 || k.commandPool == 0 || k.device != vkDevice || k.commandPool != vkCmdPool {
		return fmt.Errorf("Vulkan kernel owner mismatch")
	}
	if vkDestroyFence == nil || vkFreeCommandBuffers == nil || vkDestroyDescriptorPool == nil || vkDestroyPipeline == nil || vkDestroyPipelineLayout == nil || vkDestroyDescriptorSetLayout == nil {
		return fmt.Errorf("Vulkan kernel cleanup functions unavailable")
	}
	if k.fence != 0 {
		vkDestroyFence(k.device, k.fence, nil)
	}
	if k.cmdBuf != 0 {
		vkFreeCommandBuffers(k.device, k.commandPool, 1, &k.cmdBuf)
	}
	if k.descPool != 0 {
		vkDestroyDescriptorPool(k.device, k.descPool, nil)
	}
	if k.pipeline != 0 {
		vkDestroyPipeline(k.device, k.pipeline, nil)
	}
	if k.pipelineLayout != 0 {
		vkDestroyPipelineLayout(k.device, k.pipelineLayout, nil)
	}
	if k.descSetLayout != 0 {
		vkDestroyDescriptorSetLayout(k.device, k.descSetLayout, nil)
	}
	k.fence = 0
	k.cmdBuf = 0
	k.descPool = 0
	k.descSet = 0
	k.pipeline = 0
	k.pipelineLayout = 0
	k.descSetLayout = 0
	k.closed = true
	return nil
}
