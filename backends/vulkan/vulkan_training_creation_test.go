package vulkan

import (
	"errors"
	"testing"
)

func TestVulkanOfflineTrainingPartialOwner(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	errCreate := errors.New("create failed")
	calls := 0
	op, err := newVkTrainingF32Locked(func([]byte, int, int) (*VkComputeKernel, error) {
		calls++
		if calls == 2 {
			return nil, errCreate
		}
		return kernel, nil
	})
	if op != nil || !errors.Is(err, errCreate) || !kernel.closed || calls != 2 {
		t.Fatal("rollback", op, err, calls)
	}
	_, kernel, _ = newLifetimeMock(t)
	calls = 0
	kernel.device = 999
	op, err = newVkTrainingF32Locked(func([]byte, int, int) (*VkComputeKernel, error) {
		calls++
		if calls == 2 {
			return nil, errCreate
		}
		return kernel, nil
	})
	if op == nil || op.adam != kernel || kernel.closed || !errors.Is(err, errCreate) {
		t.Fatal("failed cleanup discarded owner", op, err)
	}
}
