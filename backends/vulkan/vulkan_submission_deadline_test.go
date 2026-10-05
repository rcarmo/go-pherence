package vulkan

import (
	"context"
	"testing"
	"time"
)

func TestVulkanOfflineSubmissionWaitBudget(t *testing.T) {
	if got := vkSubmissionWaitBudget(context.Background()); got != time.Second {
		t.Fatal("legacy bounded wait", got)
	}
	for _, d := range []time.Duration{50 * time.Millisecond, 2 * time.Second, 45 * time.Second} {
		ctx, cancel := context.WithTimeout(context.Background(), d)
		got := vkSubmissionWaitBudget(ctx)
		cancel()
		if got <= 0 || got > d || got < d-20*time.Millisecond {
			t.Fatal("caller deadline ignored", d, got)
		}
	}
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	if got := vkSubmissionWaitBudget(ctx); got != 0 {
		t.Fatal("expired deadline", got)
	}
}

func TestVulkanOfflineSubmissionOutlivesLegacyWait(t *testing.T) {
	m, k, b := newLifetimeMock(t)
	// Driver mock completes after the previous hard-coded one-second boundary.
	// No real device, retries or wall-time performance assertion is involved.
	submitted := time.Now()
	mockVK(t, &vkWaitForFences, func(VkDevice, uint32, *VkFence, uint32, uint64) VkResult {
		m.events = append(m.events, "wait")
		if time.Since(submitted) > 1100*time.Millisecond {
			return VK_SUCCESS
		}
		return VK_TIMEOUT
	})
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if e := k.DispatchContext(ctx, 1, 1, 1, []*VkBuf{b}, nil); e != nil {
		t.Fatal(e)
	}
	if vkPending != nil || eventCount(m, "submit") != 1 {
		t.Fatal("completion/resubmission", m.events)
	}
	if e := b.FreeChecked(); e != nil {
		t.Fatal(e)
	}
	if e := k.Close(); e != nil {
		t.Fatal(e)
	}
}
