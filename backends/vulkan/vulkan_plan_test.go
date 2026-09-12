package vulkan

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"
	"unsafe"
)

type planMock struct {
	events       []string
	sets         map[VkDescriptorSet][]vkBufferBinding
	dispatchSets []VkDescriptorSet
	pushes       []uint32
	failure      string
	hook         func(string)
}

func newPlanMock(t *testing.T) (*planMock, *VkComputeKernel, *VkTensorArena, *VkTensorF32) {
	t.Helper()
	_, k, _ := newLifetimeMock(t)
	mockMemory(t)
	a := mustArena(t, 64)
	x := mustTensor(t, a, 1)
	m := &planMock{sets: map[VkDescriptorSet][]vkBufferBinding{}}
	event := func(s string) {
		m.events = append(m.events, s)
		if m.hook != nil {
			m.hook(s)
		}
	}
	result := func(s string) VkResult {
		event(s)
		if m.failure == s {
			return -1
		}
		return VK_SUCCESS
	}
	mockVK(t, &vkCreateDescriptorPool, func(d VkDevice, p, alloc unsafe.Pointer, out *VkDescriptorPool) VkResult {
		if d != 100 || *(*uint32)(p) != 33 {
			t.Error("pool ABI")
		}
		count := *(*uint32)(unsafe.Add(p, 20))
		size := *(*unsafe.Pointer)(unsafe.Add(p, 32))
		if count < 1 || count > 64 || *(*uint32)(size) != 7 || *(*uint32)(unsafe.Add(size, 4)) < count {
			t.Error("descriptor pool limits")
		}
		*out = 300
		if m.failure == "null-pool" {
			*out = 0
		}
		return result("pool")
	})
	mockVK(t, &vkAllocateDescriptorSets, func(d VkDevice, p unsafe.Pointer, out *VkDescriptorSet) VkResult {
		if d != 100 || *(*uint32)(p) != 34 || *(*VkDescriptorPool)(unsafe.Add(p, 16)) != 300 {
			t.Error("set alloc ABI")
		}
		n := *(*uint32)(unsafe.Add(p, 24))
		layouts := *(*unsafe.Pointer)(unsafe.Add(p, 32))
		for i, v := range unsafe.Slice((*VkDescriptorSetLayout)(layouts), n) {
			if v == 0 {
				t.Error("null layout")
			}
			unsafe.Slice(out, n)[i] = VkDescriptorSet(400 + i)
		}
		if m.failure == "null-set" {
			*out = 0
		}
		return result("sets")
	})
	mockVK(t, &vkAllocateCommandBuffers, func(d VkDevice, p unsafe.Pointer, out *VkCommandBuffer) VkResult {
		if d != 100 || *(*VkCommandPool)(unsafe.Add(p, 16)) != 102 || *(*uint32)(unsafe.Add(p, 28)) != 1 {
			t.Error("command alloc ABI")
		}
		*out = 500
		if m.failure == "null-command" {
			*out = 0
		}
		return result("command")
	})
	mockVK(t, &vkCreateFence, func(d VkDevice, p, alloc unsafe.Pointer, out *VkFence) VkResult {
		*out = 600
		if m.failure == "null-fence" {
			*out = 0
		}
		return result("fence")
	})
	mockVK(t, &vkDestroyFence, func(d VkDevice, h VkFence, p unsafe.Pointer) { event("free-fence") })
	mockVK(t, &vkFreeCommandBuffers, func(d VkDevice, p VkCommandPool, n uint32, h *VkCommandBuffer) {
		if p != 102 || n != 1 {
			t.Error("free command ABI")
		}
		event("free-command")
	})
	mockVK(t, &vkDestroyDescriptorPool, func(d VkDevice, h VkDescriptorPool, p unsafe.Pointer) { event("free-pool") })
	mockVK(t, &vkUpdateDescriptorSets, func(d VkDevice, n uint32, p unsafe.Pointer, c uint32, q unsafe.Pointer) {
		set := *(*VkDescriptorSet)(unsafe.Add(p, 16))
		if set == k.descSet {
			t.Error("mutated kernel's own set")
		}
		list := []vkBufferBinding{}
		for i := uint32(0); i < n; i++ {
			w := unsafe.Add(p, uintptr(i)*64)
			if *(*VkDescriptorSet)(unsafe.Add(w, 16)) != set {
				t.Error("mixed set")
			}
			info := *(*unsafe.Pointer)(unsafe.Add(w, 48))
			list = append(list, vkBufferBinding{offset: *(*uint64)(unsafe.Add(info, 8)), size: *(*uint64)(unsafe.Add(info, 16))})
		}
		m.sets[set] = list
		event("update")
	})
	command := func(c VkCommandBuffer) {
		if c != 500 {
			t.Error("used kernel command not plan", c)
		}
	}
	mockVK(t, &vkBeginCommandBuffer, func(c VkCommandBuffer, p unsafe.Pointer) VkResult { command(c); return result("begin") })
	mockVK(t, &vkCmdPipelineBarrier, func(c VkCommandBuffer, src, dst, flags, n uint32, p unsafe.Pointer, nb uint32, pb unsafe.Pointer, ni uint32, pi unsafe.Pointer) {
		command(c)
		if flags != 0 || n != 1 || nb != 0 || ni != 0 || pb != nil || pi != nil {
			t.Error("barrier ABI")
		}
		b := (*vkMemoryBarrier)(p)
		switch {
		case src == 0x4800 && dst == 0x800:
			if b.srcAccessMask != 0x4040 || b.dstAccessMask != 0x60 {
				t.Error("acquire")
			}
			event("acquire")
		case src == 0x800 && dst == 0x800:
			if b.srcAccessMask != 0x40 || b.dstAccessMask != 0x60 {
				t.Error("compute dependency")
			}
			event("between")
		case src == 0x800 && dst == 0x4000:
			if b.srcAccessMask != 0x40 || b.dstAccessMask != 0x6000 {
				t.Error("release")
			}
			event("release")
		default:
			t.Error("unknown barrier")
		}
	})
	mockVK(t, &vkCmdBindPipeline, func(c VkCommandBuffer, p uint32, h VkPipeline) { command(c); event("pipeline") })
	var set VkDescriptorSet
	mockVK(t, &vkCmdBindDescriptorSets, func(c VkCommandBuffer, p uint32, l VkPipelineLayout, f, n uint32, s *VkDescriptorSet, dc uint32, ds unsafe.Pointer) {
		command(c)
		set = *s
		event("bind")
	})
	mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, stage, off, n uint32, p unsafe.Pointer) {
		command(c)
		if off != 0 || n != 4 {
			t.Error("push ABI")
		}
		m.pushes = append(m.pushes, *(*uint32)(p))
		event("push")
	})
	mockVK(t, &vkCmdDispatch, func(c VkCommandBuffer, x, y, z uint32) {
		command(c)
		m.dispatchSets = append(m.dispatchSets, set)
		event("dispatch")
	})
	mockVK(t, &vkEndCommandBuffer, func(c VkCommandBuffer) VkResult { command(c); return result("end") })
	mockVK(t, &vkResetFences, func(d VkDevice, n uint32, f *VkFence) VkResult {
		if d != 100 || n != 1 || *f != 600 {
			t.Error("reset plan fence")
		}
		return result("reset")
	})
	mockVK(t, &vkQueueSubmit, func(q VkQueue, n uint32, p unsafe.Pointer, f VkFence) VkResult {
		if q != 103 || n != 1 || f != 600 || *(*uint32)(unsafe.Add(p, 40)) != 1 {
			t.Error("one plan submission")
		}
		cmd := *(*unsafe.Pointer)(unsafe.Add(p, 48))
		if *(*VkCommandBuffer)(cmd) != 500 {
			t.Error("submit command")
		}
		return result("submit")
	})
	mockVK(t, &vkWaitForFences, func(d VkDevice, n uint32, f *VkFence, all uint32, ns uint64) VkResult {
		if d != 100 || n != 1 || *f != 600 || ns > uint64(10*time.Millisecond) {
			t.Error("wait plan fence")
		}
		event("wait")
		if m.failure == "timeout" {
			return VK_TIMEOUT
		}
		if m.failure == "wait" {
			return -4
		}
		return VK_SUCCESS
	})
	return m, k, a, x
}
func onePlanStage(k *VkComputeKernel, x *VkTensorF32) VkF32Stage {
	return VkF32Stage{Kernel: k, Groups: [3]uint32{1, 1, 1}, Tensors: []*VkTensorF32{x}}
}
func mustPlan(t *testing.T, stages ...VkF32Stage) *VkF32Plan {
	t.Helper()
	p, err := NewVkF32Plan(context.Background(), stages)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestVulkanOfflinePlanRecordAndOwnership(t *testing.T) {
	m, k, a, x := newPlanMock(t)
	y := mustTensor(t, a, 2)
	k.pushSize = 4
	input := []VkF32Stage{onePlanStage(k, x), onePlanStage(k, y)}
	input[0].PushWords = []uint32{17}
	input[1].PushWords = []uint32{23}
	p := mustPlan(t, input...)
	input[0].PushWords[0] = 99
	input[0].Tensors[0] = nil
	input[1].Groups[0] = 0
	start := len(m.events)
	if err := p.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	expected := []string{"update", "update", "begin", "acquire", "pipeline", "bind", "push", "dispatch", "between", "pipeline", "bind", "push", "dispatch", "release", "end", "reset", "submit", "wait"}
	if !reflect.DeepEqual(m.events[start:], expected) {
		t.Fatal("record order", m.events[start:])
	}
	if !reflect.DeepEqual(m.pushes, []uint32{17, 23}) || !reflect.DeepEqual(m.dispatchSets, []VkDescriptorSet{400, 401}) {
		t.Fatal("copied data/independent sets", m.pushes, m.dispatchSets)
	}
	if m.sets[400][0].offset != 0 || m.sets[400][0].size != 4 || m.sets[401][0].offset != 16 || m.sets[401][0].size != 8 {
		t.Fatal("bindings")
	}
	if vkPending != nil {
		t.Fatal("completed submission retained")
	}
	// Re-recorded with persistent resources, no native allocation per run.
	if err := p.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	n := len(m.events)
	copy := *p
	if err := copy.Close(); err != nil {
		t.Fatal(err)
	}
	p.Close()
	if !reflect.DeepEqual(m.events[n:], []string{"free-fence", "free-command", "free-pool"}) {
		t.Fatal("plan teardown", m.events[n:])
	}
	if k.closed || a.Stats().Closed {
		t.Fatal("plan closed caller owners")
	}
	expectErrorIs(t, p.Run(context.Background()), ErrVulkanClosed)
	a.Close()
	k.Close()
}
func TestVulkanOfflinePlanConstructionRollback(t *testing.T) {
	for _, failure := range []string{"pool", "null-pool", "sets", "null-set", "command", "null-command", "fence", "null-fence"} {
		t.Run(failure, func(t *testing.T) {
			m, k, a, x := newPlanMock(t)
			m.failure = failure
			p, err := NewVkF32Plan(context.Background(), []VkF32Stage{onePlanStage(k, x)})
			if err == nil || p != nil {
				t.Fatal("failed construction accepted")
			}
			frees := []string{}
			for _, e := range m.events {
				if len(e) >= 5 && e[:5] == "free-" {
					frees = append(frees, e)
				}
			}
			want := []string{}
			switch failure {
			case "sets", "null-set", "command", "null-command":
				want = []string{"free-pool"}
			case "fence", "null-fence":
				want = []string{"free-command", "free-pool"}
			}
			if !reflect.DeepEqual(frees, want) {
				t.Fatal("rollback order", frees, want)
			}
			if k.closed || a.Stats().Closed || vkPending != nil {
				t.Fatal("rollback changed caller owners")
			}
		})
	}
}
func TestVulkanOfflinePlanPreflight(t *testing.T) {
	m, k, a, x := newPlanMock(t)
	s := onePlanStage(k, x)
	cases := [][]VkF32Stage{nil, make([]VkF32Stage, 65), {{Kernel: nil}}, {{Kernel: k, Groups: [3]uint32{0, 1, 1}, Tensors: []*VkTensorF32{x}}}, {{Kernel: k, Groups: [3]uint32{1, 1, 1}, Tensors: []*VkTensorF32{x}, PushWords: []uint32{1}}}}
	for _, in := range cases {
		if p, err := NewVkF32Plan(context.Background(), in); err == nil || p != nil {
			t.Fatal("invalid plan")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := NewVkF32Plan(ctx, []VkF32Stage{s})
	expectErrorIs(t, err, context.Canceled)
	if len(m.events) != 0 {
		t.Fatal("invalid plan native effects", m.events)
	}
	p := mustPlan(t, s, s)
	n := len(m.events)
	a.Close()
	if err := p.Run(context.Background()); err == nil {
		t.Fatal("closed backing accepted")
	}
	if len(m.events) != n {
		t.Fatal("late preflight mutated descriptors")
	}
	p.Close()
}
func TestVulkanOfflinePlanRetainsAllOwners(t *testing.T) {
	m, k, a, x := newPlanMock(t)
	b := mustArena(t, 32)
	y := mustTensor(t, b, 1)
	k2 := &VkComputeKernel{device: 100, queue: 103, commandPool: 102, pipeline: 20, pipelineLayout: 21, descSetLayout: 22, descPool: 23, descSet: 24, cmdBuf: 25, fence: 26, numBuffers: 1}
	p := mustPlan(t, onePlanStage(k, x), onePlanStage(k2, y), onePlanStage(k, x))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m.hook = func(s string) {
		if s == "submit" {
			cancel()
		}
	}
	expectErrorIs(t, p.Run(ctx), ErrVulkanInFlight)
	if vkPending == nil || vkPending.plan != p.state || len(vkPending.buffers) != 2 {
		t.Fatal("plan retention")
	}
	for _, kernel := range []*VkComputeKernel{k, k2} {
		expectErrorIs(t, kernel.Close(), ErrVulkanInFlight)
	}
	expectErrorIs(t, p.Close(), ErrVulkanInFlight)
	expectErrorIs(t, a.Close(), ErrVulkanInFlight)
	expectErrorIs(t, b.Close(), ErrVulkanInFlight)
	expectErrorIs(t, x.Upload(context.Background(), []float32{1}), ErrVulkanInFlight)
	assertMemory(t, 2, 128)
	m.hook = nil
	n := len(m.events)
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m.events[n:], []string{"wait"}) {
		t.Fatal("drain resubmitted", m.events[n:])
	}
	p.Close()
	a.Close()
	b.Close()
	assertMemory(t, 0, 0)
}
func TestVulkanOfflinePlanCancellationBoundaries(t *testing.T) {
	for _, boundary := range []string{"preflight", "update", "begin", "acquire", "dispatch", "between", "release", "end", "reset", "submit", "wait"} {
		t.Run(boundary, func(t *testing.T) {
			m, k, _, x := newPlanMock(t)
			p := mustPlan(t, onePlanStage(k, x), onePlanStage(k, x))
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
			err := p.Run(ctx)
			expectErrorIs(t, err, context.Canceled)
			if boundary == "submit" {
				expectErrorIs(t, err, ErrVulkanInFlight)
				if vkPending == nil {
					t.Fatal("cancelled submit not retained")
				}
				m.hook = nil
				VulkanDrain(context.Background(), time.Second)
			} else if vkPending != nil {
				t.Fatal("unsubmitted/completed cancellation retained")
			}
			m.hook = nil
			p.Close()
		})
	}
}
func TestVulkanOfflinePlanNativeErrors(t *testing.T) {
	for _, failure := range []string{"begin", "end", "reset", "submit", "wait", "timeout"} {
		t.Run(failure, func(t *testing.T) {
			m, k, _, x := newPlanMock(t)
			p := mustPlan(t, onePlanStage(k, x), onePlanStage(k, x))
			m.failure = failure
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
			defer cancel()
			if err := p.Run(ctx); err == nil {
				t.Fatal("error accepted")
			}
			if failure == "submit" || failure == "wait" || failure == "timeout" {
				if vkPending == nil {
					t.Fatal("submission not retained")
				}
				if err := p.Close(); err == nil {
					t.Fatal("pending plan freed")
				}
				if err := k.Close(); err == nil {
					t.Fatal("pending kernel freed")
				}
				if failure == "timeout" {
					m.failure = ""
					if err := VulkanDrain(context.Background(), time.Second); err != nil {
						t.Fatal(err)
					}
					p.Close()
				}
			} else {
				if vkPending != nil {
					t.Fatal("unsubmitted retained")
				}
				p.Close()
			}
		})
	}
}
func TestVulkanOfflinePlanBoundsAndConcurrentRuns(t *testing.T) {
	m, k, _, x := newPlanMock(t)
	stages := make([]VkF32Stage, 64)
	for i := range stages {
		stages[i] = onePlanStage(k, x)
	}
	p := mustPlan(t, stages...)
	if err := p.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	between := 0
	for _, e := range m.events {
		if e == "between" {
			between++
		}
	}
	if between != 63 {
		t.Fatal("missing chain barriers", between)
	}
	p.Close()
	m, k, _, x = newPlanMock(t)
	p = mustPlan(t, onePlanStage(k, x), onePlanStage(k, x))
	const n = 8
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); errs <- p.Run(context.Background()) }()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	count := 0
	for _, e := range m.events {
		if e == "submit" {
			count++
		}
	}
	if count != n {
		t.Fatal(fmt.Sprint(m.events))
	}
	p.Close()
}

func TestVulkanOfflinePlanCancellationConstruction(t *testing.T) {
	for _, boundary := range []string{"pool", "sets", "command", "fence"} {
		t.Run(boundary, func(t *testing.T) {
			m, k, _, x := newPlanMock(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			m.hook = func(s string) {
				if s == boundary {
					cancel()
				}
			}
			p, err := NewVkF32Plan(ctx, []VkF32Stage{onePlanStage(k, x)})
			expectErrorIs(t, err, context.Canceled)
			if p != nil || vkPending != nil {
				t.Fatal("cancelled plan published")
			}
			tail := m.events[len(m.events)-3:]
			if !reflect.DeepEqual(tail, []string{"free-fence", "free-command", "free-pool"}) {
				t.Fatal("cancel cleanup", m.events)
			}
		})
	}
}
func TestVulkanOfflinePlanClosedFinalKernel(t *testing.T) {
	m, k, _, x := newPlanMock(t)
	last := &VkComputeKernel{device: 100, queue: 103, commandPool: 102, pipeline: 20, pipelineLayout: 21, descSetLayout: 22, descPool: 23, descSet: 24, cmdBuf: 25, fence: 26, numBuffers: 1}
	p := mustPlan(t, onePlanStage(k, x), onePlanStage(last, x))
	if err := last.Close(); err != nil {
		t.Fatal(err)
	}
	n := len(m.events)
	expectErrorIs(t, p.Run(context.Background()), ErrVulkanClosed)
	if len(m.events) != n {
		t.Fatal("late-stage invalidation partially mutated/recorded")
	}
	p.Close()
}
