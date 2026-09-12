package community1

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

func TestVulkanResNetTrunkLayout(t *testing.T) {
	for _, fixture := range loadResNetFixtures(t) {
		model, err := NewWeSpeakerResNet34(context.Background(), fixture.Config, fixture.Weights)
		if err != nil {
			t.Fatal(err)
		}
		frames := fixture.Frames
		layout, err := describeVulkanResNetTrunk(context.Background(), model, frames)
		if err != nil {
			t.Fatal(err)
		}
		wantOutput, err := model.FrameShape(frames)
		if err != nil || layout.input != (CHWShape{1, fixture.Config.MelBins, frames}) || layout.output != wantOutput || layout.frames != frames {
			t.Fatal("Vulkan trunk shape", layout.input, layout.output, err)
		}
		if len(layout.weights) != 17 || len(layout.scratch) != 13 || len(layout.plans) != 17 || layout.outputName == "" || layout.weightBytes == 0 || layout.scratchBytes == 0 {
			t.Fatal("Vulkan trunk counts", len(layout.weights), len(layout.scratch), len(layout.plans), layout.outputName, layout.weightBytes, layout.scratchBytes)
		}
		if len(layout.plans[0]) != 2 {
			t.Fatal("Vulkan stem stages")
		}
		stages, projections := 0, 0
		defined := map[string]bool{"input": true}
		shapes := map[string][]int{}
		for _, tensor := range layout.scratch {
			shapes[tensor.name] = tensor.shape
		}
		for _, group := range layout.weights {
			for _, tensor := range group {
				if defined[tensor.name] {
					t.Fatal("duplicate Vulkan trunk weight", tensor.name)
				}
				defined[tensor.name] = true
				shapes[tensor.name] = tensor.shape
			}
		}
		for planIndex, plan := range layout.plans {
			if planIndex > 0 {
				if len(plan) == 8 {
					projections++
				} else if len(plan) != 6 {
					t.Fatal("Vulkan trunk plan length", planIndex, len(plan))
				}
			}
			for _, step := range plan {
				for _, name := range []string{step.x, step.a, step.b} {
					if name != "" && !defined[name] {
						t.Fatal("Vulkan trunk read before write", planIndex, step.op, name)
					}
				}
				if shapes[step.out] == nil {
					t.Fatal("Vulkan trunk unknown output", step.out)
				}
				defined[step.out] = true
				stages++
			}
		}
		if projections != 3 || stages != 104 || !defined[layout.outputName] || !reflect.DeepEqual(shapes[layout.outputName], []int{wantOutput.Channels, wantOutput.Frequency, wantOutput.Frames}) {
			t.Fatal("Vulkan trunk topology", projections, stages, layout.outputName, shapes[layout.outputName])
		}
		all := make([]vulkanBlockTensor, 0, len(layout.scratch)+128)
		for _, group := range layout.weights {
			all = append(all, group...)
		}
		all = append(all, layout.scratch...)
		for _, alignment := range []uint64{1, 4, 16, 256} {
			bytes, err := vulkanBlockArenaBytes(all, alignment)
			if err != nil || bytes < layout.weightBytes+layout.scratchBytes {
				t.Fatal("Vulkan trunk arena", alignment, bytes, err)
			}
		}
		first := layout.weights[0][0].data[0]
		model.stem[0] = first + 1
		if layout.weights[0][0].data[0] != first {
			t.Fatal("Vulkan trunk retained source")
		}
	}
}

func TestVulkanResNetTrunkLayoutRejectsBeforeDevice(t *testing.T) {
	fixture := loadResNetFixtures(t)[0]
	model, err := NewWeSpeakerResNet34(context.Background(), fixture.Config, fixture.Weights)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := describeVulkanResNetTrunk(nil, model, fixture.Frames); err == nil {
		t.Fatal("nil context")
	}
	if _, err := describeVulkanResNetTrunk(context.Background(), nil, fixture.Frames); err == nil {
		t.Fatal("nil source")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := describeVulkanResNetTrunk(ctx, model, fixture.Frames); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := describeVulkanResNetTrunk(context.Background(), model, 0); err == nil {
		t.Fatal("bad frames")
	}
	bad := *model
	bad.stem = append([]float32(nil), model.stem...)
	bad.stem[0] = float32(math.NaN())
	if _, err := describeVulkanResNetTrunk(context.Background(), &bad, fixture.Frames); err == nil {
		t.Fatal("nonfinite stem")
	}
	called := false
	if result, err := newVulkanResNetTrunk(context.Background(), nil, fixture.Frames, func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error) { called = true; return nil, nil }); err == nil || result != nil || called {
		t.Fatal("invalid trunk reached plan constructor")
	}
	if result, err := newVulkanResNetTrunk(context.Background(), model, fixture.Frames, nil); err == nil || result != nil {
		t.Fatal("nil trunk plan constructor")
	}
}

func TestVulkanResNetTrunkOwnership(t *testing.T) {
	var zero *VulkanResNetTrunk
	if zero.Stats() != (VulkanResNetStats{}) {
		t.Fatal("zero trunk stats")
	}
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
	if output, shape, err := zero.ForwardFrames(context.Background(), nil, 0); err == nil || output != nil || shape != (CHWShape{}) {
		t.Fatal("zero trunk forward")
	}
	first, second := &vulkanBlockTestCloser{}, &vulkanBlockTestCloser{fail: true}
	s := &vulkanResNetState{gate: make(chan struct{}, 1), resources: []vulkanResNetCloser{first, second}}
	owner := &VulkanResNetTrunk{s: s}
	copyOwner := *owner
	if err := owner.Close(); err == nil || !s.stopping || s.closed {
		t.Fatal("retained trunk close")
	}
	if output, shape, err := copyOwner.ForwardFrames(context.Background(), nil, 0); !errors.Is(err, vk.ErrVulkanClosed) || output != nil || shape != (CHWShape{}) {
		t.Fatal("trunk forward after partial close", err)
	}
	second.fail = false
	if err := copyOwner.Close(); err != nil {
		t.Fatal(err)
	}
	if !s.closed || first.calls != 1 || second.calls != 2 {
		t.Fatal("trunk retry ownership", first, second)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	s = &vulkanResNetState{gate: make(chan struct{}, 1)}
	owner = &VulkanResNetTrunk{s: s}
	s.gate <- struct{}{}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if _, err := owner.acquire(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	<-s.gate
}
