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

func TestVulkanBasicBlockLayout(t *testing.T) {
	for _, fixture := range loadBlockFixtures(t) {
		block, err := NewWeSpeakerBasicBlock(context.Background(), fixture.Config, fixture.Weights)
		if err != nil {
			t.Fatal(err)
		}
		layout, err := describeVulkanBasicBlock(context.Background(), block, fixture.Shape)
		if err != nil {
			t.Fatal(err)
		}
		wantOutput, err := block.OutputShape(fixture.Shape)
		if err != nil || layout.input != fixture.Shape || layout.output != wantOutput {
			t.Fatal("Vulkan block shapes", layout.input, layout.output, err)
		}
		wantStages := 6
		wantTensors := 11
		if blockNeedsShortcut(fixture.Config) {
			wantStages = 8
			wantTensors = 14
		}
		if len(layout.steps) != wantStages || len(layout.tensors) != wantTensors || layout.weightBytes == 0 || layout.scratchBytes == 0 {
			t.Fatal("Vulkan block layout counts", len(layout.steps), len(layout.tensors), layout.weightBytes, layout.scratchBytes)
		}
		shapes := map[string][]int{}
		data := map[string][]float32{}
		for _, tensor := range layout.tensors {
			if _, exists := shapes[tensor.name]; exists {
				t.Fatal("duplicate Vulkan block tensor", tensor.name)
			}
			shapes[tensor.name], data[tensor.name] = tensor.shape, tensor.data
		}
		if !reflect.DeepEqual(shapes["input"], []int{fixture.Shape.Channels, fixture.Shape.Frequency, fixture.Shape.Frames}) || !reflect.DeepEqual(shapes["a"], []int{wantOutput.Channels, wantOutput.Frequency, wantOutput.Frames}) || !reflect.DeepEqual(shapes["conv1.weight"], []int{wantOutput.Channels, fixture.Shape.Channels, 3, 3}) {
			t.Fatal("Vulkan block tensor shapes", shapes)
		}
		firstWeight := data["conv1.weight"][0]
		block.weights.Conv1[0] = firstWeight + 1
		if data["conv1.weight"][0] != firstWeight {
			t.Fatal("Vulkan block layout retained source weights")
		}
		for _, prefix := range []string{"bn1", "bn2"} {
			for channel := 0; channel < wantOutput.Channels; channel++ {
				var source WeSpeakerBN
				if prefix == "bn1" {
					source = block.weights.BN1
				} else {
					source = block.weights.BN2
				}
				inv := float32(1 / math.Sqrt(float64(source.RunningVariance[channel])+1e-5))
				wantScale := inv * source.Weight[channel]
				wantShift := source.Bias[channel] - source.RunningMean[channel]*wantScale
				if math.Float32bits(data[prefix+".scale"][channel]) != math.Float32bits(wantScale) || math.Float32bits(data[prefix+".shift"][channel]) != math.Float32bits(wantShift) {
					t.Fatal("prepared BatchNorm", prefix, channel)
				}
			}
		}
		for i, value := range data["relu.scale"] {
			if value != 1 || data["relu.shift"][i] != 0 {
				t.Fatal("ReLU identity coefficients")
			}
		}
		defined := map[string]bool{"input": true}
		for _, tensor := range layout.tensors {
			if tensor.data != nil {
				defined[tensor.name] = true
			}
		}
		for _, step := range layout.steps {
			for _, name := range []string{step.x, step.a, step.b} {
				if name != "" && !defined[name] {
					t.Fatal("Vulkan block read before write", step.op, name)
				}
			}
			defined[step.out] = true
		}
		if !defined["b"] {
			t.Fatal("Vulkan block has no output")
		}
		for _, alignment := range []uint64{1, 4, 16, 256} {
			bytes, err := vulkanBlockArenaBytes(layout.tensors, alignment)
			if err != nil || bytes < layout.weightBytes+layout.scratchBytes {
				t.Fatal("Vulkan block arena bytes", alignment, bytes, err)
			}
		}
	}
}

func TestVulkanBasicBlockLayoutFinalCancellationReturnsNil(t *testing.T) {
	fixture := loadBlockFixtures(t)[1]
	block, err := NewWeSpeakerBasicBlock(context.Background(), fixture.Config, fixture.Weights)
	if err != nil {
		t.Fatal(err)
	}
	count := newPowersetContext(0)
	layout, err := describeVulkanBasicBlock(count, block, fixture.Shape)
	count.cancel()
	if err != nil || layout == nil || count.calls < 1 {
		t.Fatal(layout, err, count.calls)
	}
	ctx := newPowersetContext(count.calls)
	layout, err = describeVulkanBasicBlock(ctx, block, fixture.Shape)
	ctx.cancel()
	if layout != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("late cancellation returned layout", layout, err)
	}
}

func TestVulkanBasicBlockLayoutRejectsBeforeDevice(t *testing.T) {
	fixture := loadBlockFixtures(t)[1]
	block, err := NewWeSpeakerBasicBlock(context.Background(), fixture.Config, fixture.Weights)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := describeVulkanBasicBlock(nil, block, fixture.Shape); err == nil {
		t.Fatal("nil context")
	}
	if _, err := describeVulkanBasicBlock(context.Background(), nil, fixture.Shape); err == nil {
		t.Fatal("nil source")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := describeVulkanBasicBlock(ctx, block, fixture.Shape); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := describeVulkanBasicBlock(context.Background(), block, CHWShape{fixture.Shape.Channels, 0, fixture.Shape.Frames}); err == nil {
		t.Fatal("invalid shape")
	}
	bad := *block
	bad.weights.Conv1 = append([]float32(nil), block.weights.Conv1...)
	bad.weights.Conv1[0] = float32(math.NaN())
	if _, err := describeVulkanBasicBlock(context.Background(), &bad, fixture.Shape); err == nil {
		t.Fatal("nonfinite source")
	}
	bad = *block
	bad.weights.BN1.RunningVariance = append([]float32(nil), block.weights.BN1.RunningVariance...)
	bad.weights.BN1.RunningVariance[0] = -1
	if _, err := describeVulkanBasicBlock(context.Background(), &bad, fixture.Shape); err == nil {
		t.Fatal("negative variance")
	}
	if _, err := vulkanBlockArenaBytes([]vulkanBlockTensor{{shape: []int{int(^uint(0) >> 1), 2}}}, 256); err == nil {
		t.Fatal("arena overflow")
	}
	if _, err := vulkanBlockArenaBytes([]vulkanBlockTensor{{shape: []int{1}}}, 7); err == nil {
		t.Fatal("arena alignment")
	}
	called := false
	if result, err := newVulkanBasicBlock(context.Background(), nil, fixture.Shape, func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error) { called = true; return nil, nil }); err == nil || result != nil || called {
		t.Fatal("invalid source reached plan constructor")
	}
	if result, err := newVulkanBasicBlock(context.Background(), block, fixture.Shape, nil); err == nil || result != nil {
		t.Fatal("nil plan constructor")
	}
}

type vulkanBlockTestCloser struct {
	fail  bool
	calls int
}

func (c *vulkanBlockTestCloser) Close() error {
	c.calls++
	if c.fail {
		return errors.New("retained")
	}
	return nil
}

func TestVulkanBasicBlockOwnership(t *testing.T) {
	var zero *VulkanBasicBlock
	if zero.Stats() != (VulkanBasicBlockStats{}) {
		t.Fatal("zero stats")
	}
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
	if out, shape, err := zero.Forward(context.Background(), nil); err == nil || out != nil || shape != (CHWShape{}) {
		t.Fatal("zero forward")
	}
	first, second := &vulkanBlockTestCloser{}, &vulkanBlockTestCloser{fail: true}
	s := &vulkanBasicBlockState{gate: make(chan struct{}, 1), resources: []vulkanBlockCloser{first, second}}
	owner := &VulkanBasicBlock{s: s}
	copyOwner := *owner
	if err := owner.Close(); err == nil || !s.stopping || s.closed {
		t.Fatal("retained block close")
	}
	if out, shape, err := copyOwner.Forward(context.Background(), nil); !errors.Is(err, vk.ErrVulkanClosed) || out != nil || shape != (CHWShape{}) {
		t.Fatal("forward after partial close", err)
	}
	second.fail = false
	if err := copyOwner.Close(); err != nil {
		t.Fatal(err)
	}
	if !s.closed || first.calls != 1 || second.calls != 2 {
		t.Fatal("retry ownership", first, second)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	s = &vulkanBasicBlockState{gate: make(chan struct{}, 1)}
	owner = &VulkanBasicBlock{s: s}
	s.gate <- struct{}{}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if _, err := owner.acquire(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	<-s.gate
}
