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

func TestVulkanLSTMLayout(t *testing.T) {
	for _, fixture := range loadLSTMFixtures(t) {
		model, err := NewLSTM(context.Background(), fixture.Config, fixture.Weights)
		if err != nil {
			t.Fatal(err)
		}
		layout, err := describeVulkanLSTM(context.Background(), model, fixture.Frames)
		if err != nil {
			t.Fatal(err)
		}
		directions := lstmDirections(fixture.Config)
		if layout.frames != fixture.Frames || layout.directions != directions || layout.stateSize != fixture.Config.NumLayers*directions*fixture.Config.HiddenSize || len(layout.plans) != fixture.Config.NumLayers || len(layout.hiddenNames) != fixture.Config.NumLayers*directions || len(layout.cellNames) != fixture.Config.NumLayers*directions || layout.outputName == "" || layout.weightBytes == 0 || layout.scratchBytes == 0 {
			t.Fatal("Vulkan LSTM layout")
		}
		defined := map[string]bool{layout.inputName: true}
		shapes := map[string][]int{}
		data := map[string][]float32{}
		for _, tensor := range layout.weights {
			if defined[tensor.name] {
				t.Fatal("duplicate Vulkan LSTM weight")
			}
			defined[tensor.name] = true
			shapes[tensor.name] = tensor.shape
			data[tensor.name] = tensor.data
		}
		for _, tensor := range layout.scratch {
			shapes[tensor.name] = tensor.shape
		}
		for layer, plan := range layout.plans {
			if len(plan) != directions {
				t.Fatal("Vulkan LSTM directions", layer)
			}
			for direction, step := range plan {
				weights := model.layers[layer].Forward
				if direction == 1 {
					weights = model.layers[layer].Reverse
				}
				for name, source := range map[string][]float32{step.weightIH: weights.WeightIH, step.weightHH: weights.WeightHH} {
					rows, columns := shapes[name][0], shapes[name][1]
					for row := 0; row < rows; row++ {
						for column := 0; column < columns; column++ {
							if data[name][column*rows+row] != source[row*columns+column] {
								t.Fatal("Vulkan LSTM packed weight", layer, direction, name, row, column)
							}
						}
					}
				}
				for _, name := range []string{step.input, step.weightIH, step.weightHH, step.biasIH, step.biasHH, step.hidden, step.cell} {
					if !defined[name] && shapes[name] == nil {
						t.Fatal("Vulkan LSTM read before definition", layer, direction, name)
					}
				}
				if step.outputOffset != direction*fixture.Config.HiddenSize || step.reverse != (direction == 1) {
					t.Fatal("Vulkan LSTM direction metadata", layer, direction)
				}
			}
			defined[plan[0].output] = true
		}
		if !reflect.DeepEqual(shapes[layout.outputName], []int{fixture.Frames, directions * fixture.Config.HiddenSize}) {
			t.Fatal("Vulkan LSTM final output shape")
		}
		all := append(append([]vulkanBlockTensor(nil), layout.weights...), layout.scratch...)
		for _, alignment := range []uint64{1, 4, 16, 256} {
			bytes, err := vulkanBlockArenaBytes(all, alignment)
			if err != nil || bytes < layout.weightBytes+layout.scratchBytes {
				t.Fatal("Vulkan LSTM arena", alignment, bytes, err)
			}
		}
		first := layout.weights[0].data[0]
		model.layers[0].Forward.WeightIH[0] = first + 1
		if layout.weights[0].data[0] != first {
			t.Fatal("Vulkan LSTM retained source")
		}
	}
}

func TestVulkanLSTMLayoutFinalCancellationReturnsNil(t *testing.T) {
	fixture := loadLSTMFixtures(t)[2]
	model, err := NewLSTM(context.Background(), fixture.Config, fixture.Weights)
	if err != nil {
		t.Fatal(err)
	}
	count := newPowersetContext(0)
	layout, err := describeVulkanLSTM(count, model, fixture.Frames)
	count.cancel()
	if err != nil || layout == nil || count.calls < 1 {
		t.Fatal(layout, err, count.calls)
	}
	ctx := newPowersetContext(count.calls)
	layout, err = describeVulkanLSTM(ctx, model, fixture.Frames)
	ctx.cancel()
	if layout != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("late cancellation returned layout", layout, err)
	}
}

func TestVulkanLSTMRejectsBeforeDevice(t *testing.T) {
	fixture := loadLSTMFixtures(t)[2]
	model, err := NewLSTM(context.Background(), fixture.Config, fixture.Weights)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := describeVulkanLSTM(nil, model, fixture.Frames); err == nil {
		t.Fatal("nil context")
	}
	if _, err := describeVulkanLSTM(context.Background(), nil, fixture.Frames); err == nil {
		t.Fatal("nil source")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := describeVulkanLSTM(ctx, model, fixture.Frames); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := describeVulkanLSTM(context.Background(), model, 0); err == nil {
		t.Fatal("invalid frames")
	}
	bad := *model
	bad.layers = append([]LSTMLayer(nil), model.layers...)
	bad.layers[0].Forward.WeightIH = append([]float32(nil), model.layers[0].Forward.WeightIH...)
	bad.layers[0].Forward.WeightIH[0] = float32(math.NaN())
	if _, err := describeVulkanLSTM(context.Background(), &bad, fixture.Frames); err == nil {
		t.Fatal("nonfinite weight")
	}
	called := false
	if result, err := newVulkanLSTM(context.Background(), nil, fixture.Frames, func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error) { called = true; return nil, nil }); err == nil || result != nil || called {
		t.Fatal("invalid LSTM reached plan constructor")
	}
	if result, err := newVulkanLSTM(context.Background(), model, fixture.Frames, nil); err == nil || result != nil {
		t.Fatal("nil LSTM plan constructor")
	}
}

func TestVulkanLSTMOwnership(t *testing.T) {
	var zero *VulkanLSTM
	if zero.Stats() != (VulkanLSTMStats{}) {
		t.Fatal("zero Vulkan LSTM stats")
	}
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
	if result, err := zero.Forward(context.Background(), nil, nil, nil); err == nil || result != nil {
		t.Fatal("zero Vulkan LSTM forward")
	}
	first, second := &vulkanBlockTestCloser{}, &vulkanBlockTestCloser{fail: true}
	s := &vulkanLSTMState{gate: make(chan struct{}, 1), resources: []vulkanLSTMCloser{first, second}}
	owner := &VulkanLSTM{s: s}
	copyOwner := *owner
	if err := owner.Close(); err == nil || !s.stopping || s.closed {
		t.Fatal("retained LSTM close")
	}
	if result, err := copyOwner.Forward(context.Background(), nil, nil, nil); !errors.Is(err, vk.ErrVulkanClosed) || result != nil {
		t.Fatal("LSTM forward after partial close", err)
	}
	second.fail = false
	if err := copyOwner.Close(); err != nil {
		t.Fatal(err)
	}
	if !s.closed || first.calls != 1 || second.calls != 2 {
		t.Fatal("LSTM retry ownership", first, second)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	s = &vulkanLSTMState{gate: make(chan struct{}, 1)}
	owner = &VulkanLSTM{s: s}
	s.gate <- struct{}{}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if _, err := owner.acquire(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	<-s.gate
}
