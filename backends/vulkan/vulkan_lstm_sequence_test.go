package vulkan

import (
	"context"
	"math"
	"math/rand"
	"reflect"
	"testing"
	"time"
	"unsafe"
)

func lstmSequenceModel(input, weightIH, weightHH, biasIH, biasHH, initialHidden, initialCell []float32, frames, inputDim, hidden, outputWidth, outputOffset int, reverse bool) ([]float32, []float32, []float32) {
	output := make([]float32, frames*outputWidth)
	hiddenState, cellState := append([]float32(nil), initialHidden...), append([]float32(nil), initialCell...)
	for step := 0; step < frames; step++ {
		frame := step
		if reverse {
			frame = frames - 1 - step
		}
		gates := make([]float32, 4*hidden)
		for row := range gates {
			var inputProjection, hiddenProjection float32
			for column := 0; column < inputDim; column++ {
				inputProjection += input[frame*inputDim+column] * weightIH[row*inputDim+column]
			}
			for column := 0; column < hidden; column++ {
				hiddenProjection += hiddenState[column] * weightHH[row*hidden+column]
			}
			gates[row] = (inputProjection + biasIH[row]) + (hiddenProjection + biasHH[row])
		}
		nextHidden, nextCell := lstmCellModel(gates, cellState)
		copy(hiddenState, nextHidden)
		copy(cellState, nextCell)
		copy(output[frame*outputWidth+outputOffset:frame*outputWidth+outputOffset+hidden], hiddenState)
	}
	return output, hiddenState, cellState
}

func TestVulkanOfflineLSTMSequenceNumerics(t *testing.T) {
	rng := rand.New(rand.NewSource(83))
	comparisons := 0
	for _, shape := range [][3]int{{1, 1, 1}, {3, 2, 3}, {5, 7, 4}, {9, 5, 17}, {2, 60, 128}} {
		frames, inputDim, hidden := shape[0], shape[1], shape[2]
		input := make([]float32, frames*inputDim)
		weightIH := make([]float32, 4*hidden*inputDim)
		weightHH := make([]float32, 4*hidden*hidden)
		biasIH, biasHH := make([]float32, 4*hidden), make([]float32, 4*hidden)
		h0, c0 := make([]float32, hidden), make([]float32, hidden)
		for _, values := range [][]float32{input, weightIH, weightHH, biasIH, biasHH, h0, c0} {
			for i := range values {
				values[i] = float32(rng.NormFloat64() * .125)
			}
		}
		for _, reverse := range []bool{false, true} {
			output, h, c := lstmSequenceModel(input, weightIH, weightHH, biasIH, biasHH, h0, c0, frames, inputDim, hidden, 2*hidden, hidden, reverse)
			for frame := 0; frame < frames; frame++ {
				for i := 0; i < hidden; i++ {
					value := output[frame*2*hidden+hidden+i]
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
						t.Fatal("nonfinite LSTM sequence")
					}
					comparisons++
				}
			}
			terminalFrame := frames - 1
			if reverse {
				terminalFrame = 0
			}
			if !reflect.DeepEqual(h, output[terminalFrame*2*hidden+hidden:terminalFrame*2*hidden+2*hidden]) || len(c) != hidden {
				t.Fatal("LSTM sequence terminal state", shape, reverse)
			}
		}
	}
	t.Logf("LSTM sequence outputs=%d", comparisons)
}

func TestVulkanOfflineLSTMSequenceBindingsAndAdmission(t *testing.T) {
	_, kernel, _ := newLifetimeMock(t)
	mockAttentionMemory(t)
	kernel.numBuffers, kernel.pushSize = 8, 28
	op := &VkLSTMSequenceF32{kernel: kernel}
	arena := mustArena(t, 2048)
	input := mustTensor(t, arena, 3, 2)
	wih := mustTensor(t, arena, 12, 2)
	whh := mustTensor(t, arena, 12, 3)
	bih, bhh := mustTensor(t, arena, 12), mustTensor(t, arena, 12)
	hidden, cell := mustTensor(t, arena, 3), mustTensor(t, arena, 3)
	output := mustTensor(t, arena, 3, 6)
	var push []uint32
	var groups [3]uint32
	mockVK(t, &vkCmdPushConstants, func(_ VkCommandBuffer, _ VkPipelineLayout, _ uint32, offset, size uint32, p unsafe.Pointer) {
		if offset != 0 || size != 28 {
			t.Fatal("LSTM sequence push ABI")
		}
		push = append([]uint32(nil), unsafe.Slice((*uint32)(p), 7)...)
	})
	mockVK(t, &vkCmdDispatch, func(_ VkCommandBuffer, x, y, z uint32) { groups = [3]uint32{x, y, z} })
	if err := op.Forward(context.Background(), output, input, wih, whh, bih, bhh, hidden, cell, 3, true); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(push, []uint32{3, 2, 3, 6, 3, 1, 0}) || groups != ([3]uint32{1, 1, 1}) {
		t.Fatal(push, groups)
	}
	stage, err := op.Stage(context.Background(), output, input, wih, whh, bih, bhh, hidden, cell, 0, false)
	if err != nil || stage.PushWords[5] != 0 || stage.PushWords[6] != 0 || len(stage.Tensors) != 8 {
		t.Fatal("forward stage", stage, err)
	}
	stage, err = op.StagePacked(context.Background(), output, input, wih, whh, bih, bhh, hidden, cell, 0, false, true)
	if err != nil || stage.PushWords[6] != 1 {
		t.Fatal("packed stage", stage, err)
	}
	for slot := 0; slot < 8; slot++ {
		args := []*VkTensorF32{output, input, wih, whh, bih, bhh, hidden, cell}
		args[slot] = nil
		if _, err := op.Stage(context.Background(), args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], 0, false); err == nil {
			t.Fatal("nil LSTM sequence tensor", slot)
		}
	}
	for _, offset := range []int{-1, 4, 7} {
		if _, err := op.Stage(context.Background(), output, input, wih, whh, bih, bhh, hidden, cell, offset, false); err == nil {
			t.Fatal("LSTM sequence output offset", offset)
		}
	}
	badWeight := *wih
	badWeight.shape[0]--
	if _, err := op.Stage(context.Background(), output, input, &badWeight, whh, bih, bhh, hidden, cell, 0, false); err == nil {
		t.Fatal("bad LSTM sequence weights")
	}
	badOutput := *output
	badOutput.size -= 4
	if _, err := op.Stage(context.Background(), &badOutput, input, wih, whh, bih, bhh, hidden, cell, 0, false); err == nil {
		t.Fatal("bad LSTM sequence storage")
	}
	alias := *output
	alias.arena = input.arena
	alias.offset = input.offset
	if _, err := op.Stage(context.Background(), &alias, input, wih, whh, bih, bhh, hidden, cell, 0, false); err == nil {
		t.Fatal("LSTM sequence output alias")
	}
	if _, err := op.Stage(context.Background(), output, input, wih, whh, bih, bhh, hidden, hidden, 0, false); err == nil {
		t.Fatal("LSTM sequence state alias")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = op.Stage(ctx, output, input, wih, whh, bih, bhh, hidden, cell, 0, false)
	expectErrorIs(t, err, context.Canceled)
	arena.Close()
	op.Close()
}

func TestVulkanOfflineLSTMSequencePlanAndContract(t *testing.T) {
	mock, kernel, oldArena, _ := newPlanMock(t)
	oldArena.Close()
	mockAttentionMemory(t)
	arena := mustArena(t, 2048)
	kernel.numBuffers, kernel.pushSize = 8, 28
	op := &VkLSTMSequenceF32{kernel: kernel}
	input := mustTensor(t, arena, 1, 1)
	wih := mustTensor(t, arena, 4, 1)
	whh := mustTensor(t, arena, 4, 1)
	bih, bhh := mustTensor(t, arena, 4), mustTensor(t, arena, 4)
	hidden, cell := mustTensor(t, arena, 1), mustTensor(t, arena, 1)
	output := mustTensor(t, arena, 1, 1)
	stage, err := op.Stage(context.Background(), output, input, wih, whh, bih, bhh, hidden, cell, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	plan := mustPlan(t, stage)
	ctx, cancel := context.WithCancel(context.Background())
	mock.hook = func(event string) {
		if event == "submit" {
			cancel()
		}
	}
	expectErrorIs(t, plan.Run(ctx), ErrVulkanInFlight)
	expectErrorIs(t, op.Close(), ErrVulkanInFlight)
	expectErrorIs(t, arena.Close(), ErrVulkanInFlight)
	mock.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	plan.Close()
	op.Close()
	arena.Close()

	offlineVK(t)
	contract, err := InspectVulkanShader(spirv_lstm_sequence_f32)
	want := VulkanShaderContract{LocalSize: [3]uint32{256, 1, 1}, SharedBytes: 1024, StorageBindings: 255, PushBytes: 28}
	if err != nil || contract != want {
		t.Fatal(contract, err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	cancel()
	_, err = NewVkLSTMSequenceF32(ctx)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.WorkgroupSize[0] = 255
	_, err = NewVkLSTMSequenceF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
}
