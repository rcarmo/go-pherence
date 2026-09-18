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

func lstmCellSigmoid(value float32) float32 {
	if value >= 0 {
		return float32(1 / (1 + math.Exp(-float64(value))))
	}
	e := math.Exp(float64(value))
	return float32(e / (1 + e))
}

func lstmCellStableTanh(value float32) float32 {
	if value >= 0 {
		e := float32(math.Exp(-2 * float64(value)))
		return (1 - e) / (1 + e)
	}
	e := float32(math.Exp(2 * float64(value)))
	return (e - 1) / (e + 1)
}

func lstmCellModel(gates, cell []float32) ([]float32, []float32) {
	hidden := len(cell)
	h, c := make([]float32, hidden), make([]float32, hidden)
	for unit := 0; unit < hidden; unit++ {
		i := lstmCellSigmoid(gates[unit])
		f := lstmCellSigmoid(gates[hidden+unit])
		g := lstmCellStableTanh(gates[2*hidden+unit])
		o := lstmCellSigmoid(gates[3*hidden+unit])
		c[unit] = f*cell[unit] + i*g
		h[unit] = o * lstmCellStableTanh(c[unit])
	}
	return h, c
}

func TestVulkanOfflineLSTMCellNumericsAndSchedule(t *testing.T) {
	rng := rand.New(rand.NewSource(71))
	comparisons := 0
	for _, hidden := range []int{1, 3, 17, 128, 255, 256} {
		gates, cell := make([]float32, 4*hidden), make([]float32, hidden)
		for i := range gates {
			gates[i] = float32(rng.NormFloat64() * 4)
		}
		for i := range cell {
			cell[i] = float32(rng.NormFloat64() * 2)
		}
		wantH, wantC := lstmCellModel(gates, cell)
		for seed := int64(0); seed < 4; seed++ {
			gotH, gotC := make([]float32, hidden), append([]float32(nil), cell...)
			writers := make([]int, hidden)
			for _, unit := range rand.New(rand.NewSource(seed)).Perm(((hidden + 255) / 256) * 256) {
				if unit >= hidden {
					continue
				}
				i := lstmCellSigmoid(gates[unit])
				f := lstmCellSigmoid(gates[hidden+unit])
				g := lstmCellStableTanh(gates[2*hidden+unit])
				o := lstmCellSigmoid(gates[3*hidden+unit])
				gotC[unit] = f*gotC[unit] + i*g
				gotH[unit] = o * lstmCellStableTanh(gotC[unit])
				writers[unit]++
			}
			if !reflect.DeepEqual(gotH, wantH) || !reflect.DeepEqual(gotC, wantC) {
				t.Fatal("LSTM cell schedule", hidden, seed)
			}
			for _, writes := range writers {
				if writes != 1 {
					t.Fatal("LSTM cell output ownership")
				}
			}
			comparisons += 2 * hidden
		}
	}
	for _, value := range []float32{-math.MaxFloat32, -100, -10, -1, 0, 1, 10, 100, math.MaxFloat32} {
		for _, got := range []float32{lstmCellSigmoid(value), lstmCellStableTanh(value)} {
			if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
				t.Fatal("nonfinite stable activation", value, got)
			}
		}
	}
	t.Logf("LSTM cell schedule comparisons=%d", comparisons)
}

func TestVulkanOfflineLSTMCellBindingsAndAdmission(t *testing.T) {
	lane, kernel, _ := newLifetimeMock(t)
	mockAttentionMemory(t)
	kernel.numBuffers, kernel.pushSize = 4, 4
	op := &VkLSTMCellF32{kernel: kernel}
	arena := mustArena(t, 2048)
	gates := mustTensor(t, arena, 4, 17)
	cell := mustTensor(t, arena, 17)
	hidden := mustTensor(t, arena, 17)
	cellOut := mustTensor(t, arena, 17)
	var ranges [][2]uint64
	var push []uint32
	var groups [3]uint32
	mockVK(t, &vkUpdateDescriptorSets, func(_ VkDevice, n uint32, p unsafe.Pointer, _ uint32, _ unsafe.Pointer) {
		ranges = nil
		for i := uint32(0); i < n; i++ {
			info := *(*unsafe.Pointer)(unsafe.Add(p, uintptr(i)*64+48))
			ranges = append(ranges, [2]uint64{*(*uint64)(unsafe.Add(info, 8)), *(*uint64)(unsafe.Add(info, 16))})
		}
	})
	mockVK(t, &vkCmdPushConstants, func(_ VkCommandBuffer, _ VkPipelineLayout, _ uint32, offset, size uint32, p unsafe.Pointer) {
		if offset != 0 || size != 4 {
			t.Fatal("LSTM cell push ABI")
		}
		push = append([]uint32(nil), unsafe.Slice((*uint32)(p), 1)...)
	})
	mockVK(t, &vkCmdDispatch, func(_ VkCommandBuffer, x, y, z uint32) { groups = [3]uint32{x, y, z} })
	if err := op.Forward(context.Background(), hidden, cellOut, gates, cell); err != nil {
		t.Fatal(err)
	}
	wantRanges := [][2]uint64{{gates.offset, gates.size}, {cell.offset, cell.size}, {hidden.offset, hidden.size}, {cellOut.offset, cellOut.size}}
	if !reflect.DeepEqual(ranges, wantRanges) || !reflect.DeepEqual(push, []uint32{17}) || groups != ([3]uint32{1, 1, 1}) {
		t.Fatal(ranges, push, groups)
	}
	if err := op.Forward(context.Background(), hidden, cell, gates, cell); err != nil {
		t.Fatal("exact cell alias", err)
	}
	for slot := 0; slot < 4; slot++ {
		args := []*VkTensorF32{hidden, cellOut, gates, cell}
		args[slot] = nil
		if _, err := op.Stage(context.Background(), args[0], args[1], args[2], args[3]); err == nil {
			t.Fatal("nil LSTM cell tensor", slot)
		}
	}
	badGates := *gates
	badGates.shape[0] = 3
	if _, err := op.Stage(context.Background(), hidden, cellOut, &badGates, cell); err == nil {
		t.Fatal("bad gate shape")
	}
	badCell := *cell
	badCell.rank = 2
	if _, err := op.Stage(context.Background(), hidden, cellOut, gates, &badCell); err == nil {
		t.Fatal("bad cell rank")
	}
	partial := *cell
	partial.offset += 4
	if _, err := op.Stage(context.Background(), hidden, &partial, gates, cell); err == nil {
		t.Fatal("partial cell alias")
	}
	if _, err := op.Stage(context.Background(), hidden, hidden, gates, cell); err == nil {
		t.Fatal("hidden/cell output alias")
	}
	if _, err := op.Stage(context.Background(), cell, cellOut, gates, cell); err == nil {
		t.Fatal("hidden output/cell input alias")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := op.Stage(ctx, hidden, cellOut, gates, cell)
	expectErrorIs(t, err, context.Canceled)
	if len(lane.events) == 0 {
		// The two valid forwards above must have dispatched; admission failures do not.
		t.Fatal("missing valid LSTM dispatch")
	}
	arena.Close()
	op.Close()
	if err := new(VkLSTMCellF32).Close(); err != nil {
		t.Fatal(err)
	}
}

func TestVulkanOfflineLSTMCellPlanAndContract(t *testing.T) {
	mock, kernel, oldArena, _ := newPlanMock(t)
	oldArena.Close()
	mockAttentionMemory(t)
	arena := mustArena(t, 2048)
	kernel.numBuffers, kernel.pushSize = 4, 4
	op := &VkLSTMCellF32{kernel: kernel}
	gates := mustTensor(t, arena, 4, 1)
	cell := mustTensor(t, arena, 1)
	hidden := mustTensor(t, arena, 1)
	stage, err := op.Stage(context.Background(), hidden, cell, gates, cell)
	if err != nil {
		t.Fatal(err)
	}
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
	contract, err := InspectVulkanShader(spirv_lstm_cell_f32)
	want := VulkanShaderContract{LocalSize: [3]uint32{256, 1, 1}, StorageBindings: 15, PushBytes: 4}
	if err != nil || contract != want {
		t.Fatal(contract, err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	cancel()
	_, err = NewVkLSTMCellF32(ctx)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.WorkgroupSize[0] = 255
	_, err = NewVkLSTMCellF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
}
