package vulkan

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"reflect"
	"testing"
	"time"
	"unsafe"
)

func unpackLinearQ8Weights(packed []uint32, scales []float32, outDim, inDim int) []float32 {
	out := make([]float32, outDim*inDim)
	for row := 0; row < outDim; row++ {
		for column := 0; column < inDim; column++ {
			index := row*inDim + column
			word := packed[index/4]
			q := int8(uint8(word >> (8 * uint(index&3))))
			out[index] = float32(q) * scales[row]
		}
	}
	return out
}

func linearQ8RegTileModel(t *testing.T, x []float32, packed []uint32, scales, bias []float32, m, k, n int, seed int64) ([]float32, []int) {
	t.Helper()
	out, writers := make([]float32, m*n), make([]int, m*n)
	groupsX, groupsY := (n+31)/32, (m+31)/32
	rng := rand.New(rand.NewSource(seed))
	for _, group := range rng.Perm(groupsX * groupsY) {
		rowBase, colBase := (group/groupsX)*32, (group%groupsX)*32
		var sums [256][4]float32
		for base := 0; base < k; base += 32 {
			var xt, wt [1024]float32
			for i := 0; i < 1024; i++ {
				r, kk := i/32, base+i%32
				if kk < k && rowBase+r < m {
					xt[i] = x[(rowBase+r)*k+kk]
				}
				if kk < k && colBase+r < n {
					index := (colBase+r)*k + kk
					wt[i] = float32(int8(uint8(packed[index/4] >> (8 * uint(index&3)))))
				}
			}
			for _, lane := range rng.Perm(256) {
				cx, ry := lane%16, lane/16
				for kk := 0; kk < 32; kk++ {
					a0, a1 := xt[ry*32+kk], xt[(ry+16)*32+kk]
					b0, b1 := wt[cx*32+kk], wt[(cx+16)*32+kk]
					sums[lane][0] += a0 * b0
					sums[lane][1] += a0 * b1
					sums[lane][2] += a1 * b0
					sums[lane][3] += a1 * b1
				}
			}
		}
		for lane := 0; lane < 256; lane++ {
			cx, ry := lane%16, lane/16
			for q := 0; q < 4; q++ {
				r, c := rowBase+ry+(q/2)*16, colBase+cx+(q%2)*16
				if r < m && c < n {
					out[r*n+c] = sums[lane][q]*scales[c] + bias[c]
					writers[r*n+c]++
				}
			}
		}
	}
	return out, writers
}

func TestVulkanOfflineLinearQ8WeightPackingAndModel(t *testing.T) {
	for _, dims := range [][2]int{{1, 1}, {1, 3}, {2, 2}, {3, 5}, {17, 31}} {
		n, k := dims[0], dims[1]
		weight := make([]float32, n*k)
		for i := range weight {
			weight[i] = float32(math.Sin(float64(i)*.17)) * float32((i%7)+1)
		}
		packed, scales, err := packLinearQ8Weights(weight, n, k)
		if err != nil || len(packed) != (len(weight)+3)/4 || len(scales) != n {
			t.Fatal(dims, len(packed), len(scales), err)
		}
		got := unpackLinearQ8Weights(packed, scales, n, k)
		for row := 0; row < n; row++ {
			maximum := float32(0)
			for _, v := range weight[row*k : (row+1)*k] {
				maximum = max(maximum, float32(math.Abs(float64(v))))
			}
			if scales[row] != maximum/127 {
				t.Fatal("scale", dims, row, scales[row], maximum/127)
			}
			for column, v := range got[row*k : (row+1)*k] {
				if math.Abs(float64(v-weight[row*k+column])) > float64(scales[row])*.501 {
					t.Fatal("nearest Q8", dims, row, column, v, weight[row*k+column])
				}
			}
		}
		for i := len(weight); i < len(packed)*4; i++ {
			if uint8(packed[i/4]>>(8*uint(i&3))) != 0 {
				t.Fatal("padding byte")
			}
		}
	}
	// Exact halfway values: max=127 gives scale1, so nearest-even maps
	// ±2.5 to ±2 and ±3.5 to ±4.
	halfway := []float32{127, 2.5, 3.5, -2.5, -3.5}
	packed, scales, err := packLinearQ8Weights(halfway, 1, len(halfway))
	if err != nil || scales[0] != 1 {
		t.Fatal("halfway pack", scales, err)
	}
	if got := unpackLinearQ8Weights(packed, scales, 1, len(halfway)); !reflect.DeepEqual(got, []float32{127, 2, 4, -2, -4}) {
		t.Fatal("nearest-even changed", got)
	}
	for _, bad := range []struct {
		w    []float32
		n, k int
	}{{nil, 1, 1}, {[]float32{1}, 0, 1}, {[]float32{1}, 1, 2}, {[]float32{float32(math.NaN())}, 1, 1}, {[]float32{float32(math.Inf(1))}, 1, 1}} {
		if p, s, err := packLinearQ8Weights(bad.w, bad.n, bad.k); err == nil || p != nil || s != nil {
			t.Fatal("invalid pack", bad)
		}
	}
	for _, dims := range [][3]int{{1, 1, 1}, {2, 3, 5}, {15, 16, 17}, {17, 31, 19}, {33, 65, 63}, {3, 384, 9}, {2, 1280, 7}} {
		m, k, n := dims[0], dims[1], dims[2]
		x, weight, bias := nativeData(m*k, 101, .5), nativeData(n*k, 102, .5), nativeData(n, 103, .1)
		packed, scales, err = packLinearQ8Weights(weight, n, k)
		if err != nil {
			t.Fatal(err)
		}
		dequant := unpackLinearQ8Weights(packed, scales, n, k)
		want := nativeLinearRef(x, dequant, bias, m, k, n)
		for seed := int64(0); seed < 4; seed++ {
			got, writers := linearQ8RegTileModel(t, x, packed, scales, bias, m, k, n, seed)
			for i, v := range got {
				if writers[i] != 1 || math.Abs(float64(v)-want[i]) > 2e-5+2e-5*math.Abs(want[i]) {
					t.Fatal("source model", dims, seed, i, v, want[i], writers[i])
				}
			}
		}
	}
}

func TestVulkanOfflineLinearQ8WeightBindingsAndLifetime(t *testing.T) {
	lane, kernel, _ := newLifetimeMock(t)
	memory := mockMemory(t)
	kernel.numBuffers, kernel.pushSize = 5, 12
	storage, err := VkBufAlloc(32)
	if err != nil {
		t.Fatal(err)
	}
	op := &VkLinearQ8WeightF32{kernel: kernel, storage: storage, inDim: 2, outDim: 2, weightValues: 4, packedBytes: 4, scaleOffsetBytes: 16}
	a := mustArena(t, 64)
	x, bias, out := mustTensor(t, a, 2, 2), mustTensor(t, a, 2), mustTensor(t, a, 2, 2)
	var ranges [][3]uint64
	var push []uint32
	var groups [3]uint32
	mockVK(t, &vkUpdateDescriptorSets, func(_ VkDevice, count uint32, p unsafe.Pointer, _ uint32, _ unsafe.Pointer) {
		for i := uint32(0); i < count; i++ {
			info := *(*unsafe.Pointer)(unsafe.Add(p, uintptr(i)*64+48))
			ranges = append(ranges, [3]uint64{uint64(*(*VkBuffer)(info)), *(*uint64)(unsafe.Add(info, 8)), *(*uint64)(unsafe.Add(info, 16))})
		}
	})
	mockVK(t, &vkCmdPushConstants, func(_ VkCommandBuffer, _ VkPipelineLayout, _, _, _ uint32, p unsafe.Pointer) {
		push = append([]uint32(nil), unsafe.Slice((*uint32)(p), 3)...)
	})
	mockVK(t, &vkCmdDispatch, func(_ VkCommandBuffer, x, y, z uint32) { groups = [3]uint32{x, y, z} })
	if err := op.Forward(context.Background(), out, x, bias); err != nil {
		t.Fatal(err)
	}
	h := uint64(a.state.buffer.buf)
	if !reflect.DeepEqual(ranges, [][3]uint64{{h, 0, 16}, {uint64(storage.buf), 0, 4}, {uint64(storage.buf), 16, 8}, {h, 16, 8}, {h, 32, 16}}) || !reflect.DeepEqual(push, []uint32{2, 2, 2}) || groups != ([3]uint32{1, 1, 1}) {
		t.Fatal(ranges, push, groups)
	}
	alias := *out
	alias.arena, alias.offset = x.arena, x.offset
	if err := op.Forward(context.Background(), &alias, x, bias); err == nil {
		t.Fatal("output alias")
	}
	ctx, cancel := context.WithCancel(context.Background())
	lane.hook = func(s string) {
		if s == "submit" {
			cancel()
		}
	}
	expectErrorIs(t, op.Forward(ctx, out, x, bias), ErrVulkanInFlight)
	before := memory.frees
	expectErrorIs(t, op.Close(), ErrVulkanInFlight)
	if memory.frees != before {
		t.Fatal("pending free")
	}
	lane.hook = nil
	if err := VulkanDrain(context.Background(), time.Second); err != nil {
		t.Fatal(err)
	}
	copyOwner := *op
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	expectErrorIs(t, copyOwner.Forward(context.Background(), out, x, bias), ErrVulkanClosed)
	if err := copyOwner.Close(); err != nil {
		t.Fatal(err)
	}
	if memory.frees != before+1 || op.kernel != nil || op.storage != nil {
		t.Fatal("cleanup", memory.frees)
	}
	a.Close()
}

func TestVulkanOfflineLinearQ8WeightAdmissionAndContract(t *testing.T) {
	contract, err := InspectVulkanShader(spirv_linear_q8_weight_f32)
	if err != nil || contract != (VulkanShaderContract{LocalSize: [3]uint32{16, 16, 1}, SharedBytes: 8192, StorageBindings: 31, PushBytes: 12}) {
		t.Fatal(contract, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if op, err := NewVkLinearQ8WeightF32(ctx, []float32{1}, 1, 1); op != nil || !errors.Is(err, context.Canceled) {
		t.Fatal(op, err)
	}
	if err := (*VkLinearQ8WeightF32)(nil).Close(); err != nil {
		t.Fatal(err)
	}
	var zero *VkLinearQ8WeightF32
	if err := zero.Forward(context.Background(), nil, nil, nil); err == nil {
		t.Fatal("nil forward")
	}
}
