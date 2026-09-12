package vulkan

import (
	"math"
	"math/rand"
	"reflect"
	"sync"
	"testing"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/simd/kernels"
)

// Go execution model of the GLSL pair mapping, not shader execution. Each pair
// appears once; reversing/shuffling invocation order cannot change another
// pair's reads. Compare to the pre-existing CPU reference and exact tail bits.
func TestVulkanOfflineRoPEPairMapping(t *testing.T) {
	for _, shape := range [][4]int{{0, 1, 2, 1}, {2, 2, 5, 2}, {1, 3, 9, 3}, {3, 3, 521, 257}, {0, 257, 7, 3}} {
		pos, heads, dim, half := shape[0], shape[1], shape[2], shape[3]
		_, groups, total, freqLen, err := vkRoPEGeometry(pos, heads, dim, half)
		if err != nil {
			t.Fatal(err)
		}
		x := make([]float32, total)
		freqs := make([]float32, freqLen)
		for i := range x {
			x[i] = float32((i*7)%43-21) / 11
		}
		for i := 0; i < freqLen; i += 2 {
			angle := float64(i+1) * 0.17
			freqs[i] = float32(math.Cos(angle))
			freqs[i+1] = float32(math.Sin(angle))
		}
		want := append([]float32(nil), x...)
		kernels.ApplyRoPEPartial(want, freqs, pos, heads, dim, half)
		for seed := int64(0); seed < 8; seed++ {
			got := append([]float32(nil), x...)
			writers := make([]int, total)
			order := rand.New(rand.NewSource(seed)).Perm(int(groups) * 256)
			for _, pair := range order {
				head := pair / half
				if head >= heads {
					continue
				}
				hf := pair % half
				f := (pos*half + hf) * 2
				first := head*dim + hf
				second := first + half
				q0, q1 := got[first], got[second]
				c, s := freqs[f], freqs[f+1]
				got[first] = q0*c - q1*s
				got[second] = q0*s + q1*c
				writers[first]++
				writers[second]++
			}
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("shape%v seed%d index%d got%g want%g", shape, seed, i, got[i], want[i])
				}
				n := 0
				if i%dim < 2*half {
					n = 1
				}
				if writers[i] != n {
					t.Fatal("nonunique pair owner/tail write", i, writers[i])
				}
			}
		}
	}
}
func TestVulkanOfflineRoPEGeometry(t *testing.T) {
	for _, v := range [][4]int{{-1, 1, 2, 1}, {0, 0, 2, 1}, {0, 1, 0, 1}, {0, 1, 2, 0}, {0, 1, 3, 2}, {int(^uint(0) >> 1), 1, 2, 1}, {0, int(^uint(0) >> 1), 2, 1}, {int(1<<32 - 1), 1, 2, 1}, {0, 1, int(1 << 32), 1}} {
		if _, _, _, _, err := vkRoPEGeometry(v[0], v[1], v[2], v[3]); err == nil {
			t.Fatal("bad geometry accepted", v)
		}
	}
	for _, n := range []int{1, 255, 256, 257, 511, 512, 513} {
		p, g, total, freq, err := vkRoPEGeometry(2, n, 5, 1)
		if err != nil || g != uint32((n+255)/256) || total != n*5 || freq != 6 || p != (vkRoPEPush{2, uint32(n), 5, 1}) {
			t.Fatal(p, g, total, freq, err)
		}
	}
	// Inspect admitted edges without allocating their tensors.
	if _, _, _, _, err := vkRoPEGeometry(2147483646, 1, 2, 1); err != nil {
		t.Fatal("last safe doubled frequency extent rejected", err)
	}
	if _, _, _, _, err := vkRoPEGeometry(2147483647, 1, 2, 1); err == nil {
		t.Fatal("overflowed frequency extent")
	}
}
func skipCacheForRepairTest(t *testing.T) {
	// All tests serial. No cache native builds; restore clean once for later tests.
	vkKernelOnce = sync.Once{}
	vkKernelOnce.Do(func() {})
	t.Cleanup(func() { vkKernelOnce = sync.Once{} })
}
func TestVulkanOfflineRepairedWrapperBindings(t *testing.T) {
	for _, name := range []string{"rope", "rms"} {
		t.Run(name, func(t *testing.T) {
			_, k, b := newLifetimeMock(t)
			skipCacheForRepairTest(t)
			k.numBuffers = 2
			k.pushSize = 16
			if name == "rms" {
				k.pushSize = 8
				mockVK(t, &vkRMSNormNoScaleF32, k)
			} else {
				mockVK(t, &vkRoPEPartialF32, k)
			}
			freqs := &VkBuf{device: 100, buf: 31, mem: 32, size: 48}
			b.size = 40
			var bound [][3]uint64
			var push []uint32
			var groups [3]uint32
			mockVK(t, &vkUpdateDescriptorSets, func(d VkDevice, n uint32, p unsafe.Pointer, c uint32, q unsafe.Pointer) {
				for i := uint32(0); i < n; i++ {
					info := *(*unsafe.Pointer)(unsafe.Add(p, uintptr(i)*64+48))
					bound = append(bound, [3]uint64{uint64(*(*VkBuffer)(info)), *(*uint64)(unsafe.Add(info, 8)), *(*uint64)(unsafe.Add(info, 16))})
				}
			})
			mockVK(t, &vkCmdPushConstants, func(c VkCommandBuffer, l VkPipelineLayout, stage, off, n uint32, p unsafe.Pointer) {
				push = append([]uint32(nil), unsafe.Slice((*uint32)(p), n/4)...)
				if stage != 0x20 || off != 0 {
					t.Fatal("push header")
				}
			})
			mockVK(t, &vkCmdDispatch, func(c VkCommandBuffer, x, y, z uint32) { groups = [3]uint32{x, y, z} })
			var err error
			if name == "rope" {
				err = VkRoPEPartialF32(b, freqs, 2, 2, 5, 2)
				if !reflect.DeepEqual(push, []uint32{2, 2, 5, 2}) {
					t.Fatal("RoPEpush", push)
				}
				if !reflect.DeepEqual(bound, [][3]uint64{{uint64(b.buf), 0, 40}, {31, 0, 48}}) {
					t.Fatal("RoPEbindings", bound)
				}
			} else {
				err = VkRMSNormNoScaleF32(b, 10, 1e-6)
				if !reflect.DeepEqual(push, []uint32{10, math.Float32bits(1e-6)}) {
					t.Fatal("RMSpush", push)
				}
				if !reflect.DeepEqual(bound, [][3]uint64{{uint64(b.buf), 0, 40}, {uint64(b.buf), 0, 40}}) {
					t.Fatal("RMS exact alias", bound)
				}
			}
			if err != nil || groups != ([3]uint32{1, 1, 1}) {
				t.Fatal(err, groups)
			}
		})
	}
}
func TestVulkanOfflineRoPERejectsBeforeRecording(t *testing.T) {
	m, k, b := newLifetimeMock(t)
	skipCacheForRepairTest(t)
	k.numBuffers = 2
	k.pushSize = 16
	mockVK(t, &vkRoPEPartialF32, k)
	b.size = 64
	if err := VkRoPEPartialF32(b, b, 0, 1, 4, 2); err == nil {
		t.Fatal("aliased frequencies accepted")
	}
	small := &VkBuf{device: 100, buf: 31, mem: 32, size: 4}
	if err := VkRoPEPartialF32(b, small, 0, 1, 4, 2); err == nil {
		t.Fatal("short frequencies")
	}
	if err := VkRoPEPartialF32(small, b, 0, 1, 4, 2); err == nil {
		t.Fatal("shortQ")
	}
	if err := VkRoPEPartialF32(b, small, int(^uint(0)>>1), 1, 4, 2); err == nil {
		t.Fatal("overflow position")
	}
	if len(m.events) != 0 {
		t.Fatal("bad wrapper recorded", m.events)
	}
}

// Model the no-scale shader's shared reduction then exact in-place output.
// This proves neither driver rounding nor actual invocation scheduling.
func TestVulkanOfflineRMSAliasSchedule(t *testing.T) {
	for _, n := range []int{1, 3, 255, 256, 257, 513, 1025} {
		input := make([]float32, n)
		for i := range input {
			input[i] = float32(i%23-11) / 7
		}
		var partial [256]float32
		for tid := 0; tid < 256; tid++ {
			for i := tid; i < n; i += 256 {
				partial[tid] += input[i] * input[i]
			}
		}
		for stride := 128; stride > 0; stride /= 2 {
			for tid := 0; tid < stride; tid++ {
				partial[tid] += partial[tid+stride]
			}
		}
		inv := float32(1 / math.Sqrt(float64(partial[0]/float32(n)+1e-6)))
		want := make([]float32, n)
		for i, v := range input {
			want[i] = v * inv
		}
		for seed := int64(0); seed < 4; seed++ {
			got := append([]float32(nil), input...)
			written := make([]bool, n)
			for _, tid := range rand.New(rand.NewSource(seed)).Perm(256) {
				for i := tid; i < n; i += 256 {
					if written[i] {
						t.Fatal("duplicate output owner")
					}
					written[i] = true
					got[i] = got[i] * inv
				}
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatal("inplace differs", n, seed)
			}
		}
	}
}
func FuzzVulkanRoPEGeometry(f *testing.F) {
	f.Add(int64(0), int64(2), int64(5), int64(2))
	f.Add(int64(1<<62), int64(1), int64(2), int64(1))
	f.Fuzz(func(t *testing.T, pos, heads, dim, half int64) {
		p, g, n, freq, err := vkRoPEGeometry(int(pos), int(heads), int(dim), int(half))
		if err != nil {
			return
		}
		if g == 0 || n <= 0 || freq <= 0 || p.RotHalf == 0 || p.RotHalf > p.HeadDim/2 || uint64(p.Heads)*uint64(p.HeadDim) != uint64(n) || ((uint64(p.Pos)+1)*uint64(p.RotHalf)*2) != uint64(freq) {
			t.Fatal("bad admitted geometry")
		}
	})
}

func TestVulkanOfflineRMSNoScaleRejectsWidePush(t *testing.T) {
	m, k, b := newLifetimeMock(t)
	skipCacheForRepairTest(t)
	k.numBuffers = 2
	k.pushSize = 8
	mockVK(t, &vkRMSNormNoScaleF32, k)
	b.size = uint64(1<<32) * 4
	if err := VkRMSNormNoScaleF32(b, int(1<<32), 1e-6); err == nil {
		t.Fatal("truncated n accepted")
	}
	if len(m.events) != 0 {
		t.Fatal("wide n recorded")
	}
}
