package vulkan

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/rcarmo/go-pherence/half"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// Standalone temporary-shader diagnostic, never selected by a model/default.
func TestVulkanNativeIntegerLinearDiagnostic(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_TEST_INTDOT_LINEAR_DIR")
	if dir == "" {
		t.Skip("explicit packed-dot structural diagnostic")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	if !VulkanInitIntegerDot() {
		t.Fatal("integer-dot init")
	}
	if os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("physical match required")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	for _, shape := range [][3]int{{1, 32, 1}, {33, 96, 63}, {1500, 1280, 1280}, {1500, 1280, 5120}, {1500, 5120, 1280}} {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			m, k, n := shape[0], shape[1], shape[2]
			nb := k / 32
			// Deterministic non-tie activation input and signed/zero Q5 scales. Separate
			// original-byte scalar decoding supplies the approximate arithmetic oracle.
			x := make([]float32, m*k)
			for i := range x {
				x[i] = float32((i*13)%257-128) / 64
			}
			raw := make([]byte, n*nb*22)
			for b := 0; b < n*nb; b++ {
				p := raw[b*22 : (b+1)*22]
				binary.LittleEndian.PutUint16(p, []uint16{0x2400, 0xa400, 0x2800, 0x0000}[b%4])
				for j := 2; j < 22; j++ {
					p[j] = byte(b*11 + j*23)
				}
			}
			packed, err := packLinearQ5Blocks(context.Background(), raw)
			if err != nil {
				t.Fatal(err)
			}
			wf := make([]float32, len(packed))
			for i, w := range packed {
				wf[i] = math.Float32frombits(w)
			}
			bias := make([]float32, n)
			for i := range bias {
				bias[i] = float32(i%17-8) / 32
			}
			load := func(name string, count, push int) *VkComputeKernel {
				c, e := os.ReadFile(dir + "/" + name + "-stripped.spv")
				if e != nil {
					t.Fatal(e)
				}
				kernel, e := VkKernelCreateIntegerDot(c, count, push)
				if e != nil {
					t.Fatal(e)
				}
				nativeClose(t, kernel)
				return kernel
			}
			qkernel, lkernel := load("q8", 2, 4), load("linear", 4, 12)
			a, err := NewVkTensorArena(context.Background(), 160<<20)
			if err != nil {
				t.Fatal(err)
			}
			nativeClose(t, a)
			left := nativeGuard(t, a)
			tx := nativeTensor(t, a, x, m, k)
			tq := nativeTensor(t, a, nil, m*nb*9)
			tw := nativeTensor(t, a, wf, len(wf))
			tb := nativeTensor(t, a, bias, n)
			ty := nativeTensor(t, a, nil, m, n)
			baseout := nativeTensor(t, a, nil, m, n)
			right := nativeGuard(t, a)
			quant := VkF32Stage{Kernel: qkernel, Groups: [3]uint32{uint32((m*nb + 31) / 32), 1, 1}, Tensors: []*VkTensorF32{tx, tq}, PushWords: []uint32{uint32(m * nb)}}
			linear := VkF32Stage{Kernel: lkernel, Groups: [3]uint32{uint32((n + 63) / 64), uint32((m + 63) / 64), 1}, Tensors: []*VkTensorF32{tq, tw, tb, ty}, PushWords: []uint32{uint32(m), uint32(k), uint32(n)}}
			plan, e := NewVkF32Plan(context.Background(), []VkF32Stage{quant, linear})
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, plan)
			op, e := NewVkLinearQ5GroupedF32(context.Background(), raw, n, k)
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, op)
			cancelled, cancel := context.WithCancel(context.Background())
			cancel()
			if e := plan.Run(cancelled); !errors.Is(e, context.Canceled) {
				t.Fatal("pre-cancel", e)
			}
			nativeRun(t, plan.Run)
			got := nativeDownload(t, ty)
			// Bound the expensive independent oracle to all small cases and strategically
			// chosen full-shape edges/interior. No relaxed F32 parity claim.
			indices := []int{0, n - 1, (m * n) / 2, m*n - 1}
			if m*n < 10000 {
				indices = make([]int, m*n)
				for i := range indices {
					indices[i] = i
				}
			}
			for _, index := range indices {
				r, c := index/n, index%n
				var want float32
				for b := 0; b < nb; b++ {
					q, bytes := integerQ8Oracle(x[(r*nb+b)*32 : (r*nb+b+1)*32])
					p := raw[(c*nb+b)*22 : (c*nb+b+1)*22]
					var sum int32
					high := binary.LittleEndian.Uint32(p[2:6])
					for j, v := range bytes {
						w := uint32(p[6+j%16])
						if j >= 16 {
							w >>= 4
						}
						w = (w & 15) | ((high>>uint(j))&1)<<4
						sum += int32(w) * int32(v)
					}
					dw := half.F16ToF32(binary.LittleEndian.Uint16(p))
					dq, sq := half.F16ToF32(uint16(q[0])), half.F16ToF32(uint16(q[0]>>16))
					scaled := float32(sum) * dq
					corrected := scaled - 16*sq
					want = float32(math.FMA(float64(dw), float64(corrected), float64(want)))
				}
				want += bias[c]
				if math.Float32bits(got[index]) != math.Float32bits(want) {
					t.Fatalf("Q5Q8 scalar accumulation index%d got%g want%g", index, got[index], want)
				}
			}
			nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, baseout, tx, tb) })
			baseline := nativeDownload(t, baseout)
			var maxError, sumError float64
			for i, v := range got {
				diff := math.Abs(float64(v - baseline[i]))
				if diff > maxError {
					maxError = diff
				}
				sumError += diff
			}
			t.Logf("INTDOT_DRIFT shape=%v scalarchecked=%d maxAbs=%.9g meanAbs=%.9g", shape, len(indices), maxError, sumError/float64(len(got)))
			if m == 1500 {
				for repeat := 0; repeat < 5; repeat++ {
					timed := func(fn func(context.Context) error) float64 {
						start := time.Now()
						nativeRun(t, fn)
						return time.Since(start).Seconds()
					}
					base := timed(func(ctx context.Context) error { return op.Forward(ctx, baseout, tx, tb) })
					candidate := timed(plan.Run)
					t.Logf("INTDOT_LINEAR_SAMPLE shape=%v repeat=%d baseline=%.9f quant_plus_linear=%.9f", shape, repeat, base, candidate)
				}
			}
			nativeRun(t, plan.Run)
			reuse := nativeDownload(t, ty)
			for i, v := range reuse {
				if math.Float32bits(v) != math.Float32bits(got[i]) {
					t.Fatal("reuse", i)
				}
			}
			left()
			right()
		})
	}
}
