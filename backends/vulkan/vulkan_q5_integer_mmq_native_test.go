package vulkan

import (
	"context"
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/half"
)

// The four-block-step schedule must reproduce the base integer-dot kernel bit
// for bit: same Q8_1 bytes, integer block sums, F32 epilogue and block order.
func TestVulkanNativeQ5IntegerDotMMQExact(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_INTDOT_MMQ") != "1" {
		t.Skip("explicit native MMQ comparison")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	ctx, cancel := context.WithDeadline(context.Background(), d.Add(-time.Second))
	defer cancel()
	if !VulkanInitIntegerDot() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("physical device")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	for i, shape := range []struct{ rows, in, out int }{{1, 128, 1}, {1, 128, 64}, {63, 128, 65}, {65, 256, 130}, {2, 384, 191}, {129, 1280, 64}, {1500, 1280, 1280}, {1500, 1280, 5120}, {1500, 5120, 1280}} {
		rng := rand.New(rand.NewSource(int64(17 + i)))
		nb := shape.in / 32
		raw := make([]byte, shape.out*nb*22)
		for b := 0; b < shape.out*nb; b++ {
			p := raw[b*22 : (b+1)*22]
			scale := float32(rng.Float64()*0.02 + 1e-4)
			if rng.Intn(2) == 0 {
				scale = -scale
			}
			if rng.Intn(50) == 0 {
				scale = 0
			}
			binary.LittleEndian.PutUint16(p, half.F32ToF16(scale))
			rng.Read(p[2:])
		}
		x := make([]float32, shape.rows*shape.in)
		for j := range x {
			x[j] = float32(rng.NormFloat64())
			if rng.Intn(997) == 0 {
				x[j] *= 300
			}
		}
		for j := 0; j < 32 && shape.rows*nb > 3; j++ { // one all-zero block
			x[3*32+j] = 0
		}
		if shape.rows*nb > 9 { // invalid-marker blocks: infinity, magnitude >1000, tiny maximum
			x[5*32+3] = float32(math.Inf(1))
			x[7*32+9] = 2000
			for j := 0; j < 32; j++ {
				x[9*32+j] = 0
			}
			x[9*32+4] = 1e-31
		}
		biasValues := make([]float32, shape.out)
		for j := range biasValues {
			biasValues[j] = float32(rng.NormFloat64())
		}
		reader := func(context.Context, int) ([]byte, error) { return raw, nil }
		base, e := NewVkLinearQ5IntegerDotSetStream(ctx, []VkLinearQ5Shape{{shape.out, shape.in}}, reader)
		if e != nil {
			t.Fatal(e)
		}
		nativeClose(t, base)
		mmq, e := NewVkLinearQ5IntegerDotMMQSetStream(ctx, []VkLinearQ5Shape{{shape.out, shape.in}}, reader)
		if e != nil {
			t.Fatal(e)
		}
		nativeClose(t, mmq)
		a, e := NewVkTensorArena(ctx, 4*(shape.rows*shape.in+shape.out+2*shape.rows*shape.out+2*shape.rows*nb*9)+(16<<10))
		if e != nil {
			t.Fatal(e)
		}
		nativeClose(t, a)
		left := nativeGuard(t, a)
		tx := nativeTensor(t, a, x, shape.rows, shape.in)
		bias := nativeTensor(t, a, biasValues, shape.out)
		outBase := nativeTensor(t, a, nil, shape.rows, shape.out)
		outMMQ := nativeTensor(t, a, nil, shape.rows, shape.out)
		qBase := nativeTensor(t, a, nil, shape.rows*nb*9)
		qMMQ := nativeTensor(t, a, nil, shape.rows*nb*9)
		right := nativeGuard(t, a)
		plan := func(op *VkLinearQ5IntegerDotSet, out, q *VkTensorF32) *VkF32Plan {
			stages, e := op.Stages(ctx, 0, out, tx, bias, q)
			if e != nil {
				t.Fatal(e)
			}
			p, e := NewVkF32Plan(ctx, stages)
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, p)
			return p
		}
		pb, pm := plan(base, outBase, qBase), plan(mmq, outMMQ, qMMQ)
		var tb, tm time.Duration
		for repeat := 0; repeat < 5; repeat++ {
			if repeat%2 == 0 {
				tb += nativeRun(t, pb.Run)
				tm += nativeRun(t, pm.Run)
			} else {
				tm += nativeRun(t, pm.Run)
				tb += nativeRun(t, pb.Run)
			}
			qw, qg := nativeDownload(t, qBase), nativeDownload(t, qMMQ)
			for j := range qw {
				if math.Float32bits(qw[j]) != math.Float32bits(qg[j]) {
					t.Fatalf("shape %+v Q8_1 word %d base %08x coop %08x", shape, j, math.Float32bits(qw[j]), math.Float32bits(qg[j]))
				}
			}
			want, got := nativeDownload(t, outBase), nativeDownload(t, outMMQ)
			for j := range want {
				if math.Float32bits(want[j]) != math.Float32bits(got[j]) {
					t.Fatalf("shape %+v repeat %d index %d base %g mmq %g", shape, repeat, j, want[j], got[j])
				}
			}
		}
		t.Logf("INTDOT_MMQ_EXACT rows=%d in=%d out=%d values=%d repeats=5 base=%.6f mmq=%.6f", shape.rows, shape.in, shape.out, shape.rows*shape.out, tb.Seconds()/5, tm.Seconds()/5)
		left()
		right()
	}
	if _, e := NewVkLinearQ5IntegerDotMMQSetStream(ctx, []VkLinearQ5Shape{{64, 96}}, func(context.Context, int) ([]byte, error) { t.Fatal("reader before admission"); return nil, nil }); e == nil {
		t.Fatal("inDim%128 admission")
	}
	t.Log("INTDOT_MMQ_EXACT_PASS")
}
