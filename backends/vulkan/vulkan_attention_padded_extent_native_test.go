package vulkan

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

func TestVulkanNativePaddedExtent(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_PADDED_EXTENT") != "1" {
		t.Skip("explicit virtualpad")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("120s")
	}
	if !VulkanInit() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("physical")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	op, e := NewVkAttentionKey32PaddedExtentF32(context.Background())
	if e != nil {
		t.Fatal(e)
	}
	nativeClose(t, op)
	refop, e := NewVkAttentionKey32OutputILPF32(context.Background())
	if e != nil {
		t.Fatal(e)
	}
	nativeClose(t, refop)
	for _, shape := range [][3]int{{1, 1, 1}, {17, 37, 2}, {33, 65, 3}, {16, 255, 2}, {16, 256, 2}, {16, 257, 2}, {1500, 1500, 20}, {1, 4095, 1}, {4096, 1, 1}, {2, 33, 32}} {
		sq, sk, h := shape[0], shape[1], shape[2]
		extent := (sk + 255) / 256 * 256
		t.Run(strings.Join([]string{fmtI(sq), fmtI(sk), fmtI(h)}, "/"), func(t *testing.T) {
			q := nativeData(sq*h*64, 11, .5)
			k := nativeData(sk*h*64, 12, .5)
			v := nativeData(sk*h*64, 13, .5)
			pk := make([]float32, extent*h*64)
			pv := make([]float32, len(pk))
			copy(pk, k)
			copy(pv, v)
			a, e := NewVkTensorArena(context.Background(), 100<<20)
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, a)
			left := nativeGuard(t, a)
			tq := nativeTensor(t, a, q, sq, h*64)
			tk := nativeTensor(t, a, k, sk, h*64)
			tv := nativeTensor(t, a, v, sk, h*64)
			tpK := nativeTensor(t, a, pk, extent, h*64)
			tpV := nativeTensor(t, a, pv, extent, h*64)
			out := nativeTensor(t, a, nil, sq, h*64)
			rout := nativeTensor(t, a, nil, sq, h*64)
			right := nativeGuard(t, a)
			nativeRun(t, func(c context.Context) error { return refop.Forward(c, rout, tq, tpK, tpV, h) })
			want := nativeDownload(t, rout)
			check := func() {
				got := nativeDownload(t, out)
				for i, x := range got {
					if math.Float32bits(x) != math.Float32bits(want[i]) {
						t.Fatal("explicitpadbits", i)
					}
				}
				left()
				right()
			}
			for repeat := 0; repeat < 3; repeat++ {
				nativeRun(t, func(c context.Context) error { return op.Forward(c, out, tq, tk, tv, h) })
				check()
			}
			if sq*extent*h <= 70000 {
				nativeCompare(t, "padded64float64", want, attentionReference(q, pk, pv, sq, extent, h, 64), 2e-5, 2e-5, 0)
			}
			cancelled, cancel := context.WithCancel(context.Background())
			cancel()
			if e := op.Forward(cancelled, out, tq, tk, tv, h); !errors.Is(e, context.Canceled) {
				t.Fatal("cancel", e)
			}
			nativeRun(t, func(c context.Context) error { return op.Forward(c, out, tq, tk, tv, h) })
			check()
			if e := a.Close(); e != nil {
				t.Fatal(e)
			}
		})
	}
	t.Log("VIRTUAL_PAD_EXPLICIT_ZERO_ORACLE10_SHAPES_BITS_CANCEL_REUSE_GUARDS_PASS")
}
func fmtI(v int) string { return fmt.Sprint(v) }
