package vulkan

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

func TestVulkanNativeAttentionUnrollIndependent(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_ATTENTION_UNROLL_INDEPENDENT") != "1" {
		t.Skip("explicit fixed64 independent reference")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	if !VulkanInit() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("physicaldevice")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	op, e := NewVkAttentionKey32ScoreILPUnroll4F32(context.Background())
	if e != nil {
		t.Fatal(e)
	}
	nativeClose(t, op)
	for _, shape := range [][3]int{{1, 1, 1}, {17, 37, 2}, {33, 65, 3}, {1, 4096, 1}, {4096, 1, 1}, {2, 33, 32}} {
		for _, sign := range []float32{-1, 0, 1} {
			sq, sk, h := shape[0], shape[1], shape[2]
			q, k, v := nativeData(sq*h*64, 11, sign), nativeData(sk*h*64, 12, .5), nativeData(sk*h*64, 13, .5)
			if sign == 0 {
				for i := range k {
					k[i] = 1000
				}
			}
			a := nativeArena(t)
			left := nativeGuard(t, a)
			tq, tk, tv := nativeTensor(t, a, q, sq, h*64), nativeTensor(t, a, k, sk, h*64), nativeTensor(t, a, v, sk, h*64)
			out := nativeTensor(t, a, nil, sq, h*64)
			right := nativeGuard(t, a)
			nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, out, tq, tk, tv, h) })
			nativeCompare(t, "unroll64-independent", nativeDownload(t, out), attentionReference(q, k, v, sq, sk, h, 64), 2e-5, 2e-5, 0)
			left()
			right()
			if e = a.Close(); e != nil {
				t.Fatal(e)
			}
		}
	}
	t.Log("UNROLL_INDEPENDENT shapes6 signs3 float64Reference PASS")
}
