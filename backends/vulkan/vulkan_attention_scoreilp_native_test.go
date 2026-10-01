package vulkan

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// Explicit native qualification of the checked-in key32 candidate.
func TestVulkanNativeAttentionScoreILP(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_ATTENTION_SCOREILP") != "1" {
		t.Skip("explicit bounded native attention experiment")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	if !VulkanInit() {
		t.Fatal("Vulkan init")
	}
	if want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE"); want == "" || !strings.Contains(VulkanDeviceName(), want) {
		t.Fatal("physical device gate")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	baseline, err := NewVkAttentionKey32F32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, baseline)
	if !t.Run("independent", func(t *testing.T) { nativeSpeechAttentionWith(t, NewVkAttentionKey32ScoreILPF32) }) {
		return
	}
	candidate, err := NewVkAttentionKey32ScoreILPF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, candidate)
	ctx, cancel := context.WithDeadline(context.Background(), d.Add(-time.Second))
	defer cancel()
	a, err := NewVkTensorArena(ctx, 40<<20)
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, a)
	const rows, heads, dim = 1500, 20, 64
	q, k, v := nativeData(rows*heads*dim, 11, .5), nativeData(rows*heads*dim, 12, .5), nativeData(rows*heads*dim, 13, .5)
	left := nativeGuard(t, a)
	tq, tk, tv := nativeTensor(t, a, q, rows, heads*dim), nativeTensor(t, a, k, rows, heads*dim), nativeTensor(t, a, v, rows, heads*dim)
	out := nativeTensor(t, a, nil, rows, heads*dim)
	right := nativeGuard(t, a)
	run := func(op *VkAttentionF32) time.Duration {
		return nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, out, tq, tk, tv, heads) })
	}
	run(baseline)
	ref := nativeDownload(t, out)
	for block := 0; block < 3; block++ {
		order := []bool{false, true, true, false}
		if block%2 == 1 {
			order = []bool{true, false, false, true}
		}
		for _, useCandidate := range order {
			op := baseline
			if useCandidate {
				op = candidate
			}
			wall := run(op)
			got := nativeDownload(t, out)
			max := 0.
			for i, x := range got {
				diff := math.Abs(float64(x) - float64(ref[i]))
				max = math.Max(max, diff)
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.Float32bits(x) != math.Float32bits(ref[i]) {
					t.Fatal("fullshape drift", i, x, ref[i], diff)
				}
			}
			left()
			right()
			t.Logf("ATTENTION_SAMPLE candidate=%t block=%d ns=%d max_abs=%g shape=%s", useCandidate, block, wall.Nanoseconds(), max, fmt.Sprint([3]int{rows, heads, dim}))
		}
	}
}
