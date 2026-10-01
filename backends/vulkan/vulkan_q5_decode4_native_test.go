package vulkan

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	legacy "github.com/rcarmo/go-pherence/loader/whisperggml"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

func TestVulkanNativeQ5Decode4Trained(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_TEST_Q5_DECODE4_INPUTS")
	if dir == "" {
		t.Skip("explicit retained Q5decode4 trained qualification")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	ctx, cancel := context.WithDeadline(context.Background(), deadline.Add(-time.Second))
	defer cancel()
	pin := "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"
	if os.Getenv("GO_PHERENCE_WHISPER_GGML_SHA256") != pin {
		t.Fatal("modelpin")
	}
	file, e := legacy.Open(ctx, os.Getenv("GO_PHERENCE_WHISPER_GGML"), pin)
	if e != nil {
		t.Fatal(e)
	}
	defer file.Close()
	if !VulkanInit() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("physical")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	kernel, e := VkKernelCreate(spirv_linear_q5_decode4_f32, 4, 12)
	if e != nil {
		t.Fatal(e)
	}
	nativeClose(t, kernel)
	for _, part := range []struct {
		name, input, pin string
		k, n             int
	}{{"encoder.blocks.0.mlp.0.weight", "ff-norm", "e43e03363bafbf2f3cdbca5290e92cd5547406c3a601e37f80d365926a8333db", 1280, 5120}, {"encoder.blocks.0.mlp.2.weight", "tmp-ff", "07cdaf0e8c75e8fa7c1977c6a0bbf2a753fb7b27f3402ba575eaf9dbe0113f3f", 5120, 1280}} {
		t.Run(part.name, func(t *testing.T) {
			raw, shape, e := file.Q5Blocks(ctx, part.name)
			if e != nil || len(shape) != 2 || shape[0] != part.k || shape[1] != part.n {
				t.Fatal("tensor", e)
			}
			b, e := os.ReadFile(dir + "/" + part.input + ".f32")
			if e != nil || len(b) != 1500*part.k*4 || fmt.Sprintf("%x", sha256.Sum256(b)) != part.pin {
				t.Fatal("inputpin", e)
			}
			x := make([]float32, len(b)/4)
			for i := range x {
				x[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
			}
			baseline, e := NewVkLinearQ5GroupedF32(ctx, raw, part.n, part.k)
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, baseline)
			packed, e := packLinearQ5Blocks(ctx, raw)
			if e != nil {
				t.Fatal(e)
			}
			wf := make([]float32, len(packed))
			for i, v := range packed {
				wf[i] = math.Float32frombits(v)
			}
			a, e := NewVkTensorArena(ctx, 150<<20)
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, a)
			left := nativeGuard(t, a)
			tx := nativeTensor(t, a, x, 1500, part.k)
			tw := nativeTensor(t, a, wf, len(wf))
			bias := nativeTensor(t, a, nil, part.n)
			out := nativeTensor(t, a, nil, 1500, part.n)
			right := nativeGuard(t, a)
			plan, e := NewVkF32Plan(ctx, []VkF32Stage{{Kernel: kernel, Groups: [3]uint32{uint32((part.n + 63) / 64), uint32((1500 + 63) / 64), 1}, Tensors: []*VkTensorF32{tx, tw, bias, out}, PushWords: []uint32{1500, uint32(part.k), uint32(part.n)}}})
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, plan)
			baseRun := func(c context.Context) error { return baseline.Forward(c, out, tx, bias) }
			nativeRun(t, baseRun)
			ref := nativeDownload(t, out)
			for repeat := 0; repeat < 5; repeat++ {
				for _, candidate := range []bool{repeat%2 == 1, repeat%2 == 0} {
					fn := baseRun
					if candidate {
						fn = plan.Run
					}
					wall := nativeRun(t, fn)
					got := nativeDownload(t, out)
					for i, v := range got {
						if math.Float32bits(v) != math.Float32bits(ref[i]) {
							t.Fatal("trainedbits", i)
						}
					}
					left()
					right()
					t.Logf("DECODE4_TRAINED tensor=%s candidate=%t repeat=%d seconds=%.9f values=%d", part.name, candidate, repeat, wall.Seconds(), len(got))
				}
			}
		})
	}
}
