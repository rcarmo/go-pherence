package vulkan

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"github.com/rcarmo/go-pherence/half"
	legacy "github.com/rcarmo/go-pherence/loader/whisperggml"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// Pinned trained FFN tensors and captured actual layer0 inputs. This asserts a
// separate block arithmetic oracle, never F32 parity or model accuracy.
func TestVulkanNativeQ5IntegerTrained(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_TEST_INTDOT_TRAINED_INPUTS")
	if dir == "" {
		t.Skip("explicit pinned trained integer-dot operator")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	ctx, cancel := context.WithDeadline(context.Background(), d.Add(-time.Second))
	defer cancel()
	pin := "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"
	if os.Getenv("GO_PHERENCE_WHISPER_GGML_SHA256") != pin {
		t.Fatal("model pin")
	}
	file, e := legacy.Open(ctx, os.Getenv("GO_PHERENCE_WHISPER_GGML"), pin)
	if e != nil {
		t.Fatal(e)
	}
	defer file.Close()
	if !VulkanInitIntegerDot() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("physical device")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	for _, part := range []struct {
		name, input string
		k, n        int
	}{{"encoder.blocks.0.mlp.0.weight", "ff-norm", 1280, 5120}, {"encoder.blocks.0.mlp.2.weight", "tmp-ff", 5120, 1280}} {
		t.Run(part.name, func(t *testing.T) {
			raw, shape, e := file.Q5Blocks(ctx, part.name)
			if e != nil || len(shape) != 2 || shape[0] != part.k || shape[1] != part.n {
				t.Fatal("weight", shape, e)
			}
			f32, e := os.ReadFile(dir + "/" + part.input + ".f32")
			if e != nil || len(f32) != 1500*part.k*4 {
				t.Fatal("input", e)
			}
			inputPin := "e43e03363bafbf2f3cdbca5290e92cd5547406c3a601e37f80d365926a8333db"
			if part.input == "tmp-ff" {
				inputPin = "07cdaf0e8c75e8fa7c1977c6a0bbf2a753fb7b27f3402ba575eaf9dbe0113f3f"
			}
			if fmt.Sprintf("%x", sha256.Sum256(f32)) != inputPin {
				t.Fatal("trained input pin")
			}
			x := make([]float32, len(f32)/4)
			for i := range x {
				x[i] = math.Float32frombits(binary.LittleEndian.Uint32(f32[i*4:]))
			}
			op, e := NewVkLinearQ5IntegerDotHybridSetStream(ctx, []VkLinearQ5Shape{{part.n, part.k}}, func(context.Context, int) ([]byte, error) { return raw, nil })
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, op)
			base, e := NewVkLinearQ5GroupedF32(ctx, raw, part.n, part.k)
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, base)
			a, e := NewVkTensorArena(ctx, 150<<20)
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, a)
			left := nativeGuard(t, a)
			tx := nativeTensor(t, a, x, 1500, part.k)
			bias := nativeTensor(t, a, nil, part.n)
			out := nativeTensor(t, a, nil, 1500, part.n)
			baseout := nativeTensor(t, a, nil, 1500, part.n)
			q := nativeTensor(t, a, nil, 1500*(part.k/32)*9)
			right := nativeGuard(t, a)
			stages, e := op.Stages(ctx, 0, out, tx, bias, q)
			if e != nil {
				t.Fatal(e)
			}
			plan, e := NewVkF32Plan(ctx, stages)
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, plan)
			nativeRun(t, plan.Run)
			got := nativeDownload(t, out)
			for _, index := range []int{0, part.n - 1, part.n*3 + 31, (1500*part.n)/2 + part.n/2, 1499 * part.n, 1500*part.n - 1} {
				r, c := index/part.n, index%part.n
				var want float32
				nb := part.k / 32
				for b := 0; b < nb; b++ {
					quant, bytes := integerQ8Oracle(x[(r*nb+b)*32 : (r*nb+b+1)*32])
					p := raw[(c*nb+b)*22 : (c*nb+b+1)*22]
					high := binary.LittleEndian.Uint32(p[2:6])
					var dot int32
					for j, v := range bytes {
						w := uint32(p[6+j%16])
						if j >= 16 {
							w >>= 4
						}
						w = (w & 15) | ((high>>uint(j))&1)<<4
						dot += int32(w) * int32(v)
					}
					dq, sq := half.F16ToF32(uint16(quant[0])), half.F16ToF32(uint16(quant[0]>>16))
					scaled := float32(dot) * dq
					corrected := scaled - 16*sq
					dw := half.F16ToF32(binary.LittleEndian.Uint16(p))
					want = float32(math.FMA(float64(dw), float64(corrected), float64(want)))
				}
				if math.Float32bits(got[index]) != math.Float32bits(want) {
					t.Fatalf("trained scalar index%d got%g want%g", index, got[index], want)
				}
			}
			nativeRun(t, func(c context.Context) error { return base.Forward(c, baseout, tx, bias) })
			ref := nativeDownload(t, baseout)
			f32Stage, e := op.F32Stage(ctx, 0, out, tx, bias)
			if e != nil {
				t.Fatal(e)
			}
			f32Plan, e := NewVkF32Plan(ctx, []VkF32Stage{f32Stage})
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, f32Plan)
			nativeRun(t, f32Plan.Run)
			f32Values := nativeDownload(t, out)
			for i, v := range f32Values {
				if math.Float32bits(v) != math.Float32bits(ref[i]) {
					t.Fatal("hybrid shared F32 stage", i)
				}
			}
			nativeRun(t, plan.Run)
			var maximum, total float64
			for i, v := range got {
				delta := math.Abs(float64(v - ref[i]))
				if delta > maximum {
					maximum = delta
				}
				total += delta
			}
			t.Logf("INTDOT_TRAINED_DRIFT tensor=%s values=%d scalarChecked=6 maxAbs=%.9g meanAbs=%.9g", part.name, len(got), maximum, total/float64(len(got)))
			for repeat := 0; repeat < 5; repeat++ {
				timed := func(fn func(context.Context) error) float64 {
					start := time.Now()
					nativeRun(t, fn)
					return time.Since(start).Seconds()
				}
				var baseline, candidate float64
				if repeat%2 == 0 {
					baseline = timed(func(c context.Context) error { return base.Forward(c, baseout, tx, bias) })
					candidate = timed(plan.Run)
				} else {
					candidate = timed(plan.Run)
					baseline = timed(func(c context.Context) error { return base.Forward(c, baseout, tx, bias) })
				}
				t.Logf("INTDOT_TRAINED_SAMPLE tensor=%s repeat=%d baseline=%.9f candidate=%.9f", part.name, repeat, baseline, candidate)
			}
			reuse := nativeDownload(t, out)
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
