package vulkan

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestVulkanNativeQ8OriginalBytes(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_TEST_INTDOT_ORIGINAL_INPUT_DIR")
	if dir == "" {
		t.Skip("independent original GPU Q8_1 bytes")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	if !VulkanInitIntegerDot() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("explicit physical integer-dot device")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	for _, name := range []string{"ff-norm", "tmp-ff"} {
		t.Run(name, func(t *testing.T) {
			raw, e := os.ReadFile(filepath.Join(dir, "intdot-inputs", name+".f32"))
			if e != nil || len(raw)%128 != 0 {
				t.Fatal("F32 input", e)
			}
			inputPin := "e43e03363bafbf2f3cdbca5290e92cd5547406c3a601e37f80d365926a8333db"
			if name == "tmp-ff" {
				inputPin = "07cdaf0e8c75e8fa7c1977c6a0bbf2a753fb7b27f3402ba575eaf9dbe0113f3f"
			}
			if fmt.Sprintf("%x", sha256.Sum256(raw)) != inputPin {
				t.Fatal("layer input pin")
			}
			x := make([]float32, len(raw)/4)
			for i := range x {
				x[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
				if math.IsNaN(float64(x[i])) || math.IsInf(float64(x[i]), 0) {
					t.Fatal("nonfinite")
				}
			}
			blocks := len(x) / 32
			shader := os.Getenv("GO_PHERENCE_TEST_INTDOT_Q8_SPV")
			if shader == "" {
				t.Fatal("explicit quantiser shader")
			}
			code, e := os.ReadFile(shader)
			if e != nil {
				t.Fatal(e)
			}
			kernel, e := VkKernelCreateIntegerDot(code, 2, 4)
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, kernel)
			a, e := NewVkTensorArena(context.Background(), 100<<20)
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, a)
			left := nativeGuard(t, a)
			tx := nativeTensor(t, a, x, len(x))
			tq := nativeTensor(t, a, nil, blocks*9)
			right := nativeGuard(t, a)
			plan, e := NewVkF32Plan(context.Background(), []VkF32Stage{{Kernel: kernel, Groups: [3]uint32{uint32((blocks + 31) / 32), 1, 1}, Tensors: []*VkTensorF32{tx, tq}, PushWords: []uint32{uint32(blocks)}}})
			if e != nil {
				t.Fatal(e)
			}
			nativeClose(t, plan)
			nativeRun(t, plan.Run)
			got := nativeDownload(t, tq)
			for sub := 0; sub < 2; sub++ {
				suffix := "0"
				if sub == 1 {
					suffix = "1"
				}
				oracle, e := os.ReadFile(filepath.Join(dir, name+"-original-sub"+suffix+".q8"))
				if e != nil || len(oracle) != blocks*36 {
					t.Fatal("original bytes", e)
				}
				oraclePin := "750437525e6ea0968a648410f29253ed1ad58db3b3d2f2daa1b6fa4f650c375c"
				if name == "tmp-ff" {
					oraclePin = "6397bc395041d0a3801057873b2c879d44329d26c1de5fee53280cc5cf513cd4"
				}
				if fmt.Sprintf("%x", sha256.Sum256(oracle)) != oraclePin {
					t.Fatal("original quantised pin")
				}
				different, scales, sums := 0, 0, 0
				for b := 0; b < blocks; b++ {
					group, lane := b/4, b%4
					base := group * 144
					ds := binary.LittleEndian.Uint32(oracle[base+lane*4:])
					word := math.Float32bits(got[b*9])
					if uint16(ds) != uint16(word) {
						scales++
					}
					if uint16(ds>>16) != uint16(word>>16) {
						sums++
					}
					for g := 0; g < 8; g++ {
						want := binary.LittleEndian.Uint32(oracle[base+16+(lane*8+g)*4:])
						v := math.Float32bits(got[b*9+1+g])
						if want != v {
							different++
						}
					}
				}
				t.Logf("ORIGINAL_Q8_BYTES fixture=%s subgroup=%d blocks=%d differentQuantWords=%d differentD=%d differentSum=%d", name, sub, blocks, different, scales, sums)
				if different+scales+sums != 0 {
					t.Error("original quantisation divergence")
				}
			}
			left()
			right()
		})
	}
}
