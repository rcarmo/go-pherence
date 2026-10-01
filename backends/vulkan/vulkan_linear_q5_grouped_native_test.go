package vulkan

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	gguf "github.com/rcarmo/go-pherence/loader/gguf"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// Explicit native qualification of the standalone packed-Q5 operator.
// No model integration or production/default selection is performed.
func TestVulkanNativeQ5GroupedOperator(t *testing.T) {
	admit := os.Getenv("GO_PHERENCE_TEST_Q5_GROUPED_OPERATOR")
	if admit != "1" {
		t.Skip("explicit candidate window")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	var err error
	if !VulkanInit() {
		t.Fatal("Vulkan init")
	}
	want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if want == "" || !strings.Contains(VulkanDeviceName(), want) {
		t.Fatal("physical GPU gate")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	base, err := NewVkLinearRegTile64F32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, base)
	for _, dims := range [][3]int{{32, 32, 64}, {1, 32, 1}, {2, 64, 5}, {33, 96, 63}, {1500, 1280, 1280}, {1500, 1280, 5120}, {1500, 5120, 1280}} {
		if !t.Run(stringShape(dims), func(t *testing.T) {
			m, k, n := dims[0], dims[1], dims[2]
			ctx := context.Background()
			x, b := nativeData(m*k, 11, .125), nativeData(n, 13, .125)
			raw := make([]byte, n*k/32*22)
			for block := 0; block < n*k/32; block++ {
				binary.LittleEndian.PutUint16(raw[block*22:], 0x2400)
				for j := 2; j < 22; j++ {
					raw[block*22+j] = byte((block*37 + j*19) % 256)
				}
			}

			if m == 32 && k == 32 && n == 64 {
				fixture := os.Getenv("GO_PHERENCE_TEST_Q5_GROUPED_FIXTURE")
				raw, err = os.ReadFile(fixture + "/q5-reference-input.bin")
				if err != nil || len(raw) != 64*22 {
					t.Fatal("oracle raw", err)
				}
				h := sha256.Sum256(raw)
				if hex.EncodeToString(h[:]) != "2f540db499b6e8c083b0d8fc92d0b1f9f249d322c52012642264dc05a3589c55" {
					t.Fatal("oracle raw pin")
				}
				for i := range x {
					x[i] = 0
				}
				for i := 0; i < 32; i++ {
					x[i*32+i] = 1
				}
				for i := range b {
					b[i] = 0
				}
			}
			if path := os.Getenv("GO_PHERENCE_TEST_Q5_GROUPED_TENSOR"); path != "" && m == 1500 && fmt.Sprint(k) == os.Getenv("GO_PHERENCE_TEST_Q5_GROUPED_TENSOR_K") && fmt.Sprint(n) == os.Getenv("GO_PHERENCE_TEST_Q5_GROUPED_TENSOR_N") {
				raw, err = os.ReadFile(path)
				if err != nil || len(raw) != n*k/32*22 {
					t.Fatal("model tensor", err)
				}
				h := sha256.Sum256(raw)
				if hex.EncodeToString(h[:]) != os.Getenv("GO_PHERENCE_TEST_Q5_GROUPED_TENSOR_SHA256") {
					t.Fatal("model tensor pin")
				}
			}
			rounded, err := gguf.DequantToF32(raw, gguf.QuantQ5_0, n*k)
			if err != nil {
				t.Fatal(err)
			}

			if m == 32 && k == 32 && n == 64 {
				oracle, e := os.ReadFile(os.Getenv("GO_PHERENCE_TEST_Q5_GROUPED_FIXTURE") + "/q5-reference-output.f32")
				if e != nil || len(oracle) != len(rounded)*4 {
					t.Fatal("oracle result", e)
				}
				h := sha256.Sum256(oracle)
				if hex.EncodeToString(h[:]) != "d44a59144d4ea8153421604d78e1d87d07fd72a50d4586e8b2587988ede1ee15" {
					t.Fatal("oracle result pin")
				}
				for i, v := range rounded {
					if math.Float32bits(v) != binary.LittleEndian.Uint32(oracle[i*4:]) {
						t.Fatal("independent decode", i)
					}
				}
			}
			a, err := NewVkTensorArena(ctx, 96<<20)
			if err != nil {
				t.Fatal(err)
			}
			nativeClose(t, a)
			left := nativeGuard(t, a)
			tx, tw, tb := nativeTensor(t, a, x, m, k), nativeTensor(t, a, rounded, n, k), nativeTensor(t, a, b, n)
			out := nativeTensor(t, a, nil, m, n)
			right := nativeGuard(t, a)
			op, err := NewVkLinearQ5GroupedF32(ctx, raw, n, k)
			if err != nil {
				if op != nil {
					op.Close()
				}
				t.Fatal(err)
			}
			nativeClose(t, op)
			runPacked := func() time.Duration {
				return nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, out, tx, tb) })
			}
			runBase := func() time.Duration {
				return nativeRun(t, func(ctx context.Context) error { return base.Forward(ctx, out, tx, tw, tb) })
			}
			runBase()
			ref := nativeDownload(t, out)
			runPacked()
			got := nativeDownload(t, out)
			for i, v := range got {
				if math.Float32bits(v) != math.Float32bits(ref[i]) {
					t.Fatal("packed drift", dims, i, v, ref[i])
				}
			}
			left()
			right()
			if m < 1000 {
				nativeCompare(t, "Q5tile64", got, nativeLinearRef(x, rounded, b, m, k, n), 2e-5, 2e-5, 0)
				return
			}
			cancelCtx, stop := context.WithCancel(ctx)
			stop()
			if err := op.Forward(cancelCtx, out, tx, tb); !errors.Is(err, context.Canceled) {
				t.Fatal("precancel", err)
			}
			runPacked()
			for i, v := range nativeDownload(t, out) {
				if math.Float32bits(v) != math.Float32bits(ref[i]) {
					t.Fatal("cancel/reuse", i)
				}
			}
			for block := 0; block < 3; block++ {
				order := []bool{false, true, true, false}
				if block%2 == 1 {
					order = []bool{true, false, false, true}
				}
				for _, candidate := range order {
					var elapsed time.Duration
					if candidate {
						elapsed = runPacked()
					} else {
						elapsed = runBase()
					}
					values := nativeDownload(t, out)
					for i, v := range values {
						if math.Float32bits(v) != math.Float32bits(ref[i]) {
							t.Fatal("timed drift", dims, i)
						}
					}
					left()
					right()
					t.Logf("Q5GROUPED_SAMPLE candidate=%t block=%d ns=%d shape=%v", candidate, block, elapsed.Nanoseconds(), dims)
				}
			}
		}) {
			return
		}
	}
}
func stringShape(d [3]int) string { return fmt.Sprint(d) }
