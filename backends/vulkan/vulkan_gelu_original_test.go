package vulkan

import (
	"context"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"
)

// Float32 evaluation in the shader's operation order. Exp is rounded from Go
// float64 libm, NOT a model of every Vulkan driver's transcendental.
func geluOriginalTanhModel(v float32) float32 {
	const a, c = float32(0.044715), float32(0.79788456080286535587989211986876)
	val := float32(float32(c*v) * float32(float32(1)+float32(float32(a*v)*v)))
	e := float32(math.Exp(float64(float32(2 * val))))
	return float32(float32(0.5)*v) * float32(float32(2)-float32(float32(2)/float32(e+1)))
}

func TestVulkanOfflineGELUOriginalTanhModel(t *testing.T) {
	var maxAbs float64
	for i := -60000; i <= 60000; i++ {
		v := float32(i) / 6000
		x := float64(v)
		ref := 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
		got := float64(geluOriginalTanhModel(v))
		err := math.Abs(got - ref)
		if math.IsNaN(got) || err > 4e-6+4e-6*math.Abs(ref) {
			t.Fatalf("v%.9g got%.12g ref%.12g", v, got, ref)
		}
		maxAbs = math.Max(maxAbs, err)
	}
	// Saturation: exp overflow to +Inf gives 2-0 (identity); deep negative gives -0/0.
	if geluOriginalTanhModel(100) != 100 || geluOriginalTanhModel(-100) != 0 {
		t.Fatal("saturation", geluOriginalTanhModel(100), geluOriginalTanhModel(-100))
	}
	// Must not be the default erf GELU: the two forms differ at x=2.
	if math.Abs(float64(geluOriginalTanhModel(2)-geluErfModel(2))) < 1e-5 {
		t.Fatal("accidentally testing erf form")
	}
	t.Logf("max_abs=%g", maxAbs)
}

func TestVulkanOfflineGELUOriginalTanhContract(t *testing.T) {
	offlineVK(t)
	c, err := InspectVulkanShader(spirv_gelu_original_tanh_f32)
	if err != nil || c != (VulkanShaderContract{LocalSize: [3]uint32{256, 1, 1}, StorageBindings: 3, PushBytes: 4}) {
		t.Fatal(c, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = NewVkGELUOriginalTanhF32(ctx)
	expectErrorIs(t, err, context.Canceled)
	vkLimits.WorkgroupSize[0] = 255
	_, err = NewVkGELUOriginalTanhF32(context.Background())
	expectErrorIs(t, err, ErrVulkanLimit)
	// The only admitted extended instruction is Exp; Tanh (21) stays rejected.
	exps := 0
	words := spirvTestWords(spirv_gelu_original_tanh_f32)
	for i := 5; i < len(words); {
		n, op := int(words[i]>>16), uint16(words[i])
		if op == 12 {
			if words[i+4] != 27 {
				t.Fatal("unexpected extended instruction", words[i+4])
			}
			exps++
		}
		i += n
	}
	if exps != 1 {
		t.Fatal("exp count", exps)
	}
	bad := editSPIRV(spirv_gelu_original_tanh_f32, 12, func(a []uint32) { a[4] = 21 })
	_, err = InspectVulkanShader(bad)
	expectErrorIs(t, err, ErrVulkanShaderContract)
}

// Native GPU comparison with the float32 model (exp may differ by driver ulps),
// tail sizes, exact in-place alias and bounded runtime. Explicit opt-in.
func TestVulkanNativeGELUOriginalTanh(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_GELU_ORIGINAL") != "1" {
		t.Skip("explicit native GELU comparison")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	ctx, cancel := context.WithDeadline(context.Background(), d.Add(-time.Second))
	defer cancel()
	if !VulkanInit() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("physical device")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	op, err := NewVkGELUOriginalTanhF32(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer nativeClose(t, op)
	rng := rand.New(rand.NewSource(23))
	var maxAbs float64
	exact, total := 0, 0
	for _, n := range []int{1, 255, 256, 257, 1500 * 5120} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * 3)
		}
		if n > 8 {
			copy(x, []float32{0, float32(math.Copysign(0, -1)), 100, -100, 10, -10, 1e-30, -1e-30})
		}
		a, e := NewVkTensorArena(ctx, 8*n+(16<<10))
		if e != nil {
			t.Fatal(e)
		}
		tx := nativeTensor(t, a, x, n)
		out := nativeTensor(t, a, nil, n)
		if e := op.Forward(ctx, out, tx); e != nil {
			t.Fatal(e)
		}
		got := nativeDownload(t, out)
		if e := op.Forward(ctx, tx, tx); e != nil { // exact alias
			t.Fatal(e)
		}
		alias := nativeDownload(t, tx)
		for i, v := range x {
			want := geluOriginalTanhModel(v)
			if math.Float32bits(alias[i]) != math.Float32bits(got[i]) {
				t.Fatal("alias differs", i, alias[i], got[i])
			}
			err := math.Abs(float64(got[i] - want))
			if math.IsNaN(float64(got[i])) || err > 1e-5+2e-5*math.Abs(float64(want)) {
				t.Fatal("value", v, got[i], want)
			}
			maxAbs = math.Max(maxAbs, err)
			if math.Float32bits(got[i]) == math.Float32bits(want) {
				exact++
			}
			total++
		}
		nativeClose(t, a)
	}
	t.Logf("NATIVE_GELU_ORIGINAL device=%q samples=%d bit_equal_model=%d max_abs=%g", VulkanDeviceName(), total, exact, maxAbs)
}
