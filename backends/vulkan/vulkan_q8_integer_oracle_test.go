package vulkan

import (
	"context"
	"encoding/binary"
	"github.com/rcarmo/go-pherence/half"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// Scalar arithmetic oracle, independent of shader packing/reductions. GLSL
// Round leaves tie direction implementation-defined. This diagnostic fixes
// nearest-even and deliberately checks ties on the admitted physical device;
// original-engine quantised-block parity remains a separate gate.
func integerQ8Oracle(x []float32) ([9]uint32, [32]int8) {
	var out [9]uint32
	var bytes [32]int8
	var max float32
	for _, v := range x {
		a := float32(math.Abs(float64(v)))
		if a > max {
			max = a
		}
	}
	d := max / 127
	var inv float32
	if d != 0 {
		inv = 1 / d
	}
	sum := 0
	for i, v := range x {
		bytes[i] = int8(math.RoundToEven(float64(v * inv)))
		sum += int(bytes[i])
		out[1+i/4] |= uint32(uint8(bytes[i])) << uint((i%4)*8)
	}
	out[0] = uint32(half.F32ToF16Even(d)) | uint32(half.F32ToF16Even(float32(sum)*d))<<16
	return out, bytes
}
func integerQ5Q8Oracle(raw []byte, q [9]uint32, bytes [32]int8) float32 {
	var dot int32
	// Interpret Q5 directly from the original 22-byte format, not packed GPU words.
	high := binary.LittleEndian.Uint32(raw[2:6])
	for j := 0; j < 32; j++ {
		v := uint32(raw[6+j%16])
		if j >= 16 {
			v >>= 4
		}
		v = (v & 15) | ((high>>uint(j))&1)<<4
		dot += int32(v) * int32(bytes[j])
	}
	dw := half.F16ToF32(binary.LittleEndian.Uint16(raw))
	d := half.F16ToF32(uint16(q[0]))
	s := half.F16ToF32(uint16(q[0] >> 16))
	scaled := float32(dot) * d
	correction := float32(16) * s
	return dw * (scaled - correction)
}
func integerQ8Inputs() []float32 {
	const blocks = 205
	x := make([]float32, blocks*32)
	for b := 0; b < blocks; b++ {
		for j := 0; j < 32; j++ {
			v := float32((b*31+j*17)%257-128) / 16
			switch b {
			case 0:
				v = 0
			case 1:
				v = float32(j%2)*254 - 127
			case 2:
				if j == 0 {
					v = 127
				} else {
					v = float32(j/2) + 0.5
					if j%2 != 0 {
						v = -v
					}
				}
			case 3:
				v *= 1e-12
			case 4:
				v *= 0.0001
			case 5:
				v *= 100
			}
			x[b*32+j] = v
		}
	}
	return x
}
func TestVulkanIntegerQ8ScalarOracle(t *testing.T) {
	x := integerQ8Inputs()
	zero, z := integerQ8Oracle(x[:32])
	if zero != ([9]uint32{}) || z != ([32]int8{}) {
		t.Fatal("zero")
	}
	q, bytes := integerQ8Oracle(x[32:64])
	for j, v := range bytes {
		if v != int8(j%2*254-127) {
			t.Fatal("signed bound")
		}
	}
	if uint16(q[0]) != 0x3c00 || uint16(q[0]>>16) != 0 {
		t.Fatal("scale/sum")
	}
	_, bytes = integerQ8Oracle(x[64:96])
	if bytes[1] != 0 || bytes[2] != 2 || bytes[3] != -2 || bytes[4] != 2 || bytes[5] != -2 {
		t.Fatal("tie fixture", bytes)
	}
	q, _ = integerQ8Oracle(x[96:128])
	if q[0]&0x7fff7fff != 0 {
		t.Fatal("half underflow", q[0])
	}
	// Independent hand-computed unsigned-Q5 dot and zero-point correction.
	raw := make([]byte, 22)
	binary.LittleEndian.PutUint16(raw, 0x3c00)
	for i := 6; i < 22; i++ {
		raw[i] = 0x11
	}
	var manual [32]int8
	for i := range manual {
		manual[i] = 1
	}
	scale := [9]uint32{0x50003c00} // d=1, s=32
	if got := integerQ5Q8Oracle(raw, scale, manual); got != -480 {
		t.Fatal("Q5 offset", got)
	}
}
func TestVulkanNativeIntegerQ8Oracle(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_TEST_INTDOT_QUANT_DIR")
	if dir == "" {
		t.Skip("explicit Q8_1 arithmetic diagnostic")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	if !VulkanInitIntegerDot() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("integer-dot init")
	}
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	kernel := func(name string, n int) *VkComputeKernel {
		code, e := os.ReadFile(dir + "/" + name + "-stripped.spv")
		if e != nil {
			t.Fatal(e)
		}
		if _, e = VkKernelCreate(code, n, 4); e == nil {
			t.Fatal("baseline quant admission")
		}
		k, e := VkKernelCreateIntegerDot(code, n, 4)
		if e != nil {
			t.Fatal(e)
		}
		nativeClose(t, k)
		return k
	}
	qk, dk := kernel("q8", 2), kernel("q5q8", 3)
	x := integerQ8Inputs()
	blocks := len(x) / 32
	raw := make([]byte, blocks*22)
	for b := 0; b < blocks; b++ {
		r := raw[b*22 : (b+1)*22]
		binary.LittleEndian.PutUint16(r, []uint16{0x3c00, 0xbc00, 0x2800, 0x0001}[b%4])
		for j := 2; j < 22; j++ {
			r[j] = byte(b*17 + j*29)
		}
	}
	packed, e := packLinearQ5Blocks(context.Background(), raw)
	if e != nil {
		t.Fatal(e)
	}
	wf := make([]float32, len(packed))
	for i, v := range packed {
		wf[i] = math.Float32frombits(v)
	}
	a := nativeArena(t)
	left := nativeGuard(t, a)
	tx := nativeTensor(t, a, x, blocks, 32)
	tq := nativeTensor(t, a, nil, blocks*9)
	tw := nativeTensor(t, a, wf, len(wf))
	ty := nativeTensor(t, a, nil, blocks)
	right := nativeGuard(t, a)
	plan, e := NewVkF32Plan(context.Background(), []VkF32Stage{{Kernel: qk, Groups: [3]uint32{uint32((blocks + 31) / 32), 1, 1}, Tensors: []*VkTensorF32{tx, tq}, PushWords: []uint32{uint32(blocks)}}, {Kernel: dk, Groups: [3]uint32{uint32((blocks + 31) / 32), 1, 1}, Tensors: []*VkTensorF32{tq, tw, ty}, PushWords: []uint32{uint32(blocks)}}})
	if e != nil {
		t.Fatal(e)
	}
	nativeClose(t, plan)
	for repeat := 0; repeat < 5; repeat++ {
		nativeRun(t, plan.Run)
		qout, yout := nativeDownload(t, tq), nativeDownload(t, ty)
		for b := 0; b < blocks; b++ {
			q, bytes := integerQ8Oracle(x[b*32 : (b+1)*32])
			for j, w := range q {
				if got := math.Float32bits(qout[b*9+j]); got != w {
					t.Fatalf("Q8 block%d word%d got%08x want%08x", b, j, got, w)
				}
			}
			want := integerQ5Q8Oracle(raw[b*22:(b+1)*22], q, bytes)
			if got := yout[b]; math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("Q5Q8 block%d got%g want%g", b, got, want)
			}
		}
		left()
		right()
	}
	t.Logf("Q8_ORACLE blocks=%d words=%d repeat=5 Q5Q8 outputs exact; ties=nearest-even explicit diagnostic", blocks, blocks*9)
}
