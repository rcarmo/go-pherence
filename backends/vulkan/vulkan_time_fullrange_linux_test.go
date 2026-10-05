//go:build linux

package vulkan

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"testing"
	"time"
)

const fullrangeTimePin = "704c7ae3ab19c77a25c447ec7a7f356efa86ecb03dc883b41500eaf16167df17"

func fullrangeTimeSinCos(angle float32) (float32, float32) {
	fma := func(a, b, c float32) float32 { return float32(math.FMA(float64(a), float64(b), float64(c))) }
	x := float32(math.Abs(float64(angle)))
	q := uint32(fma(x, float32(.6366197723675814), .5))
	r := angle
	if x > float32(.7853981633974483) {
		n := float32(q)
		r = fma(-n, float32(1.570796251296997), x)
		r = fma(-n, float32(7.549789415861596e-8), r)
		r = fma(-n, float32(5.390302529957765e-15), r)
	} else {
		q = 0
	}
	z := r * r
	s := float32(1.0 / 6227020800.0)
	for _, v := range []float32{-1.0 / 39916800.0, 1.0 / 362880.0, -1.0 / 5040.0, 1.0 / 120.0, -1.0 / 6.0} {
		s = fma(z, s, v)
	}
	s = r * fma(z, s, 1)
	c := float32(-1.0 / 87178291200.0)
	for _, v := range []float32{1.0 / 479001600.0, -1.0 / 3628800.0, 1.0 / 40320.0, -1.0 / 720.0, 1.0 / 24.0, -1.0 / 2.0} {
		c = fma(z, c, v)
	}
	c = fma(z, c, 1)
	switch q & 3 {
	case 1:
		s, c = c, -s
	case 2:
		s, c = -s, -c
	case 3:
		s, c = -c, s
	}
	if x > float32(.7853981633974483) && angle < 0 {
		s = -s
	}
	return s, c
}
func fullrangeAngles() []float32 {
	a := []float32{0, math.Float32frombits(1 << 31), math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, 8192, -8192}
	for i := -262144; i <= 262144; i++ {
		a = append(a, float32(i)/32)
	}
	for q := -5215; q <= 5215; q++ {
		x := float32(float64(q) * math.Pi / 2)
		if math.Abs(float64(x)) <= 8192 {
			a = append(a, x, math.Nextafter32(x, float32(math.Inf(-1))), math.Nextafter32(x, float32(math.Inf(1))))
		}
	}
	return a
}
func TestVulkanFullrangeTimeReference(t *testing.T) {
	var maxHost, maxF64 float64
	angles := fullrangeAngles()
	for _, x := range angles {
		s, c := fullrangeTimeSinCos(x)
		for _, v := range []struct {
			got  float32
			want float64
		}{{s, math.Sin(float64(x))}, {c, math.Cos(float64(x))}} {
			maxHost = math.Max(maxHost, math.Abs(float64(v.got)-float64(float32(v.want))))
			maxF64 = math.Max(maxF64, math.Abs(float64(v.got)-v.want))
		}
	}
	if maxHost > 6e-8 || maxF64 > 8e-8 {
		t.Fatal("fullrange existing value gate/rounding", maxHost, maxF64)
	}
	s, c := fullrangeTimeSinCos(math.Float32frombits(1 << 31))
	if math.Float32bits(s) != 1<<31 || c != 1 {
		t.Fatal("signedzero")
	}
	t.Logf("fullrange signedangles=%d hostmax=%g F64max=%g", len(angles), maxHost, maxF64)
}
func fullrangeTimeCode(t *testing.T) []byte {
	t.Helper()
	code := spirvTrainingTimeEmbedding
	if path := os.Getenv("GO_PHERENCE_VULKAN_FULLRANGE_TIME_SPIRV"); path != "" {
		var e error
		code, e = os.ReadFile(path)
		if e != nil {
			t.Fatal(e)
		}
		s := sha256.Sum256(code)
		if hex.EncodeToString(s[:]) != fullrangeTimePin {
			t.Fatal("fullrangeassetpin")
		}
	}
	c, e := InspectVulkanShader(code)
	if e != nil || c.StorageBindings != 31 || c.PushBytes != 8 || c.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal("fullrangeinterface", c, e)
	}
	if e = checkFullrangeTrigInstructions(code); e != nil {
		t.Fatal(e)
	}
	return code
}

// Asset integrity alone does not establish that main calls the replacement.
// Inspect the complete executable to reject the original built-in arithmetic.
func checkFullrangeTrigInstructions(code []byte) error {
	fmas := 0
	for pos := 20; pos < len(code); {
		word := binary.LittleEndian.Uint32(code[pos:])
		n := int(word >> 16)
		if n < 1 || pos+4*n > len(code) {
			return fmt.Errorf("SPIR-V instruction bounds")
		}
		if uint16(word) == 12 {
			if n < 5 {
				return fmt.Errorf("SPIR-V extended instruction bounds")
			}
			ext := binary.LittleEndian.Uint32(code[pos+16:])
			if ext == 13 || ext == 14 {
				return fmt.Errorf("built-in trig survived replacement")
			}
			if ext == 50 {
				fmas++
			}
		}
		pos += 4 * n
	}
	if fmas == 0 {
		return fmt.Errorf("missing full-range FMA arithmetic: %d", fmas)
	}
	return nil
}
func TestVulkanFullrangeTimeRejectsBuiltin(t *testing.T) {
	for _, ext := range []uint32{13, 14} {
		code := make([]byte, 40)
		binary.LittleEndian.PutUint32(code[20:], 5<<16|12)
		binary.LittleEndian.PutUint32(code[36:], ext)
		if e := checkFullrangeTrigInstructions(code); e == nil {
			t.Fatal("built-in accepted", ext)
		}
	}
}
func TestVulkanFullrangeTimeInterface(t *testing.T) {
	_ = fullrangeTimeCode(t)
}
func TestVulkanNativeFullrangeTimeReleased(t *testing.T) {
	if os.Getenv("GO_PHERENCE_VULKAN_FULLRANGE_TIME_NATIVE") != "1" {
		t.Skip("fullrangecandidatepinnedreleasedcapture")
	}
	code := fullrangeTimeCode(t)
	testVulkanTimeBoundaryCapture(t, code, "FullrangeFMAtrigimplementationcandidate/releasedinputs/productiondomain8192;noADupdate/fullstepacceptance")
}
func TestVulkanNativeFullrangeTimeModes(t *testing.T) {
	if os.Getenv("GO_PHERENCE_VULKAN_FULLRANGE_MODES_NATIVE") != "1" {
		t.Skip("newfullrangeallmodesnativevalidation")
	}
	code := fullrangeTimeCode(t)
	captureInit(t)
	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
	defer cancel()
	if e := VulkanSetMemoryBudget(VulkanMemoryBudget{MaxBytes: 8 << 20, MaxAllocations: 64}); e != nil {
		t.Fatal(e)
	}
	k, e := VkKernelCreate(code, 5, 8)
	if e != nil {
		t.Fatal(e)
	}
	op := &VkTrainingF32{timeEmbedding: k}
	var arenas []*VkTensorArena
	newTensor := func(x []float32) *VkTensorF32 {
		a, e := NewVkTensorArena(ctx, len(x)*4)
		if e != nil {
			t.Fatal(e)
		}
		arenas = append(arenas, a)
		v, e := a.AllocF32(ctx, len(x))
		if e != nil {
			t.Fatal(e)
		}
		if e = v.Upload(ctx, x); e != nil {
			t.Fatal(e)
		}
		return v
	}
	// time8192/frequencies<=1 tests the full admitted angle range with unchanged
	// derivative gate; separate unit-time/high-frequency case uses its F32
	// multiplication/reduction error bound rather than a released-only constant.
	var valueMax, tangentMax, adjointMax float64
	count := 0
	for _, timeValue := range []float32{8192, 1} {
		for _, sign := range []float32{1, -1} {
			const n = 4096
			freq := make([]float32, n)
			gv, gt := make([]float32, 2*n), make([]float32, 2*n)
			for i := range freq {
				angle := sign * float32(i) * float32(8192.0/4095)
				freq[i] = angle / timeValue
				for q := 0; q < 2; q++ {
					gv[n*q+i] = float32(i%7-3) * 1e-4
					gt[n*q+i] = float32(i%5-2) * 1e-8
				}
			}
			f, tm, a, b := newTensor(freq), newTensor([]float32{timeValue, 1}), newTensor(gv), newTensor(gt)
			var expectedAdj float32
			for i, fr := range freq {
				angle := timeValue * fr
				s, c := float32(math.Sin(float64(angle))), float32(math.Cos(float64(angle)))
				expectedAdj += gv[i]*(-s*fr) + gv[n+i]*(c*fr)
				expectedAdj += gt[i]*((-c*fr)*fr) + gt[n+i]*((-s*fr)*fr)
			}
			for mode := uint32(0); mode < 3; mode++ {
				size := 2 * n
				if mode == 2 {
					size = 1
				}
				out := newTensor(make([]float32, size))
				stage, e := op.TimeEmbeddingStage(ctx, out, f, tm, a, b, mode)
				if e != nil {
					t.Fatal(e)
				}
				p, e := NewVkF32Plan(ctx, []VkF32Stage{stage})
				if e != nil {
					t.Fatal(e)
				}
				if e = p.Run(ctx); e != nil {
					t.Fatal("nativefailure no retry", e)
				}
				values := make([]float32, size)
				if e = out.Download(ctx, values); e != nil {
					t.Fatal(e)
				}
				if e = p.Close(); e != nil {
					t.Fatal(e)
				}
				if mode == 2 {
					d := math.Abs(float64(values[0]) - float64(expectedAdj))
					adjointMax = math.Max(adjointMax, d)
					bound := float64(2e-7) * 8192 * float64(n) * (3e-4 + 8192*2e-8)
					if d > bound {
						t.Fatal("adjoint independent rounding bound", d, bound)
					}
					continue
				}
				for i := 0; i < n; i++ {
					fr := freq[i]
					angle := timeValue * fr
					s, c := float32(math.Sin(float64(angle))), float32(math.Cos(float64(angle)))
					for q, w := range []float32{c, s} {
						if mode == 1 {
							if q == 0 {
								w = -s * fr
							} else {
								w = c * fr
							}
						}
						d := math.Abs(float64(values[n*q+i]) - float64(w))
						if mode == 0 {
							valueMax = math.Max(valueMax, d)
							count++
							if d > 6e-8 {
								t.Fatal("value gate", angle, d)
							}
						} else {
							tangentMax = math.Max(tangentMax, d)
							bound := 2e-5
							if timeValue == 1 {
								bound = math.Max(bound, 2e-7*math.Abs(float64(fr)))
							}
							if d > bound {
								t.Fatal("tangent rounding gate", d, bound)
							}
						}
					}
				}
			}
			for i := len(arenas) - 1; i >= 0; i-- {
				if e = arenas[i].Close(); e != nil {
					t.Fatal(e)
				}
			}
			arenas = nil
		}
	}
	if e = k.Close(); e != nil {
		t.Fatal(e)
	}
	u := VulkanMemoryStats()
	if u.Bytes != 0 || u.Allocations != 0 || u.Uncertain || u.InFlight || u.DeviceLost {
		t.Fatal("closure", u)
	}
	report := struct {
		Scope                            string
		ValueCount                       int
		ValueMax, TangentMax, AdjointMax float64
		Final                            VulkanMemoryUsage
	}{"Fullsignedrange8192/allmodes;highfreqderivativeformat/reductionroundingbound,productiongatesunchanged", count, valueMax, tangentMax, adjointMax, u}
	if e = captureWrite(os.Getenv("GO_PHERENCE_VULKAN_CAPTURE_OUTPUT"), report); e != nil {
		t.Fatal(e)
	}
	t.Logf("fullrangevalues=%d max=%g derivativeMax=%g adjointMax=%g", count, valueMax, tangentMax, adjointMax)
}
