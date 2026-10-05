//go:build linux

package vulkan

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"testing"
)

const narrowTimeCandidatePin = "f6b52b6b79f36904b4d9400559553dfd4d3d413cc8ef73722bc63caa4903eb50"

func narrowTimeSinCos(x float32) (float32, float32) {
	z := x * x
	fma := func(a, b, c float32) float32 { return float32(math.FMA(float64(a), float64(b), float64(c))) }
	s := float32(1.0 / 6227020800.0)
	for _, c := range []float32{-1.0 / 39916800.0, 1.0 / 362880.0, -1.0 / 5040.0, 1.0 / 120.0, -1.0 / 6.0} {
		s = fma(z, s, c)
	}
	sin := x * fma(z, s, 1)
	c := float32(-1.0 / 87178291200.0)
	for _, v := range []float32{1.0 / 479001600.0, -1.0 / 3628800.0, 1.0 / 40320.0, -1.0 / 720.0, 1.0 / 24.0, -1.0 / 2.0} {
		c = fma(z, c, v)
	}
	return sin, fma(z, c, 1)
}
func TestVulkanNarrowTimeCandidateDomain(t *testing.T) {
	var maxHost, maxF64 float64
	for i := -8192; i <= 8192; i++ {
		x := float32(i) / 8192
		s, c := narrowTimeSinCos(x)
		for _, v := range []struct {
			got float32
			ref float64
		}{{s, math.Sin(float64(x))}, {c, math.Cos(float64(x))}} {
			maxHost = math.Max(maxHost, math.Abs(float64(v.got)-float64(float32(v.ref))))
			maxF64 = math.Max(maxF64, math.Abs(float64(v.got)-v.ref))
		}
	}
	// Fixed local gate; F64 bound includes coefficient, z and final F32 rounding.
	if maxHost > 6e-8 || maxF64 > 8e-8 {
		t.Fatal("narrow domain rounding", maxHost, maxF64)
	}
	if 1.0/1307674368000.0 > 1e-12 || 1.0/20922789888000.0 > 1e-12 {
		t.Fatal("Taylor remainder")
	}
	t.Logf("signed domain16385angles hostmax=%g F64max=%g;domain[-1,1]only", maxHost, maxF64)
}
func TestVulkanNarrowTimeCandidatePinnedInputs(t *testing.T) {
	if os.Getenv("GO_PHERENCE_VULKAN_NARROW_TIME_OFFLINE") != "1" {
		t.Skip("new pinned candidate input/reference check")
	}
	var input struct{ Rows []captureTimeRow }
	if e := capturePinnedJSON(os.Getenv("GO_PHERENCE_VULKAN_CAPTURE_TIMES"), captureTimePin, &input); e != nil {
		t.Fatal(e)
	}
	if len(input.Rows) != 9984 {
		t.Fatal("input count")
	}
	var mv, mt float64
	var over int
	for _, r := range input.Rows {
		if math.Abs(float64(r.Angle)) > 1 {
			t.Fatal("candidate domain")
		}
		s, c := narrowTimeSinCos(r.Angle)
		ct, st := (-s*r.Frequency)*r.TimeTangent, (c*r.Frequency)*r.TimeTangent
		for _, v := range []struct{ a, b float32 }{{s, r.Sin}, {c, r.Cos}} {
			d := math.Abs(float64(v.a) - float64(v.b))
			mv = math.Max(mv, d)
			if d > 6e-8 {
				over++
			}
		}
		mt = math.Max(mt, math.Abs(float64(ct)-float64(r.CosTangent)))
		mt = math.Max(mt, math.Abs(float64(st)-float64(r.SinTangent)))
	}
	if over != 0 || mt > 2e-5 {
		t.Fatal("unchanged local gates", mv, mt, over)
	}
	t.Logf("pinnedcandidate9984inputs valueMax=%g tangentMax=%g valueAbove6e-8=%d", mv, mt, over)
}
func TestVulkanNativeNarrowTimeCandidate(t *testing.T) {
	if os.Getenv("GO_PHERENCE_VULKAN_NARROW_TIME_NATIVE") != "1" {
		t.Skip("explicit experimental narrow time shader only")
	}
	path := os.Getenv("GO_PHERENCE_VULKAN_NARROW_TIME_SPIRV")
	info, e := os.Stat(path)
	if e != nil || info.Size() > 65536 || !info.Mode().IsRegular() {
		t.Fatal("candidate asset bound", e)
	}
	code, e := os.ReadFile(path)
	if e != nil {
		t.Fatal(e)
	}
	sum := sha256.Sum256(code)
	if hex.EncodeToString(sum[:]) != narrowTimeCandidatePin {
		t.Fatal("candidate asset pin")
	}
	c, e := InspectVulkanShader(code)
	if e != nil || c.StorageBindings != 31 || c.PushBytes != 8 || c.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal("candidate interface", c, e)
	}
	for pos := 20; pos < len(code); {
		word := binary.LittleEndian.Uint32(code[pos:])
		n := int(word >> 16)
		if n < 1 || pos+4*n > len(code) {
			t.Fatal("instruction bound")
		}
		if uint16(word) == 12 {
			ext := binary.LittleEndian.Uint32(code[pos+16:])
			if ext == 13 || ext == 14 {
				t.Fatal("builtin trig survived")
			}
		}
		pos += 4 * n
	}
	testVulkanTimeBoundaryCapture(t, code, "ExperimentalFMApolynomialtime/domainabsangle<=1;productionbuiltinunchanged;fixedpinnedinputs/noADupdate/fullstepacceptance.")
}
