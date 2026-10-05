//go:build linux

package pockettts

import (
	"context"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

// Keep the narrow diagnostic call explicit: statistics destination first,
// immutable gradient source second. No device work occurs in the wiring test.
func capturedNormStats(ctx context.Context, d interface {
	NormStats(context.Context, train.ResidentBuffer, train.ResidentBuffer, int, bool) error
}, statistics, gradient train.ResidentBuffer, elements int) error {
	return d.NormStats(ctx, statistics, gradient, elements, false)
}

type capturedNormTestBuffer struct{ id int }

func (*capturedNormTestBuffer) Upload(context.Context, []float32) error   { return nil }
func (*capturedNormTestBuffer) Download(context.Context, []float32) error { return nil }
func (*capturedNormTestBuffer) Close() error                              { return nil }

type capturedNormTestDriver struct {
	dst, src    train.ResidentBuffer
	n           int
	calls       int
	nonnegative bool
}

func (d *capturedNormTestDriver) NormStats(_ context.Context, dst, src train.ResidentBuffer, n int, nonnegative bool) error {
	d.dst, d.src, d.n, d.nonnegative = dst, src, n, nonnegative
	d.calls++
	return nil
}
func TestPocketCapturedNormStatsWiring(t *testing.T) {
	statistics, gradient := &capturedNormTestBuffer{id: 1}, &capturedNormTestBuffer{id: 2}
	d := &capturedNormTestDriver{}
	if err := capturedNormStats(context.Background(), d, statistics, gradient, 513); err != nil {
		t.Fatal(err)
	}
	if d.dst != statistics || d.src != gradient || d.n != 513 || d.nonnegative || d.calls != 1 {
		t.Fatal("norm destination/source wiring", d)
	}
}

func TestPocketNativeCompleteGradientNormCapture(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_PARAMETER_NORM_NATIVE") != "1" {
		t.Skip("explicit full-gradient NormStats/merge/cliponly")
	}
	root := os.Getenv("GO_PHERENCE_POCKET_PARAMETER_OUTPUT")
	var input struct {
		Elements   int64
		NativeNorm float64
		Entries    []struct {
			Name       string `json:"name"`
			Elements   int    `json:"elements"`
			Offset     int64  `json:"offset"`
			NativeFile string `json:"nativeFile"`
		}
	}
	pin := os.Getenv("GO_PHERENCE_POCKET_PARAMETER_INVENTORY_SHA256")
	if pin == "" {
		pin = "5b99d89dd42a50e5332cd9933a496f463b5c90aecac805f795833f44802bbf5d"
	}
	if e := flowCapturePinned(filepath.Join(root, "full-gradient-inventory-byteorder.json"), pin, 1<<20, &input); e != nil {
		t.Fatal(e)
	}
	if input.Elements != 89449730 || len(input.Entries) != 131 {
		t.Fatal("complete norm inventory")
	}
	var count int64
	for i, e := range input.Entries {
		if e.Elements < 1 || e.Elements > 4194304 || e.Offset < 0 || filepath.Base(e.NativeFile) != e.NativeFile || i > 0 && e.Name <= input.Entries[i-1].Name {
			t.Fatal("norm input")
		}
		count += int64(e.Elements)
	}
	if count != input.Elements {
		t.Fatal("norm total")
	}
	if os.Getenv("GO_PHERENCE_DISABLE_VULKAN") != "" || os.Getenv("VK_DRIVER_FILES") != "/usr/share/vulkan/icd.d/nvidia_icd.json" {
		t.Fatal("explicit Vulkan")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	if !vulkan.VulkanInit() || vulkan.VulkanDeviceName() != "NVIDIA GeForce RTX 3060" {
		t.Fatal("runtime mismatch no fallback")
	}
	if e := vulkan.VulkanSetMemoryBudget(vulkan.VulkanMemoryBudget{MaxBytes: 17 << 20, MaxAllocations: 8}); e != nil {
		t.Fatal(e)
	}
	op, e := vulkan.NewVkTrainingF32(ctx)
	if e != nil {
		t.Fatal(e)
	}
	d := &vulkan.VulkanResidentDriver{Operator: op}
	a, e := d.NewBuffer(ctx, 4)
	if e != nil {
		t.Fatal(e)
	}
	b, e := d.NewBuffer(ctx, 4)
	if e != nil {
		t.Fatal(e)
	}
	if e = a.Upload(ctx, make([]float32, 4)); e != nil {
		t.Fatal(e)
	}
	var maximumRSS uint64
	maxN := 0
	for _, entry := range input.Entries {
		maxN = max(maxN, entry.Elements)
	}
	rawStorage := make([]byte, 4*maxN)
	valueStorage := make([]float32, maxN)
	for _, mode := range []uint32{0, 2} {
		previous := int64(0)
		for _, entry := range input.Entries {
			f, e := os.Open(filepath.Join(root, entry.NativeFile))
			if e != nil {
				t.Fatal(e)
			}
			if _, e = f.Seek(entry.Offset, io.SeekStart); e != nil {
				t.Fatal(e)
			}
			raw := rawStorage[:4*entry.Elements]
			if _, e = io.ReadFull(f, raw); e != nil {
				t.Fatal(e)
			}
			if e = f.Close(); e != nil {
				t.Fatal(e)
			}
			values := valueStorage[:entry.Elements]
			for i := range values {
				values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
				if !isFinite(values[i]) {
					t.Fatal("nonfinite")
				}
			}
			n, _ := train.NormStatsShape(entry.Elements)
			g, e := d.NewBuffer(ctx, entry.Elements)
			if e != nil {
				t.Fatal(e)
			}
			stats, e := d.NewBuffer(ctx, n)
			if e != nil {
				t.Fatal(e)
			}
			if e = g.Upload(ctx, values); e != nil {
				t.Fatal(e)
			}
			if e = capturedNormStats(ctx, d, stats, g, entry.Elements); e != nil {
				t.Fatal("native stats failure no retry", e)
			}
			if e = d.NormMerge(ctx, b, a, stats, entry.Elements, previous, mode); e != nil {
				t.Fatal("native merge failure no retry", e)
			}
			a, b = b, a
			previous += int64(entry.Elements)
			if e = stats.Close(); e != nil {
				t.Fatal(e)
			}
			if e = g.Close(); e != nil {
				t.Fatal(e)
			}
			rss, e := flowCaptureRSS()
			if e != nil || rss > 256<<20 {
				t.Fatal("norm RSS cap", rss, e)
			}
			maximumRSS = max(maximumRSS, rss)
		}
	}
	clip, e := d.NewBuffer(ctx, 4)
	if e != nil {
		t.Fatal(e)
	}
	if e = d.NormClip(ctx, clip, a, input.Elements, 1); e != nil {
		t.Fatal("clipfailure no retry", e)
	}
	scalars := make([]float32, 4)
	if e = clip.Download(ctx, scalars); e != nil {
		t.Fatal(e)
	}
	norm, scale, e := train.DecodeGlobalNorm(scalars, input.Elements)
	if e != nil {
		t.Fatal(e)
	}
	if e = clip.Close(); e != nil {
		t.Fatal(e)
	}
	if e = b.Close(); e != nil {
		t.Fatal(e)
	}
	if e = a.Close(); e != nil {
		t.Fatal(e)
	}
	if e = op.Close(); e != nil {
		t.Fatal(e)
	}
	final := vulkan.VulkanMemoryStats()
	if final.Bytes != 0 || final.Allocations != 0 || final.InFlight || final.Uncertain || final.DeviceLost {
		t.Fatal("closure", final)
	}
	report := struct {
		Scope, Device                     string
		Scalars                           []float32
		Norm, ReferenceNorm, NormAbsError float64
		ClipScale, F64ClipScale           float32
		Elements                          int64
		MaxRSS                            uint64
		Final                             vulkan.VulkanMemoryUsage
	}{Scope: "FullsavedgradientnativeNormStats/merge/cliponly;nooptimizer/update;F64normindependentreference.", Device: vulkan.VulkanDeviceName(), Scalars: scalars, Norm: norm, ReferenceNorm: input.NativeNorm, NormAbsError: math.Abs(norm - input.NativeNorm), ClipScale: scale, F64ClipScale: float32(1 / input.NativeNorm), Elements: input.Elements, MaxRSS: maximumRSS, Final: final}
	if e = flowCaptureWrite(filepath.Join(root, "native-norm.json"), report); e != nil {
		t.Fatal(e)
	}
	t.Logf("native norm=%g F64=%g error=%g scale=%g F64scale=%g RSS=%d", norm, input.NativeNorm, report.NormAbsError, scale, report.F64ClipScale, maximumRSS)
}
