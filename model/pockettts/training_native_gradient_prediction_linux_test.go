//go:build linux

package pockettts

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Standalone diagnostic arithmetic: no optimiser call or state mutation.
func parameterFirstStepPrediction(initial, gradient, scale float32, c AdamWConfig) (float32, error) {
	if !isFinite(initial) || !isFinite(gradient) || !isFinite(scale) || scale <= 0 || scale > 1 {
		return 0, os.ErrInvalid
	}
	x := gradient * scale
	m, v := (1-c.Beta1)*x, ((1-c.Beta2)*x)*x
	p := initial * (1 - c.LearningRate*c.WeightDecay)
	p -= c.LearningRate * (m / (1 - c.Beta1)) / (float32(math.Sqrt(float64(v/(1-c.Beta2)))) + c.Epsilon)
	if !isFinite(p) {
		return 0, os.ErrInvalid
	}
	return p, nil
}
func TestPocketParameterPredictionTiny(t *testing.T) {
	c := DefaultAdamWConfig()
	for _, g := range []float32{0, 1, -1, 1e-9, -1e-9} {
		p, e := parameterFirstStepPrediction(.5, g, 1, c)
		if e != nil {
			t.Fatal(e)
		}
		ideal := float64(.5) * (1 - float64(c.LearningRate*c.WeightDecay))
		if g != 0 {
			ideal -= float64(c.LearningRate) * float64(g) / (math.Abs(float64(g)) + float64(c.Epsilon))
		}
		if math.Abs(float64(p)-ideal) > 1e-7 {
			t.Fatal(p, ideal)
		}
	}
	for _, x := range []struct{ p, g, s float32 }{{1, 1, 0}, {1, 1, 2}, {1, float32(math.NaN()), 1}, {float32(math.Inf(1)), 1, 1}} {
		if _, e := parameterFirstStepPrediction(x.p, x.g, x.s, c); e == nil {
			t.Fatal("admission")
		}
	}
}
func TestPocketCompleteCapturedGradientPrediction(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_PARAMETER_PREDICT") != "1" {
		t.Skip("explicit read-only fullcapturedgradientprediction")
	}
	root := os.Getenv("GO_PHERENCE_POCKET_PARAMETER_OUTPUT")
	var inventory struct {
		NativeNorm, ReferenceNorm float64
		Elements                  int64
		Entries                   []struct {
			Name                      string `json:"name"`
			Elements                  int    `json:"elements"`
			Offset                    int64  `json:"offset"`
			NativeFile, ReferenceFile string
		}
	}
	pin := os.Getenv("GO_PHERENCE_POCKET_PARAMETER_INVENTORY_SHA256")
	if pin == "" {
		pin = "5b99d89dd42a50e5332cd9933a496f463b5c90aecac805f795833f44802bbf5d"
	}
	if e := flowCapturePinned(filepath.Join(root, "full-gradient-inventory-byteorder.json"), pin, 1<<20, &inventory); e != nil {
		t.Fatal(e)
	}
	if inventory.Elements != 89449730 || len(inventory.Entries) != 131 {
		t.Fatal("inventory")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	params, e := fullParameterMap(lm, flow, w)
	if e != nil {
		t.Fatal(e)
	}
	cpuPath := releasedAsset(t, "GO_PHERENCE_POCKET_SHARED_FULL_CPU", "d0d366f789a6ab0078bd0f1534d16a38dddb4e08a3193797be23c9f6b1794e5c", 1431217112)
	gpuPath := releasedAsset(t, "GO_PHERENCE_POCKET_SHARED_FULL_NATIVE", "bc4092eb94c8434f11332c04d4b95715862d913ef89087ac57a4b794852868fd", 1431217112)
	cpu, e := sharedOpenCheckpointP(cpuPath)
	if e != nil {
		t.Fatal(e)
	}
	defer cpu.file.Close()
	gpu, e := sharedOpenCheckpointP(gpuPath)
	if e != nil {
		t.Fatal(e)
	}
	defer gpu.file.Close()
	if cpu.header != gpu.header || cpu.count != 131 || binary.LittleEndian.Uint64(cpu.header[4:12]) != 1 {
		t.Fatal("checkpoint association")
	}
	c := DefaultAdamWConfig()
	expected := []float32{c.LearningRate, c.Beta1, c.Beta2, c.Epsilon, c.WeightDecay, .999}
	for i, v := range expected {
		if math.Float32frombits(binary.LittleEndian.Uint32(cpu.header[12+4*i:])) != v {
			t.Fatal("optimizer config")
		}
	}
	nativeScale, refScale := float32(1/inventory.NativeNorm), float32(1/inventory.ReferenceNorm)
	qualified := os.Getenv("GO_PHERENCE_POCKET_PARAMETER_ENFORCE_CPU_GATE") == "1"
	if qualified {
		var norm struct {
			ClipScale float32
			Elements  int64
		}
		if e := flowCapturePinned(filepath.Join(root, "native-norm.json"), os.Getenv("GO_PHERENCE_POCKET_PARAMETER_NORM_SHA256"), 1<<20, &norm); e != nil {
			t.Fatal(e)
		}
		if norm.Elements != inventory.Elements || !isFinite(norm.ClipScale) || norm.ClipScale <= 0 || norm.ClipScale > 1 {
			t.Fatal("qualified norm input")
		}
		nativeScale = norm.ClipScale
	}
	type distribution struct {
		Values, Above4e6 int64
		Max, Mean, RMS   float64
	}
	report := struct {
		Scope                                                                                    string
		NativeNorm, ReferenceNorm                                                                float64
		NativeScale, ReferenceScale                                                              float32
		CapturedVsSavedNative, CapturedVsSavedCPU, ReferenceVsSavedCPU, OriginalSavedCPUvsNative distribution
	}{Scope: "Allcaptured89,449,730gradients/F64clipfirststepdiagnostic,neverapplied. Nativeclippingfailedpreflight/notqualified;CPUrefsatsharednativeconditionsnotoriginalfullCPUoracle.", NativeNorm: inventory.NativeNorm, ReferenceNorm: inventory.ReferenceNorm, NativeScale: nativeScale, ReferenceScale: refScale}
	if qualified {
		report.Scope = "Corrected production arithmetic, all named gradients and measured native clipping; read-only first-step prediction vs independent original CPU checkpoint at unchanged4e-6, no optimiser mutation."
	}
	type tensorError struct {
		Name             string
		Values, Above4e6 int64
		Max              float64
	}
	var tensorErrors []tensorError
	type outlier struct {
		Name                                                                  string
		Index                                                                 int
		Initial, NativeGradient, ReferenceGradient, NativeP, ReferenceP, CPUP float32
	}
	var outliers []outlier
	ds := []*distribution{&report.CapturedVsSavedNative, &report.CapturedVsSavedCPU, &report.ReferenceVsSavedCPU, &report.OriginalSavedCPUvsNative}
	sums, squares := [4]float64{}, [4]float64{}
	var a, b, g, r [32768]byte
	files := map[string]*os.File{}
	defer func() {
		for _, f := range files {
			f.Close()
		}
	}()
	readGradient := func(name string, offset int64, dst []byte) {
		t.Helper()
		f := files[name]
		if f == nil {
			var err error
			f, err = os.Open(filepath.Join(root, name))
			if err != nil {
				t.Fatal(err)
			}
			files[name] = f
		}
		if _, err := f.ReadAt(dst, offset); err != nil {
			t.Fatal(err)
		}
	}
	for index, entry := range inventory.Entries {
		name, n, e := cpu.next()
		if e != nil {
			t.Fatal(e)
		}
		other, m, e := gpu.next()
		if e != nil || name != entry.Name || other != name || m != n || entry.Elements != n || len(params[name]) != n {
			t.Fatal("gradient/P inventory", index, name, entry.Name, e)
		}
		te := tensorError{Name: name}
		for offset := 0; offset < n; {
			count := min(n-offset, len(a)/4)
			if _, e = io.ReadFull(cpu.file, a[:4*count]); e != nil {
				t.Fatal(e)
			}
			if _, e = io.ReadFull(gpu.file, b[:4*count]); e != nil {
				t.Fatal(e)
			}
			readGradient(entry.NativeFile, entry.Offset+int64(4*offset), g[:4*count])
			readGradient(entry.ReferenceFile, entry.Offset+int64(4*offset), r[:4*count])
			for i := 0; i < count; i++ {
				cv, nv := math.Float32frombits(binary.LittleEndian.Uint32(a[4*i:])), math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
				gv, rv := math.Float32frombits(binary.LittleEndian.Uint32(g[4*i:])), math.Float32frombits(binary.LittleEndian.Uint32(r[4*i:]))
				p, e := parameterFirstStepPrediction(params[name][offset+i], gv, nativeScale, c)
				if e != nil {
					t.Fatal(e)
				}
				q, e := parameterFirstStepPrediction(params[name][offset+i], rv, refScale, c)
				if e != nil {
					t.Fatal(e)
				}
				if !isFinite(cv) || !isFinite(nv) {
					t.Fatal("nonfinite savedP")
				}
				dcpu := math.Abs(float64(p) - float64(cv))
				te.Values++
				te.Max = math.Max(te.Max, dcpu)
				if dcpu > 4e-6 {
					te.Above4e6++
					if len(outliers) < 4096 {
						outliers = append(outliers, outlier{name, offset + i, params[name][offset+i], gv, rv, p, q, cv})
					}
				}
				for j, d := range []float64{math.Abs(float64(p) - float64(nv)), math.Abs(float64(p) - float64(cv)), math.Abs(float64(q) - float64(cv)), math.Abs(float64(cv) - float64(nv))} {
					out := ds[j]
					out.Values++
					out.Max = math.Max(out.Max, d)
					sums[j] += d
					squares[j] += d * d
					if d > 4e-6 {
						out.Above4e6++
					}
				}
			}
			offset += count
		}
		tensorErrors = append(tensorErrors, te)
	}
	for j, out := range ds {
		out.Mean = sums[j] / float64(out.Values)
		out.RMS = math.Sqrt(squares[j] / float64(out.Values))
	}
	if report.OriginalSavedCPUvsNative.Above4e6 != 1293 || report.CapturedVsSavedNative.Values != 89449730 {
		t.Fatal("original gate/inventory reproduction", report)
	}
	if e = flowCaptureWrite(filepath.Join(root, "captured-P-prediction.json"), report); e != nil {
		t.Fatal(e)
	}
	if e = flowCaptureWrite(filepath.Join(root, "captured-P-tensor-errors.json"), tensorErrors); e != nil {
		t.Fatal(e)
	}
	if e = flowCaptureWrite(filepath.Join(root, "captured-P-outliers.json"), outliers); e != nil {
		t.Fatal(e)
	}
	t.Logf("Pprediction CPUabove=%d max=%g refCPUabove=%d original=%d;no mutation", report.CapturedVsSavedCPU.Above4e6, report.CapturedVsSavedCPU.Max, report.ReferenceVsSavedCPU.Above4e6, report.OriginalSavedCPUvsNative.Above4e6)
	if qualified && report.CapturedVsSavedCPU.Above4e6 != 0 {
		t.Fatal("independent CPU parameter gate failed", report.CapturedVsSavedCPU)
	}
}
