package nvidia

import (
	"compress/gzip"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// Opt-in native RTX 3060 gate for the released diarization 8-frame stacking
// projection. It does not execute the 31-layer transformer or speaker head.
func TestNemotronPTXNativeDiarizationStacking(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_PTX_STACKING") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_PTX_STACKING=1")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 2*time.Minute {
		t.Fatal("native test requires -timeout<=2m")
	}
	dir := os.Getenv("GO_PHERENCE_NEMOTRON_STACKING_FIXTURE_DIR")
	if dir == "" {
		t.Fatal("set GO_PHERENCE_NEMOTRON_STACKING_FIXTURE_DIR")
	}
	const rows, in, outDim = 138, 1024, 512
	read := func(name string, n int) []float32 {
		t.Helper()
		raw, e := os.ReadFile(filepath.Join(dir, name))
		if e != nil || len(raw) != n*4 {
			t.Fatalf("%s bytes=%d err=%v", name, len(raw), e)
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out
	}
	features, weight := read("stacked.f32", rows*in), read("weight.f32", outDim*in)
	file, err := os.Open("../../../model/nemotrondiarization/testdata/jfk_stacking_transformers_5_18.f32.gz")
	if err != nil {
		t.Fatal(err)
	}
	gz, err := gzip.NewReader(file)
	if err != nil {
		file.Close()
		t.Fatal(err)
	}
	ref, err := io.ReadAll(io.LimitReader(gz, rows*outDim*4+1))
	gz.Close()
	file.Close()
	if err != nil || len(ref) != rows*outDim*4 {
		t.Fatalf("reference size=%d err=%v", len(ref), err)
	}
	if !Init() || !SgemmReady() {
		t.Fatal("CUDA SGEMM unavailable")
	}
	device := DeviceName()
	if !strings.Contains(device, "GeForce RTX 3060") {
		t.Fatalf("unexpected CUDA device %q", device)
	}
	start := time.Now()
	transposed := make([]float32, in*outDim)
	for k := 0; k < in; k++ {
		for j := 0; j < outDim; j++ {
			transposed[k*outDim+j] = weight[j*in+k]
		}
	}
	transposeTime := time.Since(start)
	dX, err := Malloc(rows * in)
	if err != nil {
		t.Fatal(err)
	}
	defer dX.Free()
	dW, err := Malloc(in * outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer dW.Free()
	dY, err := Malloc(rows * outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer dY.Free()
	start = time.Now()
	if err := dX.Upload(features); err != nil {
		t.Fatal(err)
	}
	if err := dW.Upload(transposed); err != nil {
		t.Fatal(err)
	}
	upload := time.Since(start)
	start = time.Now()
	if err := Sgemm(rows, outDim, in, 1, dX, dW, dY); err != nil {
		t.Fatal(err)
	}
	if err := SyncErr(); err != nil {
		t.Fatal(err)
	}
	first := time.Since(start)
	got := make([]float32, rows*outDim)
	start = time.Now()
	if err := dY.Download(got); err != nil {
		t.Fatal(err)
	}
	download := time.Since(start)
	var maxAbs, sumAbs float64
	var outside int
	for i, v := range got {
		want := math.Float32frombits(binary.LittleEndian.Uint32(ref[i*4:]))
		d := math.Abs(float64(v - want))
		if d > maxAbs {
			maxAbs = d
		}
		sumAbs += d
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 2e-3+2e-4*math.Abs(float64(want)) {
			outside++
		}
	}
	mean := sumAbs / float64(len(got))
	if outside != 0 || mean > 1e-4 {
		t.Fatalf("PTX stack max=%g mean=%g outside=%d", maxAbs, mean, outside)
	}
	cpu := make([]float32, len(got))
	cpuTimes := make([]time.Duration, 5)
	for i := range cpuTimes {
		clear(cpu)
		start = time.Now()
		if !simd.DenseNTTo(cpu, features, weight, rows, outDim, in, 1, in, in, outDim) {
			t.Fatal("CPU SIMD rejected shape")
		}
		cpuTimes[i] = time.Since(start)
	}
	var cpuMax float64
	for i, v := range got {
		cpuMax = math.Max(cpuMax, math.Abs(float64(v-cpu[i])))
	}
	if cpuMax > 2e-3 {
		t.Fatalf("PTX/CPU drift=%g", cpuMax)
	}
	resident := make([]time.Duration, 5)
	for i := range resident {
		start = time.Now()
		if err := Sgemm(rows, outDim, in, 1, dX, dW, dY); err != nil {
			t.Fatal(err)
		}
		if err := SyncErr(); err != nil {
			t.Fatal(err)
		}
		resident[i] = time.Since(start)
	}
	if err := dY.Download(got); err != nil {
		t.Fatal(err)
	}
	for _, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("non-finite resident output")
		}
	}
	t.Logf("NEMOTRON_PTX_STACKING device=%q max_abs=%g mean_abs=%g cpu_max_abs=%g transpose=%s upload=%s first_dispatch_sync=%s download=%s resident_dispatch_sync=%v cpu_simd=%v", device, maxAbs, mean, cpuMax, transposeTime, upload, first, download, resident, cpuTimes)
}
