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

// Prepared ASR final subsampling projection only. The preceding causal convolutions,
// encoder, RNN-T and full-request transfer/dispatch are not covered.
func TestNemotronPTXNativeASRFinalProjection(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_PTX_ASR_PROJECTION") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_PTX_ASR_PROJECTION=1")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 2*time.Minute {
		t.Fatal("native test requires -timeout<=2m")
	}
	dir := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_PROJECTION_FIXTURE_DIR")
	if dir == "" {
		t.Fatal("set GO_PHERENCE_NEMOTRON_ASR_PROJECTION_FIXTURE_DIR")
	}
	const rows, in, outDim = 5, 4352, 1024
	read := func(name string, n int) []float32 {
		t.Helper()
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil || len(raw) != n*4 {
			t.Fatalf("%s bytes=%d err=%v", name, len(raw), err)
		}
		values := make([]float32, n)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return values
	}
	features, weight := read("input.f32", rows*in), read("weight.f32", outDim*in)
	biasValues := read("bias.f32", outDim)
	fixture, err := os.Open("../../../model/nemotronasr/testdata/projected.f32.gz")
	if err != nil {
		t.Fatal(err)
	}
	gz, err := gzip.NewReader(fixture)
	if err != nil {
		fixture.Close()
		t.Fatal(err)
	}
	raw, err := io.ReadAll(io.LimitReader(gz, rows*outDim*4+1))
	gz.Close()
	fixture.Close()
	if err != nil || len(raw) != rows*outDim*4 {
		t.Fatalf("reference bytes=%d err=%v", len(raw), err)
	}
	reference := make([]float32, rows*outDim)
	for i := range reference {
		reference[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
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
	for col := 0; col < in; col++ {
		for j := 0; j < outDim; j++ {
			transposed[col*outDim+j] = weight[j*in+col]
		}
	}
	transpose := time.Since(start)
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
	got := make([]float32, len(reference))
	start = time.Now()
	if err := dY.Download(got); err != nil {
		t.Fatal(err)
	}
	download := time.Since(start)
	for row := 0; row < rows; row++ {
		for j := 0; j < outDim; j++ {
			got[row*outDim+j] += biasValues[j]
		}
	}
	var maxAbs, sumAbs float64
	var outside int
	for i, value := range got {
		delta := math.Abs(float64(value - reference[i]))
		maxAbs = math.Max(maxAbs, delta)
		sumAbs += delta
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(reference[i])) {
			outside++
		}
	}
	mean := sumAbs / float64(len(got))
	if outside != 0 || mean > 5e-5 {
		t.Fatalf("PTX ASR projection max=%g mean=%g outside=%d", maxAbs, mean, outside)
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
	for row := 0; row < rows; row++ {
		for j := 0; j < outDim; j++ {
			cpu[row*outDim+j] += biasValues[j]
		}
	}
	var cpuMax float64
	for i, value := range got {
		cpuMax = math.Max(cpuMax, math.Abs(float64(value-cpu[i])))
	}
	if cpuMax > 3e-3 {
		t.Fatalf("PTX/CPU ASR projection drift=%g", cpuMax)
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
	for i, value := range got {
		actual := value + biasValues[i%outDim]
		delta := math.Abs(float64(actual - reference[i]))
		if math.IsNaN(float64(actual)) || math.IsInf(float64(actual), 0) || delta > 3e-4+2e-5*math.Abs(float64(reference[i])) {
			t.Fatalf("resident output drift at %d: %g", i, delta)
		}
	}
	t.Logf("NEMOTRON_PTX_ASR_PROJECTION device=%q max_abs=%g mean_abs=%g cpu_max_abs=%g transpose=%s upload=%s first_dispatch_sync=%s download=%s resident_dispatch_sync=%v cpu_simd=%v", device, maxAbs, mean, cpuMax, transpose, upload, first, download, resident, cpuTimes)
}
