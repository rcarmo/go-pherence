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
)

// Probe the existing PTX F32 SGEMM on PyTorch-generated first-stem patches.
// It does not implement the remaining ASR subsampler or a GPU audio frontend.
func TestNemotronPTXNativeASRStemProjection(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_PTX_STEM") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_PTX_STEM=1")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 2*time.Minute {
		t.Fatal("native test requires -timeout<=2m")
	}
	dir := os.Getenv("GO_PHERENCE_NEMOTRON_STEM_FIXTURE_DIR")
	if dir == "" {
		t.Fatal("set GO_PHERENCE_NEMOTRON_STEM_FIXTURE_DIR")
	}
	const positions, k, outDim = 1105, 9, 256
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
	patches, weight, bias := read("patches.f32", positions*k), read("weight.f32", outDim*k), read("bias.f32", outDim)
	file, e := os.Open("../../../model/nemotronasr/testdata/output.f32.gz")
	if e != nil {
		t.Fatal(e)
	}
	gz, e := gzip.NewReader(file)
	if e != nil {
		file.Close()
		t.Fatal(e)
	}
	ref, e := io.ReadAll(io.LimitReader(gz, positions*outDim*4+1))
	gz.Close()
	file.Close()
	if e != nil || len(ref) != positions*outDim*4 {
		t.Fatalf("reference bytes=%d err=%v", len(ref), e)
	}
	if !Init() || !SgemmReady() {
		t.Fatal("CUDA SGEMM unavailable")
	}
	device := DeviceName()
	if !strings.Contains(device, "GeForce RTX 3060") {
		t.Fatalf("unexpected CUDA device %q", device)
	}
	start := time.Now()
	transposed := make([]float32, k*outDim)
	for i := 0; i < k; i++ {
		for j := 0; j < outDim; j++ {
			transposed[i*outDim+j] = weight[j*k+i]
		}
	}
	transpose := time.Since(start)
	dX, e := Malloc(positions * k)
	if e != nil {
		t.Fatal(e)
	}
	defer dX.Free()
	dW, e := Malloc(k * outDim)
	if e != nil {
		t.Fatal(e)
	}
	defer dW.Free()
	dY, e := Malloc(positions * outDim)
	if e != nil {
		t.Fatal(e)
	}
	defer dY.Free()
	start = time.Now()
	if e := dX.Upload(patches); e != nil {
		t.Fatal(e)
	}
	if e := dW.Upload(transposed); e != nil {
		t.Fatal(e)
	}
	upload := time.Since(start)
	start = time.Now()
	if e := Sgemm(positions, outDim, k, 1, dX, dW, dY); e != nil {
		t.Fatal(e)
	}
	if e := SyncErr(); e != nil {
		t.Fatal(e)
	}
	first := time.Since(start)
	projected := make([]float32, positions*outDim)
	start = time.Now()
	if e := dY.Download(projected); e != nil {
		t.Fatal(e)
	}
	download := time.Since(start)
	result := make([]float32, len(projected))
	start = time.Now()
	for ch := 0; ch < outDim; ch++ {
		for pos := 0; pos < positions; pos++ {
			result[ch*positions+pos] = projected[pos*outDim+ch] + bias[ch]
		}
	}
	reorder := time.Since(start)
	var maxAbs, sumAbs, float64Max, float64Sum float64
	var outside, differing int
	for i, v := range result {
		want := math.Float32frombits(binary.LittleEndian.Uint32(ref[i*4:]))
		ch, pos := i/positions, i%positions
		var independent float64 = float64(bias[ch])
		for j := 0; j < k; j++ {
			independent += float64(patches[pos*k+j]) * float64(weight[ch*k+j])
		}
		refError := math.Abs(float64(v) - independent)
		float64Max = math.Max(float64Max, refError)
		float64Sum += refError
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || refError > 3e-5+2e-5*math.Abs(independent) {
			outside++
		}
		d := math.Abs(float64(v - want))
		if d != 0 {
			differing++
		}
		if d > maxAbs {
			maxAbs = d
		}
		sumAbs += d
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 3e-5+2e-5*math.Abs(float64(want)) {
			outside++
		}
	}
	mean := sumAbs / float64(len(result))
	if outside != 0 || mean > 3e-6 || float64Sum/float64(len(result)) > 3e-6 {

		t.Fatalf("PTX ASR stem max=%g mean=%g outside=%d", maxAbs, mean, outside)
	}
	resident := make([]time.Duration, 5)
	for i := range resident {
		start = time.Now()
		if e := Sgemm(positions, outDim, k, 1, dX, dW, dY); e != nil {
			t.Fatal(e)
		}
		if e := SyncErr(); e != nil {
			t.Fatal(e)
		}
		resident[i] = time.Since(start)
	}
	t.Logf("NEMOTRON_PTX_ASR_STEM device=%q max_abs=%g mean_abs=%g float64_max=%g float64_mean=%g differing=%d weight_transpose=%s upload=%s first_dispatch_sync=%s download=%s reorder_bias=%s resident=%v", device, maxAbs, mean, float64Max, float64Sum/float64(len(result)), differing, transpose, upload, first, download, reorder, resident)
}
