package nvidia

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// Opt-in native PTX projection probe on independently generated PyTorch STFT
// power and mel filters. This does not implement the FFT or Nemotron models.
func TestNemotronPTXNativeMelProjection(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_PTX") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_PTX=1 for authorised native qualification")
	}
	deadline, bounded := t.Deadline()
	if !bounded || time.Until(deadline) > 2*time.Minute {
		t.Fatal("native qualification requires go test -timeout at most 2m")
	}
	folder := os.Getenv("GO_PHERENCE_NEMOTRON_PTX_FIXTURE_DIR")
	if folder == "" {
		t.Fatal("set GO_PHERENCE_NEMOTRON_PTX_FIXTURE_DIR to independently generated power/filter tensors")
	}
	const rows, bins, mels = 1100, 257, 128
	read := func(name string, count int) []float32 {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(folder, name))
		if err != nil || len(b) != count*4 {
			t.Fatalf("%s: bytes=%d want=%d err=%v", name, len(b), count*4, err)
		}
		out := make([]float32, count)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
		return out
	}
	power := read("power.f32", 1101*bins)[:rows*bins]
	filters := read("filters.f32", mels*bins)
	if !Init() || !SgemmReady() {
		t.Fatal("CUDA SGEMM unavailable")
	}
	device := DeviceName()
	if !strings.Contains(device, "GeForce RTX 3060") {
		t.Fatalf("unexpected CUDA device %q", device)
	}
	// The runtime GEMM consumes B[K,N]. Prepare the transposed filter once;
	// transpose time belongs in full-request accounting, not resident dispatch.
	start := time.Now()
	weights := make([]float32, bins*mels)
	for k := 0; k < bins; k++ {
		for m := 0; m < mels; m++ {
			weights[k*mels+m] = filters[m*bins+k]
		}
	}
	transposeTime := time.Since(start)
	dA, err := Malloc(rows * bins)
	if err != nil {
		t.Fatal(err)
	}
	defer dA.Free()
	dB, err := Malloc(bins * mels)
	if err != nil {
		t.Fatal(err)
	}
	defer dB.Free()
	dC, err := Malloc(rows * mels)
	if err != nil {
		t.Fatal(err)
	}
	defer dC.Free()
	start = time.Now()
	if err := dA.Upload(power); err != nil {
		t.Fatal(err)
	}
	if err := dB.Upload(weights); err != nil {
		t.Fatal(err)
	}
	upload := time.Since(start)
	start = time.Now()
	if err := Sgemm(rows, mels, bins, 1, dA, dB, dC); err != nil {
		t.Fatal(err)
	}
	if err := SyncErr(); err != nil {
		t.Fatal(err)
	}
	first := time.Since(start)
	got := make([]float32, rows*mels)
	start = time.Now()
	if err := dC.Download(got); err != nil {
		t.Fatal(err)
	}
	download := time.Since(start)
	var maxAbs, sumAbs float64
	var outside int
	for r := 0; r < rows; r++ {
		for m := 0; m < mels; m++ {
			var want float64
			for k := 0; k < bins; k++ {
				want += float64(power[r*bins+k]) * float64(filters[m*bins+k])
			}
			v := float64(got[r*mels+m])
			d := math.Abs(v - want)
			if d > maxAbs {
				maxAbs = d
			}
			sumAbs += d
			if math.IsNaN(v) || math.IsInf(v, 0) || d > 2e-3+2e-4*math.Abs(want) {
				outside++
			}
		}
	}
	if outside != 0 || sumAbs/float64(len(got)) > 1e-4 {
		t.Fatalf("PTX mel projection error max=%g mean=%g outside=%d", maxAbs, sumAbs/float64(len(got)), outside)
	}
	cpu := make([]float32, rows*mels)
	cpuTimes := make([]time.Duration, 5)
	for i := range cpuTimes {
		clear(cpu)
		start = time.Now()
		if !simd.DenseNTTo(cpu, power, filters, rows, mels, bins, 1, bins, bins, mels) {
			t.Fatal("CPU SIMD projection rejected input")
		}
		cpuTimes[i] = time.Since(start)
	}
	var cpuMaxAbs float64
	for i, v := range got {
		cpuMaxAbs = math.Max(cpuMaxAbs, math.Abs(float64(v-cpu[i])))
	}
	if cpuMaxAbs > 2e-3 {
		t.Fatalf("PTX/CPU projection drift max=%g", cpuMaxAbs)
	}
	resident := make([]time.Duration, 5)
	for i := range resident {
		start = time.Now()
		if err := Sgemm(rows, mels, bins, 1, dA, dB, dC); err != nil {
			t.Fatal(err)
		}
		if err := SyncErr(); err != nil {
			t.Fatal(err)
		}
		resident[i] = time.Since(start)
	}
	if err := dC.Download(got); err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(float64(got[0])) || math.IsInf(float64(got[0]), 0) {
		t.Fatal("non-finite resident output")
	}
	t.Logf("NEMOTRON_PTX_PROJECTION device=%q rows=%d bins=%d mels=%d max_abs=%g mean_abs=%g cpu_max_abs=%g transpose=%s upload=%s first_dispatch_sync=%s download=%s resident_dispatch_sync=%v cpu_simd_projection=%v", device, rows, bins, mels, maxAbs, sumAbs/float64(len(got)), cpuMaxAbs, transposeTime, upload, first, download, resident, cpuTimes)
}
