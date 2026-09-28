package qwen3tts

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// TestDecoderOutputProjectionPinnedGPU isolates one released decoder projection.
// Its F32 input is the independently traced Rust/Candle final-norm output;
// neither the production CPU decoder nor this test supplies an expected GPU value.
func TestDecoderOutputProjectionPinnedGPU(t *testing.T) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for the opt-in NVIDIA decoder projection gate")
	}
	root := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	if root == "" {
		t.Fatal("set GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	}
	for name, sha := range map[string]string{
		"speech_tokenizer/config.json":       "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
		"speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
	} {
		if err := verifyReleasedFile(filepath.Join(root, name), sha, 0); err != nil {
			t.Fatal(err)
		}
	}
	sentence64Fixture(t)
	traceDir := filepath.Join("testdata", "customvoice_0b6_ryan_hello", "decoder_projection")
	readF32 := func(name, sha string, count int) []float32 {
		t.Helper()
		path := filepath.Join(traceDir, name)
		if err := verifyReleasedFile(path, sha, int64(count*4)); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		values := make([]float32, count)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return values
	}
	const frames, inDim, outDim = 64, 512, 1024
	input := readF32("finalnorm.full.f32le", "a0e9e1ee8b5ff796aaf8a2228ac0fdb53666b870474b44d1b79fe630f3145713", frames*inDim)
	ref := readF32("outputproj.f32le", "85ef81eefb6e8896d39fd320f000952904ca4c41b9edec71b24e32b2560c7df8", 2048)
	fullRef := readF32("outputproj.full.f32le", "9324dd4c394cf844f55dd121ee384b51b0afc83bf4082401c2c24cb59d50745d", frames*outDim)
	for i, want := range ref {
		index := i * (len(fullRef) - 1) / (len(ref) - 1)
		if fullRef[index] != want {
			t.Fatalf("sampled Rust output differs from full Rust trace at index %d", index)
		}
	}
	m, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		t.Fatal(err)
	}
	l := m.outputProjection
	if m.cfg.HiddenSize != inDim || m.cfg.LatentDim != outDim || l.inDim != inDim || l.outDim != outDim || len(l.weight) != inDim*outDim || len(l.bias) != outDim {
		t.Fatal("unexpected released decoder output-projection topology")
	}
	cpu := make([]float32, frames*outDim)
	for row := 0; row < frames; row++ {
		if err := decoderLinearForward(l, cpu[row*outDim:(row+1)*outDim], input[row*inDim:(row+1)*inDim]); err != nil {
			t.Fatal(err)
		}
	}
	maxSample := func(label string, got []float32, allowed float64) float64 {
		t.Helper()
		maxErr := float64(0)
		for i, want := range ref {
			index := i * (len(got) - 1) / (len(ref) - 1)
			diff := math.Abs(float64(got[index]) - float64(want))
			if math.IsNaN(diff) || math.IsInf(diff, 0) {
				t.Fatalf("%s nonfinite difference at sampled index %d", label, i)
			}
			if diff > maxErr {
				maxErr = diff
			}
		}
		if maxErr > allowed {
			t.Fatalf("%s maximum sampled error %.9g exceeds %.9g", label, maxErr, allowed)
		}
		return maxErr
	}
	cpuErr := maxSample("Rust/Go CPU output projection", cpu, 0)
	for i, want := range fullRef {
		if cpu[i] != want {
			t.Fatalf("full Rust/Go CPU output differs at index %d: got %.9g want %.9g", i, cpu[i], want)
		}
	}
	if !nvidia.SgemmReady() {
		t.Fatal("opt-in GPU gate requested but NVIDIA SGEMM is unavailable")
	}
	transposed := make([]float32, inDim*outDim)
	for out := 0; out < outDim; out++ {
		for in := 0; in < inDim; in++ {
			transposed[in*outDim+out] = l.weight[out*inDim+in]
		}
	}
	previousStats := nvidia.SetStatsEnabled(true)
	defer nvidia.SetStatsEnabled(previousStats)
	before := nvidia.StatsSnapshot()
	gpu, err := nvidia.SgemmHost(frames, outDim, inDim, 1, input, transposed)
	after := nvidia.StatsSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if after.KernelLaunches-before.KernelLaunches != 1 || after.HostToDeviceBytes-before.HostToDeviceBytes != uint64((frames*inDim+inDim*outDim)*4) || after.DeviceToHostBytes-before.DeviceToHostBytes != uint64(frames*outDim*4) || after.Mallocs-before.Mallocs != 3 || after.Frees-before.Frees != 3 || after.MallocBytes-before.MallocBytes != after.FreeBytes-before.FreeBytes {
		t.Fatalf("unexpected GPU transfer or allocation lifecycle: before=%+v after=%+v", before, after)
	}
	for row := 0; row < frames; row++ {
		for out, bias := range l.bias {
			gpu[row*outDim+out] += bias
		}
	}
	gpuErr := maxSample("Rust/GPU output projection", gpu, 1e-6)
	maxCPU := float64(0)
	maxRust := float64(0)
	for i, value := range gpu {
		diff := math.Abs(float64(value) - float64(cpu[i]))
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			t.Fatalf("nonfinite GPU/CPU difference at index %d", i)
		}
		if diff > maxCPU {
			maxCPU = diff
		}
		rustDiff := math.Abs(float64(value) - float64(fullRef[i]))
		if rustDiff > maxRust {
			maxRust = rustDiff
		}
	}
	if maxRust > 1e-6 {
		t.Fatalf("full GPU/Rust output-projection error %.9g exceeds 1e-6", maxRust)
	}
	if maxCPU > 1e-6 {
		t.Fatalf("GPU/CPU output-projection error %.9g exceeds 1e-6", maxCPU)
	}
	t.Logf("64x512 -> 64x1024 output projection: Rust/CPU sampled=%.9g Rust/GPU sampled=%.9g full GPU/Rust=%.9g full GPU/CPU=%.9g; launches=%d H2D=%d bytes D2H=%d bytes alloc/free=%d bytes", cpuErr, gpuErr, maxRust, maxCPU, after.KernelLaunches-before.KernelLaunches, after.HostToDeviceBytes-before.HostToDeviceBytes, after.DeviceToHostBytes-before.DeviceToHostBytes, after.MallocBytes-before.MallocBytes)
}
