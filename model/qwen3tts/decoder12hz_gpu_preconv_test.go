package qwen3tts

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// TestDecoderPreConvPinnedGPU is a single released causal-convolution probe.
// It uses independently traced full Rust F32 tensors and does not dispatch
// production synthesis to the GPU.
func TestDecoderPreConvPinnedGPU(t *testing.T) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for the NVIDIA pre-convolution diagnostic")
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
	const frames, inputChannels, outputChannels, taps = 64, 512, 1024, 3
	readF32 := func(name, sha string, count int) []float32 {
		t.Helper()
		path := filepath.Join("testdata", "customvoice_0b6_ryan_hello", "decoder_preconv", name)
		if err := verifyReleasedFile(path, sha, int64(count*4)); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, count)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		}
		return out
	}
	input := readF32("quantized.full.f32le", "4cce48ec0c48189105f28e24ab6404cadbdec7130f84a3f07c19979989298a9d", inputChannels*frames)
	ref := readF32("preconv.full.f32le", "75c0a7feda1be45023577e1f2675f909b75351fd5bc977978e385597a4703685", outputChannels*frames)
	m, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		t.Fatal(err)
	}
	c := m.preConv
	if c.inChannels != inputChannels || c.outChannels != outputChannels || c.k != taps || c.dilation != 1 || c.groups() != 1 || len(c.weight) != inputChannels*outputChannels*taps || len(c.bias) != outputChannels {
		t.Fatal("unexpected released decoder pre-convolution geometry")
	}
	cpu, length, err := c.forward(input, frames)
	if err != nil || length != frames || len(cpu) != len(ref) {
		t.Fatalf("CPU pre-convolution geometry: length=%d values=%d error=%v", length, len(cpu), err)
	}
	for i, want := range ref {
		if cpu[i] != want {
			t.Fatalf("Rust/CPU pre-convolution mismatch at %d: got %.9g want %.9g", i, cpu[i], want)
		}
	}
	if !nvidia.SgemmReady() {
		t.Fatal("opt-in GPU gate requested but NVIDIA SGEMM is unavailable")
	}
	// Lower [channel,time] causal convolution to A=[time,channel*tap].
	// Zero-fill only the left causal boundary, matching the CPU im2col path.
	k := inputChannels * taps
	a := make([]float32, frames*k)
	for pos := 0; pos < frames; pos++ {
		for ic := 0; ic < inputChannels; ic++ {
			for tap := 0; tap < taps; tap++ {
				src := pos + tap - (taps - 1)
				if src >= 0 {
					a[pos*k+ic*taps+tap] = input[ic*frames+src]
				}
			}
		}
	}
	b := make([]float32, k*outputChannels)
	for out := 0; out < outputChannels; out++ {
		for in := 0; in < k; in++ {
			b[in*outputChannels+out] = c.weight[out*k+in]
		}
	}
	previousStats := nvidia.SetStatsEnabled(true)
	defer nvidia.SetStatsEnabled(previousStats)
	before := nvidia.StatsSnapshot()
	gemm := make([]float32, frames*outputChannels)
	// Mirror Candle's fixed 512-wide reductions. This is a diagnostic only:
	// separate host staging/downloads do not model production device residency.
	const block = 512
	for kk := 0; kk < k; kk += block {
		width := min(block, k-kk)
		chunkA := make([]float32, frames*width)
		chunkB := make([]float32, width*outputChannels)
		for row := 0; row < frames; row++ {
			copy(chunkA[row*width:(row+1)*width], a[row*k+kk:row*k+kk+width])
		}
		copy(chunkB, b[kk*outputChannels:(kk+width)*outputChannels])
		partial, err := nvidia.SgemmHost(frames, outputChannels, width, 1, chunkA, chunkB)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range partial {
			gemm[i] += v
		}
	}
	after := nvidia.StatsSnapshot()
	if after.KernelLaunches-before.KernelLaunches != 3 || after.HostToDeviceBytes-before.HostToDeviceBytes != uint64((frames*k+k*outputChannels)*4) || after.DeviceToHostBytes-before.DeviceToHostBytes != uint64(3*frames*outputChannels*4) || after.Mallocs-before.Mallocs != 9 || after.Frees-before.Frees != 9 || after.MallocBytes-before.MallocBytes != after.FreeBytes-before.FreeBytes {
		t.Fatalf("unexpected GPU lifecycle: before=%+v after=%+v", before, after)
	}
	maxRust, maxCPU := float64(0), float64(0)
	for out := 0; out < outputChannels; out++ {
		for pos := 0; pos < frames; pos++ {
			value := gemm[pos*outputChannels+out] + c.bias[out]
			index := out*frames + pos
			rustDiff := math.Abs(float64(value) - float64(ref[index]))
			cpuDiff := math.Abs(float64(value) - float64(cpu[index]))
			if math.IsNaN(rustDiff) || math.IsInf(rustDiff, 0) || math.IsNaN(cpuDiff) || math.IsInf(cpuDiff, 0) {
				t.Fatalf("nonfinite GPU pre-convolution difference at %d", index)
			}
			if rustDiff > maxRust {
				maxRust = rustDiff
			}
			if cpuDiff > maxCPU {
				maxCPU = cpuDiff
			}
		}
	}
	t.Logf("64-frame decoder pre-convolution GPU/Rust max=%.9g GPU/CPU max=%.9g; launches=%d H2D=%d D2H=%d alloc/free=%d bytes", maxRust, maxCPU, after.KernelLaunches-before.KernelLaunches, after.HostToDeviceBytes-before.HostToDeviceBytes, after.DeviceToHostBytes-before.DeviceToHostBytes, after.MallocBytes-before.MallocBytes)
	if maxRust > 1e-6 || maxCPU > 1e-6 {
		t.Fatalf("GPU pre-convolution error exceeds diagnostic 1e-6 threshold: Rust=%.9g CPU=%.9g", maxRust, maxCPU)
	}
}
