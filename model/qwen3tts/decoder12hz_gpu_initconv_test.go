package qwen3tts

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// TestDecoderInitConvPinnedGPU isolates the released 7-tap decoder-initial
// causal convolution. It is a diagnostic, not a GPU waveform dispatch.
func TestDecoderInitConvPinnedGPU(t *testing.T) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for the NVIDIA decoder-initial convolution gate")
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
	const length, inputChannels, outputChannels, taps = 256, 1024, 1536, 7
	readF32 := func(name, sha string, count int) []float32 {
		t.Helper()
		path := filepath.Join("testdata", "customvoice_0b6_ryan_hello", "decoder_initconv", name)
		if err := verifyReleasedFile(path, sha, int64(count*4)); err != nil {
			t.Fatal(err)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		values := make([]float32, count)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return values
	}
	input := readF32("upsample1.full.f32le", "5884389f6891c21e173029dab67d3d138a3dd8dc4b76611cc32b0abdf854f372", inputChannels*length)
	ref := readF32("decoderinit.full.f32le", "2dd3e28511ffb314f05ab258df7ed8b69d8663b29a33f04fe0dc9ddb7c5afc85", outputChannels*length)
	m, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		t.Fatal(err)
	}
	c := m.decoderInit
	if c.inChannels != inputChannels || c.outChannels != outputChannels || c.k != taps || c.dilation != 1 || c.groups() != 1 || len(c.weight) != inputChannels*outputChannels*taps || len(c.bias) != outputChannels {
		t.Fatal("unexpected released decoder-initial convolution geometry")
	}
	cpu, gotLength, err := c.forward(input, length)
	if err != nil || gotLength != length || len(cpu) != len(ref) {
		t.Fatalf("CPU decoder-initial convolution geometry: length=%d output=%d error=%v", gotLength, len(cpu), err)
	}
	for i, want := range ref {
		if cpu[i] != want {
			t.Fatalf("Rust/CPU decoder-initial convolution mismatch at %d: got %.9g want %.9g", i, cpu[i], want)
		}
	}
	if !nvidia.SgemmReady() {
		t.Fatal("opt-in GPU gate requested but NVIDIA SGEMM is unavailable")
	}
	const k = inputChannels * taps
	const block = 512
	var a [k / block][]float32
	var b [k / block][]float32
	for group := range a {
		a[group] = make([]float32, length*block)
		b[group] = make([]float32, block*outputChannels)
	}
	for pos := 0; pos < length; pos++ {
		for ic := 0; ic < inputChannels; ic++ {
			for tap := 0; tap < taps; tap++ {
				src := pos + tap - (taps - 1)
				if src >= 0 {
					index := ic*taps + tap
					a[index/block][pos*block+index%block] = input[ic*length+src]
				}
			}
		}
	}
	for out := 0; out < outputChannels; out++ {
		for in := 0; in < k; in++ {
			b[in/block][(in%block)*outputChannels+out] = c.weight[out*k+in]
		}
	}
	previousStats := nvidia.SetStatsEnabled(true)
	defer nvidia.SetStatsEnabled(previousStats)
	before := nvidia.StatsSnapshot()
	result := make([]float32, length*outputChannels)
	for group := range a {
		partial, err := nvidia.SgemmHost(length, outputChannels, block, 1, a[group], b[group])
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range partial {
			result[i] += v
		}
	}
	after := nvidia.StatsSnapshot()
	if after.KernelLaunches-before.KernelLaunches != uint64(k/block) || after.HostToDeviceBytes-before.HostToDeviceBytes != uint64((length*k+k*outputChannels)*4) || after.DeviceToHostBytes-before.DeviceToHostBytes != uint64(k/block*length*outputChannels*4) || after.Mallocs-before.Mallocs != uint64(3*k/block) || after.Frees-before.Frees != uint64(3*k/block) || after.MallocBytes-before.MallocBytes != after.FreeBytes-before.FreeBytes {
		t.Fatalf("unexpected decoder-initial GPU lifecycle: before=%+v after=%+v", before, after)
	}
	maxRust, maxCPU := float64(0), float64(0)
	for out := 0; out < outputChannels; out++ {
		for pos := 0; pos < length; pos++ {
			value := result[pos*outputChannels+out] + c.bias[out]
			index := out*length + pos
			rustDiff := math.Abs(float64(value) - float64(ref[index]))
			cpuDiff := math.Abs(float64(value) - float64(cpu[index]))
			if math.IsNaN(rustDiff) || math.IsInf(rustDiff, 0) || math.IsNaN(cpuDiff) || math.IsInf(cpuDiff, 0) {
				t.Fatalf("nonfinite decoder-initial GPU difference at %d", index)
			}
			if rustDiff > maxRust {
				maxRust = rustDiff
			}
			if cpuDiff > maxCPU {
				maxCPU = cpuDiff
			}
		}
	}
	t.Logf("256-position 7-tap decoder-initial convolution: GPU/Rust max %.9g GPU/CPU max %.9g; launches=%d H2D=%d D2H=%d alloc/free=%d bytes", maxRust, maxCPU, after.KernelLaunches-before.KernelLaunches, after.HostToDeviceBytes-before.HostToDeviceBytes, after.DeviceToHostBytes-before.DeviceToHostBytes, after.MallocBytes-before.MallocBytes)
	if maxRust > 1e-6 || maxCPU > 1e-6 {
		t.Fatalf("decoder-initial GPU convolution exceeds unchanged 1e-6 diagnostic gate: Rust %.9g CPU %.9g", maxRust, maxCPU)
	}
}
