package qwen3tts

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// BenchmarkDecoderPreConvResidentProbe reserves one request's three F32
// weight/output pairs on the GPU. Each call still does host im2col, uploads
// activations, downloads partial outputs, and applies the bias on the CPU.
// This is a benchmark-only trial, never a production decoder dispatch.
func BenchmarkDecoderPreConvResidentProbe(b *testing.B) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		b.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for the released GPU benchmark")
	}
	root := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	if root == "" {
		b.Fatal("set GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	}
	b.StopTimer()
	for name, sha := range map[string]string{
		"speech_tokenizer/config.json":       "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
		"speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
	} {
		if err := verifyReleasedFile(filepath.Join(root, name), sha, 0); err != nil {
			b.Fatal(err)
		}
	}
	const frames, inputChannels, outputChannels, taps, block = 64, 512, 1024, 3, 512
	traceDir := filepath.Join("testdata", "customvoice_0b6_ryan_hello", "decoder_preconv")
	inputPath := filepath.Join(traceDir, "quantized.full.f32le")
	refPath := filepath.Join(traceDir, "preconv.full.f32le")
	if err := verifyReleasedFile(inputPath, "4cce48ec0c48189105f28e24ab6404cadbdec7130f84a3f07c19979989298a9d", inputChannels*frames*4); err != nil {
		b.Fatal(err)
	}
	if err := verifyReleasedFile(refPath, "75c0a7feda1be45023577e1f2675f909b75351fd5bc977978e385597a4703685", outputChannels*frames*4); err != nil {
		b.Fatal(err)
	}
	raw, err := os.ReadFile(inputPath)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float32, inputChannels*frames)
	for i := range input {
		input[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	refBytes, err := os.ReadFile(refPath)
	if err != nil {
		b.Fatal(err)
	}
	m, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		b.Fatal(err)
	}
	c := m.preConv
	if c.inChannels != inputChannels || c.outChannels != outputChannels || c.k != taps || c.dilation != 1 || len(c.weight) != inputChannels*outputChannels*taps || len(c.bias) != outputChannels {
		b.Fatal("unexpected released decoder pre-convolution geometry")
	}
	if !nvidia.SgemmReady() {
		b.Fatal("NVIDIA SGEMM unavailable")
	}
	previousStats := nvidia.SetStatsEnabled(true)
	beforeStats := nvidia.StatsSnapshot()
	defer nvidia.SetStatsEnabled(previousStats)
	var devA [3]*nvidia.Buffer
	var devB [3]*nvidia.Buffer
	var devC [3]*nvidia.Buffer
	defer func() {
		for i := range devA {
			if devA[i] != nil {
				devA[i].Free()
			}
			if devB[i] != nil {
				devB[i].Free()
			}
			if devC[i] != nil {
				devC[i].Free()
			}
		}
		afterStats := nvidia.StatsSnapshot()
		if afterStats.Mallocs-beforeStats.Mallocs != 9 || afterStats.Frees-beforeStats.Frees != 9 || afterStats.MallocBytes-beforeStats.MallocBytes != afterStats.FreeBytes-beforeStats.FreeBytes {
			b.Errorf("resident GPU allocation/free imbalance: before=%+v after=%+v", beforeStats, afterStats)
		}
		b.Logf("resident setup/lifecycle: alloc/free=%d bytes; 3 weight uploads=%d bytes (timed calls upload activations and download partials)", afterStats.MallocBytes-beforeStats.MallocBytes, uint64(3*block*outputChannels*4))
	}()
	k := inputChannels * taps
	setupStarted := time.Now()
	for group := 0; group < 3; group++ {
		devA[group], err = nvidia.Malloc(frames * block)
		if err != nil {
			b.Fatal(err)
		}
		devB[group], err = nvidia.Malloc(block * outputChannels)
		if err != nil {
			b.Fatal(err)
		}
		devC[group], err = nvidia.Malloc(frames * outputChannels)
		if err != nil {
			b.Fatal(err)
		}
		weights := make([]float32, block*outputChannels)
		for out := 0; out < outputChannels; out++ {
			for in := 0; in < block; in++ {
				weights[in*outputChannels+out] = c.weight[out*k+group*block+in]
			}
		}
		if err := devB[group].Upload(weights); err != nil {
			b.Fatal(err)
		}
	}
	b.Logf("resident GPU buffer allocation and weight packing/upload: %s (excludes model load and kernel initialization)", time.Since(setupStarted))
	var a [3][]float32
	var partial [3][]float32
	for group := range a {
		a[group] = make([]float32, frames*block)
		partial[group] = make([]float32, frames*outputChannels)
	}
	result := make([]float32, frames*outputChannels)
	outTime := make([]float32, frames*outputChannels)
	operation := func() error {
		for group := range a {
			clear(a[group])
		}
		for pos := 0; pos < frames; pos++ {
			for ic := 0; ic < inputChannels; ic++ {
				for tap := 0; tap < taps; tap++ {
					src := pos + tap - (taps - 1)
					if src >= 0 {
						index := ic*taps + tap
						a[index/block][pos*block+index%block] = input[ic*frames+src]
					}
				}
			}
		}
		for group := range a {
			if err := devA[group].Upload(a[group]); err != nil {
				return err
			}
			if err := nvidia.Sgemm(frames, outputChannels, block, 1, devA[group], devB[group], devC[group]); err != nil {
				return err
			}
			nvidia.Sync()
			if err := devC[group].Download(partial[group]); err != nil {
				return err
			}
		}
		clear(result)
		for group := range partial {
			for i, v := range partial[group] {
				result[i] += v
			}
		}
		for channel := 0; channel < outputChannels; channel++ {
			for pos := 0; pos < frames; pos++ {
				outTime[channel*frames+pos] = result[pos*outputChannels+channel] + c.bias[channel]
			}
		}
		return nil
	}
	check := func() {
		b.Helper()
		for out := 0; out < outputChannels; out++ {
			for pos := 0; pos < frames; pos++ {
				index := out*frames + pos
				if math.Float32bits(outTime[index]) != binary.LittleEndian.Uint32(refBytes[index*4:]) {
					b.Fatalf("resident GPU pre-convolution differs from pinned Rust output at %d", index)
				}
			}
		}
	}
	if err := operation(); err != nil {
		b.Fatal(err)
	}
	check()
	b.Run("CPU", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, length, err := c.forward(input, frames)
			if err != nil || length != frames || len(out) != len(outTime) {
				b.Fatalf("CPU pre-convolution failed: %v", err)
			}
		}
	})
	b.Run("GPUResident", func(b *testing.B) {
		b.ReportAllocs()
		before := nvidia.StatsSnapshot()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := operation(); err != nil {
				b.Fatal(err)
			}
		}
		b.StopTimer()
		check()
		after := nvidia.StatsSnapshot()
		if after.KernelLaunches-before.KernelLaunches != uint64(3*b.N) || after.HostToDeviceBytes-before.HostToDeviceBytes != uint64(3*frames*block*4*b.N) || after.DeviceToHostBytes-before.DeviceToHostBytes != uint64(3*frames*outputChannels*4*b.N) || after.Mallocs != before.Mallocs || after.Frees != before.Frees {
			b.Fatalf("unexpected resident GPU per-call lifecycle: before=%+v after=%+v n=%d", before, after, b.N)
		}
	})
	b.Run("GPUResidentOwned", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := operation(); err != nil {
				b.Fatal(err)
			}
			owned := append([]float32(nil), outTime...)
			if len(owned) != len(outTime) {
				b.Fatal("GPU owned output size mismatch")
			}
		}
		b.StopTimer()
		check()
	})
}
