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

// BenchmarkDecoderInitConvResidentProbe isolates one 256-position released
// decoder convolution. Immutable weights and 14 request-local activation,
// weight and output buffer triples are reserved before timing.
// This remains a test-only probe and never changes synthesis dispatch.
func BenchmarkDecoderInitConvResidentProbe(b *testing.B) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		b.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for the decoder-initial GPU benchmark")
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
	const length, inChannels, outChannels, taps, block = 256, 1024, 1536, 7, 512
	const groups = inChannels * taps / block
	traceDir := filepath.Join("testdata", "customvoice_0b6_ryan_hello", "decoder_initconv")
	inputPath := filepath.Join(traceDir, "upsample1.full.f32le")
	refPath := filepath.Join(traceDir, "decoderinit.full.f32le")
	if err := verifyReleasedFile(inputPath, "5884389f6891c21e173029dab67d3d138a3dd8dc4b76611cc32b0abdf854f372", inChannels*length*4); err != nil {
		b.Fatal(err)
	}
	if err := verifyReleasedFile(refPath, "2dd3e28511ffb314f05ab258df7ed8b69d8663b29a33f04fe0dc9ddb7c5afc85", outChannels*length*4); err != nil {
		b.Fatal(err)
	}
	raw, err := os.ReadFile(inputPath)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float32, inChannels*length)
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
	c := m.decoderInit
	if c.inChannels != inChannels || c.outChannels != outChannels || c.k != taps || c.dilation != 1 || c.groups() != 1 || len(c.weight) != inChannels*outChannels*taps || len(c.bias) != outChannels || !nvidia.SgemmReady() {
		b.Fatal("unexpected released convolution geometry or NVIDIA SGEMM unavailable")
	}
	previousStats := nvidia.SetStatsEnabled(true)
	beforeStats := nvidia.StatsSnapshot()
	defer nvidia.SetStatsEnabled(previousStats)
	var devA [groups]*nvidia.Buffer
	var devB [groups]*nvidia.Buffer
	var devC [groups]*nvidia.Buffer
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
		after := nvidia.StatsSnapshot()
		if after.Mallocs-beforeStats.Mallocs != uint64(groups*3) || after.Frees-beforeStats.Frees != uint64(groups*3) || after.MallocBytes-beforeStats.MallocBytes != after.FreeBytes-beforeStats.FreeBytes {
			b.Errorf("decoder-initial GPU allocation/free mismatch: before=%+v after=%+v", beforeStats, after)
		}
		b.Logf("resident decoder-initial setup alloc/free=%d bytes, weight uploads=%d bytes", after.MallocBytes-beforeStats.MallocBytes, uint64(groups*block*outChannels*4))
	}()
	setupStarted := time.Now()
	const k = inChannels * taps
	for group := 0; group < groups; group++ {
		devA[group], err = nvidia.Malloc(length * block)
		if err != nil {
			b.Fatal(err)
		}
		devB[group], err = nvidia.Malloc(block * outChannels)
		if err != nil {
			b.Fatal(err)
		}
		devC[group], err = nvidia.Malloc(length * outChannels)
		if err != nil {
			b.Fatal(err)
		}
		weights := make([]float32, block*outChannels)
		for out := 0; out < outChannels; out++ {
			for in := 0; in < block; in++ {
				weights[in*outChannels+out] = c.weight[out*k+group*block+in]
			}
		}
		if err := devB[group].Upload(weights); err != nil {
			b.Fatal(err)
		}
	}
	b.Logf("resident decoder-initial allocation and weight packing/upload: %s (excludes model and kernel load)", time.Since(setupStarted))
	var a [groups][]float32
	var partial [groups][]float32
	for group := range a {
		a[group] = make([]float32, length*block)
		partial[group] = make([]float32, length*outChannels)
	}
	result := make([]float32, length*outChannels)
	outTime := make([]float32, length*outChannels)
	operation := func() error {
		for group := range a {
			clear(a[group])
		}
		for pos := 0; pos < length; pos++ {
			for ic := 0; ic < inChannels; ic++ {
				for tap := 0; tap < taps; tap++ {
					src := pos + tap - (taps - 1)
					if src >= 0 {
						index := ic*taps + tap
						a[index/block][pos*block+index%block] = input[ic*length+src]
					}
				}
			}
		}
		for group := range a {
			if err := devA[group].Upload(a[group]); err != nil {
				return err
			}
			if err := nvidia.Sgemm(length, outChannels, block, 1, devA[group], devB[group], devC[group]); err != nil {
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
		for out := 0; out < outChannels; out++ {
			for pos := 0; pos < length; pos++ {
				outTime[out*length+pos] = result[pos*outChannels+out] + c.bias[out]
			}
		}
		return nil
	}
	check := func() {
		b.Helper()
		for i, value := range outTime {
			if math.Float32bits(value) != binary.LittleEndian.Uint32(refBytes[i*4:]) {
				b.Fatalf("resident decoder-initial output differs from pinned Rust at %d", i)
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
			out, gotLength, err := c.forward(input, length)
			if err != nil || gotLength != length || len(out) != len(outTime) {
				b.Fatalf("CPU decoder-initial convolution failed: %v", err)
			}
		}
	})
	b.Run("GPUResidentOwned", func(b *testing.B) {
		b.ReportAllocs()
		before := nvidia.StatsSnapshot()
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
		after := nvidia.StatsSnapshot()
		if after.KernelLaunches-before.KernelLaunches != uint64(groups*b.N) || after.HostToDeviceBytes-before.HostToDeviceBytes != uint64(groups*length*block*4*b.N) || after.DeviceToHostBytes-before.DeviceToHostBytes != uint64(groups*length*outChannels*4*b.N) || after.Mallocs != before.Mallocs || after.Frees != before.Frees {
			b.Fatalf("unexpected per-call GPU lifecycle: before=%+v after=%+v calls=%d", before, after, b.N)
		}
	})
}
