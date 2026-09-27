package qwen3tts

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// BenchmarkDecoderPreConvReleased compares the existing CPU stage with the
// host-staged three-block GPU diagnostic. Neither includes decoder loading;
// the GPU operation includes host packing, uploads, launches and downloads.
func BenchmarkDecoderPreConvReleased(b *testing.B) {
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
	const frames, inputChannels, outputChannels, taps = 64, 512, 1024, 3
	path := filepath.Join("testdata", "customvoice_0b6_ryan_hello", "decoder_preconv", "quantized.full.f32le")
	if err := verifyReleasedFile(path, "4cce48ec0c48189105f28e24ab6404cadbdec7130f84a3f07c19979989298a9d", inputChannels*frames*4); err != nil {
		b.Fatal(err)
	}
	if err := verifyReleasedFile(filepath.Join("testdata", "customvoice_0b6_ryan_hello", "decoder_preconv", "preconv.full.f32le"), "75c0a7feda1be45023577e1f2675f909b75351fd5bc977978e385597a4703685", outputChannels*frames*4); err != nil {
		b.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float32, inputChannels*frames)
	for i := range input {
		input[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	m, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		b.Fatal(err)
	}
	c := m.preConv
	if c.inChannels != inputChannels || c.outChannels != outputChannels || c.k != taps || c.dilation != 1 || len(c.weight) != inputChannels*outputChannels*taps || len(c.bias) != outputChannels {
		b.Fatal("unexpected decoder pre-convolution geometry")
	}
	if !nvidia.SgemmReady() {
		b.Fatal("NVIDIA SGEMM unavailable")
	}
	const width = 512
	k := inputChannels * taps
	// Keep the CPU operation on its existing production path. GPU timing below
	// includes this lowering and transposition, so it cannot hide preparation.
	gpuOperation := func() ([]float32, error) {
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
		transposed := make([]float32, k*outputChannels)
		for out := 0; out < outputChannels; out++ {
			for in := 0; in < k; in++ {
				transposed[in*outputChannels+out] = c.weight[out*k+in]
			}
		}
		sums := make([]float32, frames*outputChannels)
		for kk := 0; kk < k; kk += width {
			chunkA := make([]float32, frames*width)
			for pos := 0; pos < frames; pos++ {
				copy(chunkA[pos*width:(pos+1)*width], a[pos*k+kk:pos*k+kk+width])
			}
			chunkB := transposed[kk*outputChannels : (kk+width)*outputChannels]
			partial, err := nvidia.SgemmHost(frames, outputChannels, width, 1, chunkA, chunkB)
			if err != nil {
				return nil, err
			}
			for i, v := range partial {
				sums[i] += v
			}
		}
		out := make([]float32, len(sums))
		for channel := 0; channel < outputChannels; channel++ {
			for pos := 0; pos < frames; pos++ {
				out[channel*frames+pos] = sums[pos*outputChannels+channel] + c.bias[channel]
			}
		}
		return out, nil
	}
	cpuWarm, _, err := c.forward(input, frames)
	if err != nil {
		b.Fatal(err)
	}
	gpuWarm, err := gpuOperation()
	if err != nil {
		b.Fatal(err)
	}
	refBytes, err := os.ReadFile(filepath.Join("testdata", "customvoice_0b6_ryan_hello", "decoder_preconv", "preconv.full.f32le"))
	if err != nil {
		b.Fatal(err)
	}
	for i, v := range cpuWarm {
		want := math.Float32frombits(binary.LittleEndian.Uint32(refBytes[i*4:]))
		if v != want || gpuWarm[i] != want {
			b.Fatalf("warm CPU/GPU independent-reference mismatch at %d", i)
		}
	}
	b.Run("CPU", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, _, err := c.forward(input, frames)
			if err != nil || len(out) != len(cpuWarm) {
				b.Fatalf("CPU stage failed: %v", err)
			}
		}
	})
	b.Run("GPUHostStaged", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := gpuOperation()
			if err != nil || len(out) != len(cpuWarm) {
				b.Fatalf("GPU stage failed: %v", err)
			}
		}
	})
}
