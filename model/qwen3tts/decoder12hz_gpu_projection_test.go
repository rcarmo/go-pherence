package qwen3tts

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// This is a decoder-stage diagnostic, not a GPU synthesis path. The Rust traces
// are 2048 evenly spaced F32 samples of the 64-frame Decoder12Hz tensors.
func TestDecoderInputProjectionPinnedGPU(t *testing.T) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for the opt-in NVIDIA decoder projection gate")
	}
	root := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	if root == "" {
		t.Fatal("set GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	}
	traceDir := filepath.Join("testdata", "customvoice_0b6_ryan_hello", "decoder_projection")
	for path, sha := range map[string]string{
		"speech_tokenizer/config.json":       "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
		"speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
	} {
		if err := verifyReleasedFile(filepath.Join(root, path), sha, 0); err != nil {
			t.Fatal(err)
		}
	}
	_, codeBytes := sentence64Fixture(t)
	const frames, latent, hidden = 64, 1024, 512
	readTrace := func(name, hash string) []float32 {
		t.Helper()
		path := filepath.Join(traceDir, name+".f32le")
		if err := verifyReleasedFile(path, hash, 2048*4); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, 2048)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return out
	}
	preconvRef := readTrace("preconv", "62d2766cb5f9bae22af5783268953e4b5d824ed54c885dc775157935a7b433cf")
	projectionRef := readTrace("inputproj", "65ab117cc051fc6bf0817590a75d605acf86d2fe02a854944c10caf209b2979a")
	m, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		t.Fatal(err)
	}
	if m.cfg.CodebookDim != 256 || m.cfg.LatentDim != latent || m.cfg.HiddenSize != hidden || m.cfg.Quantizers != 16 || len(m.inputProjection.bias) != hidden {
		t.Fatal("unexpected released decoder topology")
	}
	codes := make([]uint32, len(codeBytes)/4)
	for i := range codes {
		codes[i] = binary.LittleEndian.Uint32(codeBytes[i*4:])
	}
	c := m.cfg.CodebookDim
	quantized := make([]float32, frames*2*c)
	first, rest := make([]float32, c), make([]float32, c)
	firstProjected, restProjected := make([]float32, 2*c), make([]float32, 2*c)
	for frame := 0; frame < frames; frame++ {
		copy(first, m.codebooks[0][int(codes[frame*16])*c:][:c])
		clear(rest)
		for group := 1; group < 16; group++ {
			row := m.codebooks[group][int(codes[frame*16+group])*c:][:c]
			for i, v := range row {
				rest[i] += v
			}
		}
		if !decoderProjectionRows(firstProjected, first, m.firstProjection, 2*c, c) || !decoderProjectionRows(restProjected, rest, m.restProjection, 2*c, c) {
			t.Fatal("codebook projection failed")
		}
		for channel := 0; channel < 2*c; channel++ {
			quantized[channel*frames+frame] = firstProjected[channel] + restProjected[channel]
		}
	}
	preconv, length, err := m.preConv.forward(quantized, frames)
	if err != nil || length != frames || len(preconv) != frames*latent {
		t.Fatalf("pre-convolution shape: %d %d %v", length, len(preconv), err)
	}
	checkSamples := func(label string, got, ref []float32, maxAllowed float64) float64 {
		t.Helper()
		maxErr := float64(0)
		for i, want := range ref {
			index := i * (len(got) - 1) / (len(ref) - 1)
			diff := math.Abs(float64(got[index]) - float64(want))
			if math.IsNaN(diff) || math.IsInf(diff, 0) {
				t.Fatalf("%s nonfinite difference at sample %d", label, i)
			}
			if diff > maxErr {
				maxErr = diff
			}
		}
		if maxErr > maxAllowed {
			t.Fatalf("%s max error %.9g > %.9g", label, maxErr, maxAllowed)
		}
		return maxErr
	}
	preErr := checkSamples("Rust pre-convolution", preconv, preconvRef, 0)
	input := channelToTime(preconv, latent, frames)
	cpu := make([]float32, frames*hidden)
	for row := 0; row < frames; row++ {
		if err := decoderLinearForward(m.inputProjection, cpu[row*hidden:(row+1)*hidden], input[row*latent:(row+1)*latent]); err != nil {
			t.Fatal(err)
		}
	}
	cpuErr := checkSamples("Rust CPU projection", cpu, projectionRef, 0)
	if !nvidia.SgemmReady() {
		t.Fatal("opt-in GPU gate requested but NVIDIA SGEMM is unavailable")
	}
	// The decoder owns row-major [out,in] weights; SGEMM expects [in,out].
	transposed := make([]float32, latent*hidden)
	for out := 0; out < hidden; out++ {
		for in := 0; in < latent; in++ {
			transposed[in*hidden+out] = m.inputProjection.weight[out*latent+in]
		}
	}
	previousStats := nvidia.SetStatsEnabled(true)
	defer nvidia.SetStatsEnabled(previousStats)
	before := nvidia.StatsSnapshot()
	gpu, err := nvidia.SgemmHost(frames, hidden, latent, 1, input, transposed)
	after := nvidia.StatsSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if after.Mallocs-before.Mallocs != 3 || after.Frees-before.Frees != 3 || after.MallocBytes-before.MallocBytes != after.FreeBytes-before.FreeBytes {
		t.Fatalf("GPU allocation/free imbalance: before=%+v after=%+v", before, after)
	}
	t.Logf("GPU diagnostic: launches=%d H2D=%d bytes D2H=%d bytes alloc/free=%d bytes", after.KernelLaunches-before.KernelLaunches, after.HostToDeviceBytes-before.HostToDeviceBytes, after.DeviceToHostBytes-before.DeviceToHostBytes, after.MallocBytes-before.MallocBytes)
	for row := 0; row < frames; row++ {
		for out, bias := range m.inputProjection.bias {
			gpu[row*hidden+out] += bias
		}
	}
	gpuErr := checkSamples("Rust GPU projection", gpu, projectionRef, 1e-6)
	maxCPU := float64(0)
	for i, v := range gpu {
		diff := math.Abs(float64(v) - float64(cpu[i]))
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			t.Fatalf("nonfinite GPU/CPU difference at %d", i)
		}
		if diff > maxCPU {
			maxCPU = diff
		}
	}
	if maxCPU > 1e-6 {
		t.Fatalf("GPU/CPU projection max error %.9g > 1e-6", maxCPU)
	}
	t.Logf("64x1024 input projection -> 64x512: Rust sampled preconv max=%.9g CPU max=%.9g GPU max=%.9g; full GPU/CPU max=%.9g", preErr, cpuErr, gpuErr, maxCPU)
}
