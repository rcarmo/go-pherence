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

// BenchmarkDecoderThreeGPUConvsProbe compares decoder-only warm calls on one
// pinned 64-frame input. GPU setup and CPU/GPU waveform checks are untimed;
// each timed hybrid call includes host packing, transfers and GPU execution.
func BenchmarkDecoderThreeGPUConvsProbe(b *testing.B) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		b.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for the opt-in GPU benchmark")
	}
	root := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	oracle := os.Getenv("GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR")
	if root == "" || oracle == "" {
		b.Fatal("set model and sentence64 oracle directories")
	}
	if !nvidia.SgemmReady() {
		b.Fatal("opt-in GPU benchmark requested but NVIDIA SGEMM unavailable")
	}
	for name, sha := range map[string]string{
		"speech_tokenizer/config.json":       "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
		"speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
	} {
		if err := verifyReleasedFile(filepath.Join(root, name), sha, 0); err != nil {
			b.Fatal(err)
		}
	}
	codesPath := filepath.Join("testdata", "customvoice_0b6_ryan_hello", "sentence_64_codes.u32le")
	if err := verifyReleasedFile(codesPath, "fe1f4c0dd8f99eb2dfeb1f9528a015142fdac186f0bd1237eaf0506d41a003cf", 64*16*4); err != nil {
		b.Fatal(err)
	}
	wavePath := filepath.Join(oracle, "sentence_64_waveform.f32le")
	if err := verifyReleasedFile(wavePath, sentence64WaveSHA, 122880*4); err != nil {
		b.Fatal(err)
	}
	rawCodes, err := os.ReadFile(codesPath)
	if err != nil {
		b.Fatal(err)
	}
	rawWave, err := os.ReadFile(wavePath)
	if err != nil {
		b.Fatal(err)
	}
	codes := make([]uint32, len(rawCodes)/4)
	for i := range codes {
		codes[i] = binary.LittleEndian.Uint32(rawCodes[4*i:])
	}
	m, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		b.Fatal(err)
	}
	cpuWave, err := m.decodeCodes(codes, 64)
	if err != nil {
		b.Fatal(err)
	}
	verifyWave := func(label string, wave []float32) {
		b.Helper()
		if len(wave) != len(rawWave)/4 {
			b.Fatalf("%s waveform length=%d", label, len(wave))
		}
		maxErr := float64(0)
		for i, v := range wave {
			ref := math.Float32frombits(binary.LittleEndian.Uint32(rawWave[4*i:]))
			diff := math.Abs(float64(v) - float64(ref))
			if math.IsNaN(diff) || math.IsInf(diff, 0) {
				b.Fatalf("%s nonfinite waveform difference at %d", label, i)
			}
			maxErr = max(maxErr, diff)
		}
		b.Logf("%s waveform max_abs=%.9g gate=1.6e-6", label, maxErr)
		if maxErr > 1.6e-6 {
			b.Fatalf("%s waveform exceeds unchanged gate", label)
		}
	}
	verifyWave("CPU", cpuWave)
	setupStart := time.Now()
	pre := newTestDecoderGPUConv(b, m.preConv, 64, "preconv", "quantized.full.f32le", "preconv.full.f32le")
	init := newTestDecoderGPUConv(b, m.decoderInit, 256, "initconv", "upsample1.full.f32le", "decoderinit.full.f32le")
	final := newLiveFinalConvGPU(b, m.finalConv)
	b.Logf("GPU buffer/weight setup excluding model and kernel load: %s", time.Since(setupStart))
	gpuDecode := func() ([]float32, error) {
		return m.decodeCodesWithDiagnosticConvs(codes, 64, pre.forward, init.forward, final.forward)
	}
	gpuWave, err := gpuDecode()
	if err != nil {
		b.Fatal(err)
	}
	verifyWave("GPU hybrid", gpuWave)
	maxPeer := float64(0)
	for i, v := range gpuWave {
		maxPeer = max(maxPeer, math.Abs(float64(v)-float64(cpuWave[i])))
	}
	b.Logf("GPU/CPU decoded waveform max_abs=%.9g", maxPeer)
	if maxPeer > 1e-6 {
		b.Fatalf("GPU/CPU decoded waveform exceeds 1e-6")
	}
	for _, item := range []struct {
		name string
		fn   func() ([]float32, error)
	}{
		{"CPUDecoder", func() ([]float32, error) { return m.decodeCodes(codes, 64) }},
		{"GPUThreeConvsDecoder", gpuDecode},
	} {
		b.Run(item.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				wave, err := item.fn()
				if err != nil || len(wave) != 122880 {
					b.Fatalf("%s iteration %d: samples=%d err=%v", item.name, i, len(wave), err)
				}
			}
			b.StopTimer()
		})
	}
	pre.close()
	init.close()
	final.close()
}
