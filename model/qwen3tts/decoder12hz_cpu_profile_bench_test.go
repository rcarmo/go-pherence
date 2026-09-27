package qwen3tts

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// BenchmarkDecoderCPUOnlyPinnedProfile isolates one warm decoder call for CPU
// profiling. No GPU setup, kernel load, Talker or CodePredictor is involved.
func BenchmarkDecoderCPUOnlyPinnedProfile(b *testing.B) {
	root := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	oracle := os.Getenv("GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR")
	if root == "" || oracle == "" {
		b.Skip("set model and sentence64 oracle directories for the opt-in decoder profile")
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
	codes := make([]uint32, len(rawCodes)/4)
	for i := range codes {
		codes[i] = binary.LittleEndian.Uint32(rawCodes[4*i:])
	}
	m, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		b.Fatal(err)
	}
	verify := func(wave []float32) {
		b.Helper()
		raw, err := os.ReadFile(wavePath)
		if err != nil {
			b.Fatal(err)
		}
		if len(wave)*4 != len(raw) {
			b.Fatalf("waveform samples=%d want=122880", len(wave))
		}
		maxErr := float64(0)
		for i, v := range wave {
			diff := math.Abs(float64(v) - float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))))
			if math.IsNaN(diff) || math.IsInf(diff, 0) {
				b.Fatalf("nonfinite waveform difference at %d", i)
			}
			maxErr = max(maxErr, diff)
		}
		b.Logf("decoder-only CPU waveform max_abs=%.9g gate=1.6e-6", maxErr)
		if maxErr > 1.6e-6 {
			b.Fatalf("CPU waveform exceeds unchanged gate")
		}
	}
	wave, err := m.decodeCodes(codes, 64)
	if err != nil {
		b.Fatal(err)
	}
	verify(wave)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wave, err = m.decodeCodes(codes, 64)
		if err != nil || len(wave) != 122880 {
			b.Fatalf("CPU decode iteration %d: samples=%d err=%v", i, len(wave), err)
		}
	}
	b.StopTimer()
	verify(wave)
}
