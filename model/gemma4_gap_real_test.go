package model

import (
	"os"
	"runtime/pprof"
	"testing"
	"time"
)

func TestGemma4RealCPUGap124x48(t *testing.T) {
	if os.Getenv("GO_PHERENCE_GEMMA4_GAP_REAL") == "" {
		t.Skip("set GO_PHERENCE_GEMMA4_GAP_REAL=1 for the 124-prefill/48-decode CPU gap gate")
	}
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = "../models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf"
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: 48})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	prompt := make([]int, 123) // Gemma4 preparation adds BOS, yielding 124 tokens.
	for i := range prompt {
		prompt[i] = 10979
	}
	var prefillProfile *os.File
	if profilePath := os.Getenv("GO_PHERENCE_GEMMA4_PREFILL_CPU_PROFILE"); profilePath != "" {
		prefillProfile, err = os.Create(profilePath)
		if err != nil {
			t.Fatal(err)
		}
		if err := pprof.StartCPUProfile(prefillProfile); err != nil {
			prefillProfile.Close()
			t.Fatal(err)
		}
	}
	prefillStart := time.Now()
	prefill, err := s.PrefillChunk(prompt)
	if err != nil {
		t.Fatal(err)
	}
	prefillElapsed := time.Since(prefillStart)
	if prefillProfile != nil {
		pprof.StopCPUProfile()
		if err := prefillProfile.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if !prefill.ReadyToDecode || prefill.Position != 124 {
		t.Fatalf("unexpected prefill result %+v", prefill)
	}
	if os.Getenv("GO_PHERENCE_GEMMA4_PREFILL_ONLY") != "" {
		prefillRate := 124 / prefillElapsed.Seconds()
		const oraclePrefill = 447.166367
		t.Logf("gemma4_gap prefill_tokens=124 prefill=%s prefill_tok_s=%.6f prefill_efficiency=%.4f", prefillElapsed, prefillRate, prefillRate/oraclePrefill)
		return
	}
	var decodeProfile *os.File
	if profilePath := os.Getenv("GO_PHERENCE_GEMMA4_DECODE_CPU_PROFILE"); profilePath != "" {
		decodeProfile, err = os.Create(profilePath)
		if err != nil {
			t.Fatal(err)
		}
		if err := pprof.StartCPUProfile(decodeProfile); err != nil {
			decodeProfile.Close()
			t.Fatal(err)
		}
	}
	decodeStart := time.Now()
	decoded := 0
	for decoded < 48 {
		step, err := s.DecodeStep()
		if err != nil {
			t.Fatal(err)
		}
		decoded++
		if step.Finished && decoded != 48 {
			t.Fatalf("decode finished after %d tokens", decoded)
		}
	}
	decodeElapsed := time.Since(decodeStart)
	if decodeProfile != nil {
		pprof.StopCPUProfile()
		if err := decodeProfile.Close(); err != nil {
			t.Fatal(err)
		}
	}
	prefillRate := 124 / prefillElapsed.Seconds()
	decodeRate := 48 / decodeElapsed.Seconds()
	const oraclePrefill = 447.166367
	const oracleDecode = 9.309777
	t.Logf("gemma4_gap prefill_tokens=124 prefill=%s prefill_tok_s=%.6f prefill_efficiency=%.4f decode_tokens=48 decode=%s decode_tok_s=%.6f decode_efficiency=%.4f", prefillElapsed, prefillRate, prefillRate/oraclePrefill, decodeElapsed, decodeRate, decodeRate/oracleDecode)
}
