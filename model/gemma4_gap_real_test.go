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
	prefillSetupStart := time.Now()
	if err := s.BeginPrefill(prompt); err != nil {
		t.Fatal(err)
	}
	prefillSetupElapsed := time.Since(prefillSetupStart)
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
	prefill, err := s.PrefillNext(124)
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
	const oraclePrefill = 91.229561 // CPU-only b607 exact-prompt F32-KV median over six paired phase samples.
	prefillRate := 124 / prefillElapsed.Seconds()
	prefillTotalRate := 124 / (prefillSetupElapsed + prefillElapsed).Seconds()
	if os.Getenv("GO_PHERENCE_GEMMA4_PREFILL_ONLY") != "" {
		t.Logf("gemma4_gap prefill_tokens=124 prefill_setup=%s prefill_compute=%s prefill_compute_tok_s=%.6f prefill_total_tok_s=%.6f prefill_efficiency=%.4f", prefillSetupElapsed, prefillElapsed, prefillRate, prefillTotalRate, prefillRate/oraclePrefill)
		return
	}
	// Prefill already produced the first output token's boundary logits. Consume
	// that result outside the generation-evaluation phase, then time the 47 model
	// evaluations needed to emit the remaining 47 outputs.
	first, err := s.DecodeStep()
	if err != nil {
		t.Fatal(err)
	}
	if first.Finished {
		t.Fatal("decode finished at the prefill boundary")
	}
	generated := make([]int, 0, 48)
	generated = append(generated, first.Token)
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
	finalFinished := false
	for evaluated := 0; evaluated < 47; evaluated++ {
		step, err := s.DecodeStep()
		if err != nil {
			t.Fatal(err)
		}
		generated = append(generated, step.Token)
		finalFinished = step.Finished
		if step.Finished && evaluated != 46 {
			t.Fatalf("decode finished after %d evaluated tokens", evaluated+1)
		}
	}
	decodeElapsed := time.Since(decodeStart)
	if decodeProfile != nil {
		pprof.StopCPUProfile()
		if err := decodeProfile.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if !finalFinished {
		t.Fatal("decode did not finish after the 47th evaluated token")
	}
	wantGenerated := [...]int{
		70990, 4941, 159371, 1390, 67109, 1485, 237064, 9120,
		70846, 9120, 13063, 1390, 236743, 40594, 1390, 238007,
		1390, 238122, 4829, 1390, 1390, 237307, 1390, 113858,
		9222, 1390, 9222, 246293, 9222, 1390, 1390, 9222,
		783, 82106, 9222, 246293, 236929, 4862, 237275, 1390,
		237083, 237249, 244359, 238280, 236929, 395, 236782, 236929,
	}
	for i, want := range wantGenerated {
		if generated[i] != want {
			t.Fatalf("generated token %d=%d, want frozen exact token %d", i, generated[i], want)
		}
	}
	t.Logf("gemma4_gap generated_tokens=%v", generated)
	decodeRate := 47 / decodeElapsed.Seconds()
	const oracleDecode = 10.526478 // CPU-only b607 exact Go-trajectory F32-KV median over six paired phase samples.
	t.Logf("gemma4_gap prefill_tokens=124 prefill_setup=%s prefill_compute=%s prefill_compute_tok_s=%.6f prefill_total_tok_s=%.6f prefill_efficiency=%.4f output_tokens=48 decode_evals=47 decode=%s decode_eval_tok_s=%.6f decode_efficiency=%.4f", prefillSetupElapsed, prefillElapsed, prefillRate, prefillTotalRate, prefillRate/oraclePrefill, decodeElapsed, decodeRate, decodeRate/oracleDecode)
}
