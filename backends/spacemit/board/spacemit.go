package board

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// SpacemiTBackend uses SpacemiT's llama.cpp fork for GGUF inference and
// falls back to the SIMD path for individual tensor ops (since the ORT EP
// targets ONNX models, not raw tensor dispatch).
//
// For op-level calls (GemvF32, RMSNormF32, …) the SpacemiT backend delegates
// to the SIMDBackend; those paths already benefit from the K3-tuned RVV/A100
// code path because spacemit-llama.cpp is compiled with the same ISA flags.
//
// Use RunGGUF for full-model benchmarking via the vendor llama-bench binary.
type SpacemiTBackend struct {
	simd SIMDBackend
}

func (s SpacemiTBackend) Name() string { return TierSpacemiT.String() }

// Op-level calls delegate to SIMD; the SpacemiT ORT EP doesn't expose
// a C tensor-dispatch surface to Go without CGo.
func (s SpacemiTBackend) GemvF32(out, x, w []float32, inDim, outDim int) error {
	return s.simd.GemvF32(out, x, w, inDim, outDim)
}
func (s SpacemiTBackend) RMSNormF32(x, w []float32, eps float32) error {
	return s.simd.RMSNormF32(x, w, eps)
}
func (s SpacemiTBackend) RMSNormNoScaleF32(x []float32, eps float32) error {
	return s.simd.RMSNormNoScaleF32(x, eps)
}
func (s SpacemiTBackend) SiLUMulF32(dst, gate, up []float32) error {
	return s.simd.SiLUMulF32(dst, gate, up)
}
func (s SpacemiTBackend) GELUTanhMulF32(dst, gate, up []float32) error {
	return s.simd.GELUTanhMulF32(dst, gate, up)
}
func (s SpacemiTBackend) RoPEPartialF32(x, freqs []float32, pos, nHeads, headDim, rotHalf int) error {
	return s.simd.RoPEPartialF32(x, freqs, pos, nHeads, headDim, rotHalf)
}
func (s SpacemiTBackend) AttentionScoresF32(out, q, kCache []float32, seqLen, nHeads, nKVHeads, headDim int, scale float32) error {
	return s.simd.AttentionScoresF32(out, q, kCache, seqLen, nHeads, nKVHeads, headDim, scale)
}

// GGUFBenchResult holds the output of a llama-bench run.
type GGUFBenchResult struct {
	Model    string
	Backend  string
	PP       float64 // prompt-processing tokens/s
	TG       float64 // token-generation tokens/s
	RawLines []string
}

// RunGGUF runs llama-bench on the given GGUF file using the SpacemiT binary.
// ppTokens = prompt tokens to benchmark; tgTokens = generation tokens.
// Timeout limits the whole run (default 120s if zero).
func RunGGUF(modelPath string, threads, ppTokens, tgTokens int, timeout time.Duration) (*GGUFBenchResult, error) {
	if timeout == 0 {
		timeout = 120 * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	args := []string{
		"-m", modelPath,
		"-t", strconv.Itoa(threads),
		"-p", strconv.Itoa(ppTokens),
		"-n", strconv.Itoa(tgTokens),
		"--output", "json",
	}
	cmd := exec.CommandContext(ctx, "llama-bench", args...)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("llama-bench: %w\noutput: %s", err, out.String())
	}

	res := &GGUFBenchResult{
		Model:   modelPath,
		Backend: TierSpacemiT.String(),
	}
	res.RawLines = strings.Split(strings.TrimSpace(out.String()), "\n")

	// llama-bench --output json emits one JSON object per test line.
	for _, line := range res.RawLines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if v, ok := row["pp"].(float64); ok && v > 0 {
			res.PP = v
		}
		if v, ok := row["tg"].(float64); ok && v > 0 {
			res.TG = v
		}
	}
	return res, nil
}

// RVVCaps returns a compact string describing the RVV / A100 runtime info
// as reported by spacemit-tcm-smi, plus the simd capabilities.
func RVVCaps() string {
	caps := simd.RuntimeCapabilities()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "spacemit-tcm-smi").Output()
	simdLine := fmt.Sprintf("simd: dot=%v sgem=%v", caps.HasDot, caps.HasSGEMM)
	if err != nil {
		return fmt.Sprintf("%s\nspacemit-tcm-smi: %v", simdLine, err)
	}

	var keep []string
	keep = append(keep, simdLine)
	for _, ln := range strings.Split(string(out), "\n") {
		l := strings.ToLower(ln)
		if strings.Contains(l, "tcm") || strings.Contains(l, "ime") ||
			strings.Contains(l, "core") || strings.Contains(l, "cpu_mask") ||
			strings.Contains(l, "perfer") || strings.Contains(l, "block") {
			keep = append(keep, strings.TrimSpace(ln))
		}
	}
	return strings.Join(keep, "\n")
}
