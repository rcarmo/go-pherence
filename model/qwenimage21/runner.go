package qwenimage21

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type Assets struct{ DiffusionModel, VAE, LLM string }
type GenerateOptions struct {
	Prompt, Output                string
	Width, Height, Steps, Threads int
	Guidance                      float32
	Seed                          int64
	Timeout                       time.Duration
}
type GenerateResult struct {
	Output  string        `json:"output"`
	SHA256  string        `json:"sha256"`
	Bytes   int64         `json:"bytes"`
	Elapsed time.Duration `json:"elapsed"`
	Log     string        `json:"log"`
}
type Runner struct {
	Executable string
	Assets     Assets
}

func (r Runner) Validate() error {
	if r.Executable == "" {
		return fmt.Errorf("qwen-image-2.1: missing stable-diffusion.cpp executable")
	}
	for name, p := range map[string]string{"executable": r.Executable, "diffusion": r.Assets.DiffusionModel, "vae": r.Assets.VAE, "llm": r.Assets.LLM} {
		st, e := os.Stat(p)
		if e != nil {
			return fmt.Errorf("qwen-image-2.1: %s: %w", name, e)
		}
		if !st.Mode().IsRegular() {
			return fmt.Errorf("qwen-image-2.1: %s is not a regular file", name)
		}
	}
	return nil
}
func (o GenerateOptions) validate() error {
	if strings.TrimSpace(o.Prompt) == "" || o.Output == "" {
		return fmt.Errorf("qwen-image-2.1: prompt/output required")
	}
	if o.Width <= 0 || o.Height <= 0 || o.Width%32 != 0 || o.Height%32 != 0 {
		return fmt.Errorf("qwen-image-2.1: dimensions must be positive multiples of 32")
	}
	if o.Steps < 1 || o.Steps > 1000 {
		return fmt.Errorf("qwen-image-2.1: steps must be 1..1000")
	}
	if o.Guidance < 0 {
		return fmt.Errorf("qwen-image-2.1: guidance must be nonnegative")
	}
	return nil
}
func (r Runner) Generate(ctx context.Context, o GenerateOptions) (GenerateResult, error) {
	if e := r.Validate(); e != nil {
		return GenerateResult{}, e
	}
	if e := o.validate(); e != nil {
		return GenerateResult{}, e
	}
	if o.Threads <= 0 {
		o.Threads = 1
	}
	if o.Timeout <= 0 {
		o.Timeout = 30 * time.Minute
	}
	abs, e := filepath.Abs(o.Output)
	if e != nil {
		return GenerateResult{}, e
	}
	if e = os.MkdirAll(filepath.Dir(abs), 0700); e != nil {
		return GenerateResult{}, e
	}
	owned, cancel := context.WithTimeout(ctx, o.Timeout)
	defer cancel()
	args := []string{"--diffusion-model", r.Assets.DiffusionModel, "--vae", r.Assets.VAE, "--llm", r.Assets.LLM, "-p", o.Prompt, "--cfg-scale", strconv.FormatFloat(float64(o.Guidance), 'g', -1, 32), "--sampling-method", "euler", "--steps", strconv.Itoa(o.Steps), "--width", strconv.Itoa(o.Width), "--height", strconv.Itoa(o.Height), "--threads", strconv.Itoa(o.Threads), "--seed", strconv.FormatInt(o.Seed, 10), "-o", abs}
	args = append(args, "--offload-to-cpu")
	cmd := exec.CommandContext(owned, r.Executable, args...)
	cmd.WaitDelay = 2 * time.Second
	cmd.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES=")
	var log bytes.Buffer
	cmd.Stdout = &log
	cmd.Stderr = &log
	start := time.Now()
	e = cmd.Run()
	elapsed := time.Since(start)
	if e != nil {
		if owned.Err() != nil {
			return GenerateResult{}, fmt.Errorf("qwen-image-2.1: generation timeout: %w", owned.Err())
		}
		return GenerateResult{}, fmt.Errorf("qwen-image-2.1: backend: %w: %s", e, tail(log.String(), 4096))
	}
	st, e := os.Stat(abs)
	if e != nil {
		return GenerateResult{}, fmt.Errorf("qwen-image-2.1: backend did not create output: %w", e)
	}
	data, e := os.ReadFile(abs)
	if e != nil {
		return GenerateResult{}, e
	}
	if len(data) < 8 || string(data[:8]) != "\x89PNG\r\n\x1a\n" {
		return GenerateResult{}, fmt.Errorf("qwen-image-2.1: backend output is not PNG")
	}
	sum := sha256.Sum256(data)
	return GenerateResult{Output: abs, SHA256: hex.EncodeToString(sum[:]), Bytes: st.Size(), Elapsed: elapsed, Log: tail(log.String(), 8192)}, nil
}
func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}
