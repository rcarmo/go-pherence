package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/rcarmo/go-pherence/model/diffusiongemma"
)

type report struct {
	ModelPath string                          `json:"model_path"`
	PromptIDs []int                           `json:"prompt_ids"`
	Options   diffusiongemma.InferenceOptions `json:"options"`
	Result    *diffusiongemma.InferenceResult `json:"result,omitempty"`
	Error     string                          `json:"error,omitempty"`
}

func main() {
	modelDir := flag.String("model", "", "DiffusionGemma model directory")
	promptCSV := flag.String("prompt-ids", "", "comma-separated already-tokenized prompt IDs")
	maxNew := flag.Int("max-new", 0, "maximum generated tokens")
	canvas := flag.Int("canvas", 0, "override canvas length")
	seed := flag.Int64("seed", 1, "deterministic canvas RNG seed")
	asJSON := flag.Bool("json", false, "emit JSON")
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: diffusiongemmarun -model PATH [-prompt-ids 1,2] [-max-new N] [-json]")
		os.Exit(2)
	}
	promptIDs, err := parseIDs(*promptCSV)
	if err != nil {
		fatal(err)
	}
	m, err := diffusiongemma.LoadMetadata(*modelDir)
	if err != nil {
		fatal(err)
	}
	eng, err := diffusiongemma.NewEngine(m, nil)
	if err != nil {
		fatal(err)
	}
	opts := diffusiongemma.InferenceOptions{MaxNewTokens: *maxNew, CanvasLength: *canvas, Seed: *seed}
	out := report{ModelPath: *modelDir, PromptIDs: promptIDs, Options: opts}
	res, err := eng.GenerateTokenIDs(promptIDs, opts)
	if err != nil {
		out.Error = err.Error()
	} else {
		out.Result = &res
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fatal(err)
		}
		if out.Error != "" {
			os.Exit(1)
		}
		return
	}
	fmt.Printf("DiffusionGemma run scaffold: %s\n", *modelDir)
	fmt.Printf("  prompt_ids=%v max_new=%d canvas=%d seed=%d\n", promptIDs, opts.MaxNewTokens, opts.CanvasLength, opts.Seed)
	if out.Error != "" {
		fmt.Printf("  error: %s\n", out.Error)
		os.Exit(1)
	}
	fmt.Printf("  generated=%v canvases=%d\n", res.Generated, len(res.Canvases))
}

func parseIDs(csv string) ([]int, error) {
	if strings.TrimSpace(csv) == "" {
		return nil, nil
	}
	parts := strings.Split(csv, ",")
	ids := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		id, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("bad token id %q: %w", part, err)
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "diffusiongemmarun:", err)
	os.Exit(1)
}
