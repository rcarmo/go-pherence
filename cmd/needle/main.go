// Command needle runs the native FP32 Needle2/Needle3 token-ID reference path.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"syscall"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
	"github.com/rcarmo/go-pherence/model/needle"
)

type input struct {
	Tokens []int     `json:"tokens"`
	Mask   []float32 `json:"mask,omitempty"`
}

func readInput(path string) (input, error) {
	var out input
	f, e := os.Open(path)
	if e != nil {
		return out, e
	}
	defer f.Close()
	info, e := f.Stat()
	if e != nil {
		return out, e
	}
	if info.Size() > 1<<20 {
		return out, fmt.Errorf("input exceeds 1 MiB")
	}
	d := json.NewDecoder(io.LimitReader(f, (1<<20)+1))
	d.DisallowUnknownFields()
	if e = d.Decode(&out); e != nil {
		return out, e
	}
	if e = d.Decode(new(any)); !errors.Is(e, io.EOF) {
		return out, fmt.Errorf("input must contain one JSON object")
	}
	return out, nil
}
func run(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("needle", flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("model", "", "Needle safetensors checkpoint (explicit generation 2/3)")
	tokens := fs.String("input", "", "JSON {tokens:[...],mask:[...]} input file")
	mode := fs.String("mode", "infer", "infer or train (FP32 only)")
	maxNew := fs.Int("max-new", 8, "maximum greedy tokens (recomputes prefix)")
	eos := fs.Int("eos", 1, "EOS token, -1 disables early stop")
	steps := fs.Int("steps", 1, "number of optimization steps on the input example")
	lr := fs.Float64("lr", .001, "AdamW learning rate")
	rank := fs.Int("lora-rank", 0, "0 trains full trunk; positive selects LoRA rank")
	alpha := fs.Float64("lora-alpha", 16, "LoRA alpha")
	seed := fs.Uint64("seed", 0, "LoRA initialization seed")
	out := fs.String("out", "", "new merged safetensors checkpoint for train mode")
	work := fs.Int64("work-mib", 512, "logical per-forward/backward workspace cap (MiB)")
	if e := fs.Parse(args); e != nil {
		return e
	}
	if fs.NArg() != 0 || *path == "" || *tokens == "" {
		return fmt.Errorf("-model and -input are required; positional arguments are not accepted")
	}
	if *mode != "infer" && *mode != "train" {
		return fmt.Errorf("mode must be infer or train")
	}
	if *work < 1 || *work > 4096 {
		return fmt.Errorf("work-mib must be 1..4096")
	}
	if *steps < 1 || *steps > 100000 || *rank < 0 || *rank > 256 || *lr <= 0 || math.IsNaN(*lr) || math.IsInf(*lr, 0) || *alpha <= 0 || math.IsNaN(*alpha) || math.IsInf(*alpha, 0) || *alpha > math.MaxFloat32 {
		return fmt.Errorf("invalid training controls")
	}
	if *mode == "train" {
		if *out == "" {
			return fmt.Errorf("train requires -out")
		}
		if _, e := os.Stat(*out); e == nil {
			return fmt.Errorf("refusing to replace existing output %s", *out)
		} else if !errors.Is(e, os.ErrNotExist) {
			return e
		}
	}
	in, e := readInput(*tokens)
	if e != nil {
		return e
	}
	if e = ctx.Err(); e != nil {
		return e
	}
	m, e := needle.Load(*path)
	if e != nil {
		return e
	}
	opts := needle.Options{MaxWorkBytes: *work << 20}
	enc := json.NewEncoder(stdout)
	if *mode == "infer" {
		generated, e := m.Generate(ctx, in.Tokens, *maxNew, *eos, opts)
		if e != nil {
			return e
		}
		return enc.Encode(map[string]any{"generation": m.Configuration().Generation, "numerics": "fp32", "generated_ids": generated})
	}
	opt := needle.NewAdamW()
	var ad *needle.Adapter
	if *rank > 0 {
		ad, e = m.NewAdapter(*rank, float32(*alpha), *seed)
		if e != nil {
			return e
		}
	}
	losses := make([]float64, 0, *steps)
	for step := 0; step < *steps; step++ {
		if e = ctx.Err(); e != nil {
			return e
		}
		var loss float64
		if ad == nil {
			m, loss, e = m.TrainStep(opt, in.Tokens, in.Mask, *lr, opts)
		} else {
			ad, loss, e = m.TrainAdapterStep(ad, opt, in.Tokens, in.Mask, *lr, opts)
		}
		if e != nil {
			return e
		}
		losses = append(losses, loss)
	}
	if ad != nil {
		m, e = m.Merge(ad)
		if e != nil {
			return e
		}
	}
	if e = ctx.Err(); e != nil {
		return e
	}
	if e = checkpoint.Save(*out, m.Checkpoint()); e != nil {
		return e
	}
	return enc.Encode(map[string]any{"generation": m.Configuration().Generation, "numerics": "fp32", "steps": *steps, "losses": losses, "checkpoint": *out, "lora_rank": *rank})
}
func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if e := run(ctx, os.Args[1:], os.Stdout, os.Stderr); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
