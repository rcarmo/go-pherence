// Command needle runs native Needle2/Needle3 token-ID reference inference/training.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"strings"
	"syscall"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
	"github.com/rcarmo/go-pherence/model/needle"
)

type input struct {
	Tokens []int     `json:"tokens"`
	Mask   []float32 `json:"mask,omitempty"`
	Target []float32 `json:"target,omitempty"`
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
	path := fs.String("model", "", "Needle safetensors or Needle3 .cact archive")
	tokens := fs.String("input", "", "JSON {tokens:[...],mask:[...]} input file")
	textPath := fs.String("text-file", "", "UTF-8 text file, .cact tokenizer required; prepends BOS")
	mode := fs.String("mode", "infer", "infer, tools, train, train-head, embedding, contrastive, confidence or router")
	toolsPath := fs.String("tools", "", "tool schema JSON array (tools mode, no execution)")
	systemText := fs.String("system", "", "optional tool-mode system instruction")
	maxCalls := fs.Int("max-calls", 1, "maximum schema-constrained tool calls (1..4)")
	headKind := fs.String("head", "confidence", "train-head objective: confidence BCE, router CE, embedding MSE")
	width := fs.Int("width", 0, "source Needle3 half-width rung; 0 keeps width")
	layers := fs.Int("layers", 0, "Needle3 depth rung (0 keeps all loaded layers)")
	numerics := fs.String("numerics", "fp32", "fp32 or needle3-cq4-a8-kv8 (STE training)")
	maxNew := fs.Int("max-new", 8, "maximum greedy tokens (tools mode defaults to 128)")
	cached := fs.Bool("cached", true, "use bounded incremental KV/convolution state (infer only)")
	packed := fs.Bool("packed", false, "opt-in direct CQ projections from original .cact; dense tensors remain retained")
	cacheMiB := fs.Int64("cache-mib", 512, "decoder retained/preparation budget (MiB)")
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
	if fs.NArg() != 0 || *path == "" || ((*tokens == "") == (*textPath == "")) {
		return fmt.Errorf("-model and exactly one of -input/-text-file are required; positional arguments are not accepted")
	}
	if *mode != "infer" && *mode != "tools" && *mode != "train" && *mode != "train-head" && *mode != "embedding" && *mode != "contrastive" && *mode != "confidence" && *mode != "router" {
		return fmt.Errorf("mode must be infer, tools, train, train-head, embedding, contrastive, confidence or router")
	}
	if *mode == "tools" && (*toolsPath == "" || *textPath == "") {
		return fmt.Errorf("tools mode requires -tools and -text-file")
	}
	if *mode == "tools" {
		var incompatible string
		explicitMax := false
		fs.Visit(func(f *flag.Flag) {
			switch f.Name {
			case "max-new":
				explicitMax = true
			case "eos", "out", "head", "steps", "lr", "lora-rank", "lora-alpha", "seed":
				incompatible = f.Name
			}
		})
		if incompatible != "" {
			return fmt.Errorf("tools mode does not support -%s", incompatible)
		}
		if !*cached || *width != 0 {
			return fmt.Errorf("tools mode requires cached decoding and does not support width slicing")
		}
		if !explicitMax {
			*maxNew = 128
		}
		if *maxNew < 1 || *maxNew > 1024 || *maxCalls < 1 || *maxCalls > 4 {
			return fmt.Errorf("tools mode requires -max-new 1..1024 and -max-calls 1..4")
		}
	} else {
		var toolControl string
		fs.Visit(func(f *flag.Flag) {
			if f.Name == "tools" || f.Name == "system" || f.Name == "max-calls" {
				toolControl = f.Name
			}
		})
		if toolControl != "" {
			return fmt.Errorf("-%s requires tools mode", toolControl)
		}
	}
	training := *mode == "train" || *mode == "train-head"
	if *width < 0 || *width > 4096 {
		return fmt.Errorf("width must be 0..4096")
	}
	if *mode == "train-head" {
		if *headKind != "embedding" && *headKind != "contrastive" && *headKind != "confidence" && *headKind != "router" {
			return fmt.Errorf("invalid training head")
		}
		if *rank != 0 || *textPath != "" {
			return fmt.Errorf("train-head requires token JSON with target and no LoRA")
		}
	}
	if *layers < 0 || *layers == 1 || *layers > 128 {
		return fmt.Errorf("layers must be 0 or 2..128")
	}
	if *numerics != "fp32" && *numerics != "needle3-cq4-a8-kv8" {
		return fmt.Errorf("unsupported numerics mode")
	}
	if *cacheMiB < 1 || *cacheMiB > 8192 {
		return fmt.Errorf("cache-mib must be 1..8192")
	}
	if *work < 1 || *work > 4096 {
		return fmt.Errorf("work-mib must be 1..4096")
	}
	if *steps < 1 || *steps > 100000 || *rank < 0 || *rank > 256 || *lr <= 0 || math.IsNaN(*lr) || math.IsInf(*lr, 0) || *alpha <= 0 || math.IsNaN(*alpha) || math.IsInf(*alpha, 0) || *alpha > math.MaxFloat32 {
		return fmt.Errorf("invalid training controls")
	}
	if training {
		if *out == "" {
			return fmt.Errorf("train requires -out")
		}
		if _, e := os.Stat(*out); e == nil {
			return fmt.Errorf("refusing to replace existing output %s", *out)
		} else if !errors.Is(e, os.ErrNotExist) {
			return e
		}
	}
	if e := ctx.Err(); e != nil {
		return e
	}
	var m *needle.Model
	var tok *checkpoint.Tokenizer
	var e error
	if strings.HasSuffix(strings.ToLower(*path), ".cact") {
		m, tok, e = needle.LoadArchive(*path)
	} else {
		m, e = needle.Load(*path)
	}
	if e != nil {
		return e
	}
	if m.Configuration().ArchiveDecoded {
		if *numerics != "fp32" {
			return fmt.Errorf("archive numerics are fixed; omit -numerics")
		}
		*numerics = "archive-a8-kv8"
		if training {
			return fmt.Errorf("train from source safetensors, not a deployment archive")
		}
	}
	if *mode == "tools" {
		if tok == nil {
			return fmt.Errorf("tools mode requires original archive tokenizer")
		}
		b, err := readText(*toolsPath)
		if err != nil {
			return err
		}
		if len(b) > 128<<10 {
			return fmt.Errorf("tool schema list exceeds 128 KiB")
		}
		var tools []needle.ToolSchema
		dec := json.NewDecoder(bytes.NewReader(b))
		dec.DisallowUnknownFields()
		if err = dec.Decode(&tools); err != nil {
			return err
		}
		if err = dec.Decode(new(any)); err != io.EOF {
			return fmt.Errorf("tool schema file has trailing data")
		}
		query, err := readText(*textPath)
		if err != nil {
			return err
		}
		if *layers > 0 {
			m, err = m.SliceDepth(*layers)
			if err != nil {
				return err
			}
		}
		result, err := m.GenerateTools(ctx, tok, tools, *systemText, string(query), needle.ToolOptions{MaxNewTokens: *maxNew, MaxCalls: *maxCalls, Decoder: needle.DecoderOptions{MaxCacheBytes: *cacheMiB << 20, Execution: needle.Options{MaxWorkBytes: *work << 20, Packed: *packed}}})
		if err != nil {
			return err
		}
		return json.NewEncoder(stdout).Encode(result)
	}
	var in input
	if *textPath != "" {
		if tok == nil {
			return fmt.Errorf("text input requires archive tokenizer")
		}
		b, err := readText(*textPath)
		if err != nil {
			return err
		}
		ids, err := tok.Encode(string(b))
		if err != nil {
			return err
		}
		_, defaultEOS, bos, _ := tok.SpecialIDs()
		in.Tokens = append([]int{bos}, ids...)
		setEOS := false
		fs.Visit(func(f *flag.Flag) {
			if f.Name == "eos" {
				setEOS = true
			}
		})
		if !setEOS {
			*eos = defaultEOS
		}
	} else {
		in, e = readInput(*tokens)
		if e != nil {
			return e
		}
	}
	if *layers > 0 {
		m, e = m.SliceDepth(*layers)
		if e != nil {
			return e
		}
	}
	if *width > 0 {
		m, e = m.SliceWidth(*width)
		if e != nil {
			return e
		}
	}
	opts := needle.Options{MaxWorkBytes: *work << 20, Packed: *packed}
	if *numerics == "needle3-cq4-a8-kv8" {
		opts.Quant = &needle.Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}
	}
	enc := json.NewEncoder(stdout)
	if *mode == "infer" {
		var generated []int
		if *cached {
			generated, e = m.GenerateCached(ctx, in.Tokens, *maxNew, *eos, needle.DecoderOptions{Execution: opts, MaxCacheBytes: *cacheMiB << 20})
		} else {
			generated, e = m.Generate(ctx, in.Tokens, *maxNew, *eos, opts)
		}
		if e != nil {
			return e
		}
		result := map[string]any{"generation": m.Configuration().Generation, "numerics": *numerics, "generated_ids": generated, "cached": *cached, "packed": *packed}
		if tok != nil {
			decoded, err := tok.Decode(generated)
			if err != nil {
				return err
			}
			result["generated_text"] = decoded
		}
		return enc.Encode(result)
	}
	if !training {
		values, err := m.Head(in.Tokens, needle.HeadKind(*mode), opts)
		if err != nil {
			return err
		}
		return enc.Encode(map[string]any{"generation": m.Configuration().Generation, "numerics": *numerics, "head": *mode, "values": values, "calibrated": false, "packed": *packed})
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
		if *mode == "train-head" {
			m, loss, e = m.TrainHeadStep(opt, in.Tokens, needle.HeadKind(*headKind), in.Target, *lr, opts)
		} else if ad == nil {
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
	return enc.Encode(map[string]any{"generation": m.Configuration().Generation, "numerics": *numerics, "steps": *steps, "losses": losses, "checkpoint": *out, "lora_rank": *rank, "mode": *mode, "calibrated": false})
}
func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if e := run(ctx, os.Args[1:], os.Stdout, os.Stderr); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}

func readText(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, (1<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(b) > 1<<20 {
		return nil, fmt.Errorf("text input exceeds 1 MiB")
	}
	return b, nil
}
