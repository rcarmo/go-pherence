package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

func TestCLITrainingAndInference(t *testing.T) {
	b, e := os.ReadFile("../../model/needle/testdata/needle3.json")
	if e != nil {
		t.Fatal(e)
	}
	var f struct {
		Config  json.RawMessage
		Tensors map[string]checkpoint.Tensor
		Tokens  []int
	}
	if e = json.Unmarshal(b, &f); e != nil {
		t.Fatal(e)
	}
	dir := t.TempDir()
	base := filepath.Join(dir, "base.safetensors")
	if e = checkpoint.Save(base, &checkpoint.Checkpoint{Config: f.Config, FormatVersion: 2, Tensors: f.Tensors}); e != nil {
		t.Fatal(e)
	}
	input := filepath.Join(dir, "tokens.json")
	data, _ := json.Marshal(map[string]any{"tokens": f.Tokens})
	if e = os.WriteFile(input, data, 0600); e != nil {
		t.Fatal(e)
	}
	out := filepath.Join(dir, "trained.safetensors")
	var stdout, stderr bytes.Buffer
	args := []string{"-model", base, "-input", input, "-mode", "train", "-steps", "3", "-lora-rank", "2", "-out", out}
	if e = run(context.Background(), args, &stdout, &stderr); e != nil {
		t.Fatal(e)
	}
	if e = run(context.Background(), args, &stdout, &stderr); e == nil {
		t.Fatal("overwrote checkpoint")
	}
	stdout.Reset()
	if e = run(context.Background(), []string{"-model", out, "-input", input, "-max-new", "2"}, &stdout, &stderr); e != nil {
		t.Fatal(e)
	}
	var result struct {
		Generated []int `json:"generated_ids"`
	}
	if e = json.Unmarshal(stdout.Bytes(), &result); e != nil {
		t.Fatal(e)
	}
	if len(result.Generated) == 0 {
		t.Fatal("no generated tokens")
	}
}
func TestCLIAdmission(t *testing.T) {
	for _, args := range [][]string{nil, {"-model", "x", "-input", "y", "-mode", "bad"}, {"-model", "x", "-input", "y", "-work-mib", "-1"}, {"-model", "x", "-input", "y", "-mode", "train"}, {"-model", "x", "-input", "y", "-lr", "NaN"}} {
		if err := run(context.Background(), args, new(bytes.Buffer), new(bytes.Buffer)); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}

func TestCLIHeadsAndDepth(t *testing.T) {
	b, e := os.ReadFile("../../model/needle/testdata/needle3-extended.json")
	if e != nil {
		t.Fatal(e)
	}
	var f struct {
		Base struct {
			Config  json.RawMessage
			Tensors map[string]checkpoint.Tensor
			Tokens  []int
		}
	}
	if e = json.Unmarshal(b, &f); e != nil {
		t.Fatal(e)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "base.safetensors")
	if e = checkpoint.Save(path, &checkpoint.Checkpoint{Config: f.Base.Config, FormatVersion: 2, Tensors: f.Base.Tensors}); e != nil {
		t.Fatal(e)
	}
	inp := filepath.Join(dir, "tokens.json")
	data, _ := json.Marshal(map[string]any{"tokens": f.Base.Tokens})
	if e = os.WriteFile(inp, data, 0600); e != nil {
		t.Fatal(e)
	}
	for _, mode := range []string{"embedding", "confidence", "router"} {
		var out bytes.Buffer
		if e = run(context.Background(), []string{"-model", path, "-input", inp, "-mode", mode, "-layers", "2"}, &out, new(bytes.Buffer)); e != nil {
			t.Fatal(e)
		}
		var got struct {
			Values     []float32
			Calibrated bool
		}
		if e = json.Unmarshal(out.Bytes(), &got); e != nil {
			t.Fatal(e)
		}
		if len(got.Values) == 0 || got.Calibrated {
			t.Fatal("invalid head result")
		}
	}
}

func TestCLIArchiveText(t *testing.T) {
	path := "../../loader/needle/testdata/needle3.cact"
	text := filepath.Join(t.TempDir(), "prompt.txt")
	if err := os.WriteFile(text, []byte("hello"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"infer", "embedding", "router"} {
		var out bytes.Buffer
		if err := run(context.Background(), []string{"-model", path, "-text-file", text, "-mode", mode, "-max-new", "2", "-eos", "-1"}, &out, new(bytes.Buffer)); err != nil {
			t.Fatal(err)
		}
		var obj map[string]any
		if err := json.Unmarshal(out.Bytes(), &obj); err != nil {
			t.Fatal(err)
		}
		if obj["numerics"] != "archive-a8-kv8" {
			t.Fatal("wrong archive numerics")
		}
	}
	if err := run(context.Background(), []string{"-model", path, "-text-file", text, "-numerics", "needle3-cq4-a8-kv8"}, new(bytes.Buffer), new(bytes.Buffer)); err == nil {
		t.Fatal("requantization accepted")
	}
}

func TestCLIReferenceAndCachedTextParity(t *testing.T) {
	path := "../../loader/needle/testdata/needle3.cact"
	text := filepath.Join(t.TempDir(), "prompt.txt")
	if err := os.WriteFile(text, []byte("hello"), 0600); err != nil {
		t.Fatal(err)
	}
	var generated [2]string
	for i, cache := range []string{"true", "false"} {
		var out bytes.Buffer
		if err := run(context.Background(), []string{"-model", path, "-text-file", text, "-cached=" + cache, "-max-new", "4", "-eos=-1"}, &out, new(bytes.Buffer)); err != nil {
			t.Fatal(err)
		}
		var obj struct {
			IDs    []int `json:"generated_ids"`
			Cached bool
		}
		if err := json.Unmarshal(out.Bytes(), &obj); err != nil {
			t.Fatal(err)
		}
		if obj.Cached != (i == 0) {
			t.Fatal("wrong cache report")
		}
		b, _ := json.Marshal(obj.IDs)
		generated[i] = string(b)
	}
	if generated[0] != generated[1] {
		t.Fatalf("cached/reference differ %v", generated)
	}
}
