package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
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

func TestCLIArchivePacked(t *testing.T) {
	text := filepath.Join(t.TempDir(), "prompt.txt")
	if err := os.WriteFile(text, []byte("hello"), 0600); err != nil {
		t.Fatal(err)
	}
	var results [2]string
	for i, flag := range []string{"false", "true"} {
		var out bytes.Buffer
		if err := run(context.Background(), []string{"-model", "../../loader/needle/testdata/needle3.cact", "-text-file", text, "-packed=" + flag, "-max-new", "3", "-eos=-1"}, &out, new(bytes.Buffer)); err != nil {
			t.Fatal(err)
		}
		var obj struct {
			IDs    []int `json:"generated_ids"`
			Packed bool
		}
		if err := json.Unmarshal(out.Bytes(), &obj); err != nil {
			t.Fatal(err)
		}
		if obj.Packed != (i == 1) {
			t.Fatal("incorrect packed report")
		}
		b, _ := json.Marshal(obj.IDs)
		results[i] = string(b)
	}
	if results[0] != results[1] {
		t.Fatalf("packed/dense generation differs %v", results)
	}
}

func TestCLIHeadTrainingAndWidth(t *testing.T) {
	b, e := os.ReadFile("../../model/needle/testdata/needle3-head-width.json")
	if e != nil {
		t.Fatal(e)
	}
	var f struct {
		Width struct {
			Config  json.RawMessage
			Tensors map[string]checkpoint.Tensor
			Tokens  []int
		}
	}
	if e = json.Unmarshal(b, &f); e != nil {
		t.Fatal(e)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "source.safetensors")
	if e = checkpoint.Save(path, &checkpoint.Checkpoint{FormatVersion: 2, Config: f.Width.Config, Tensors: f.Width.Tensors}); e != nil {
		t.Fatal(e)
	}
	inp := filepath.Join(dir, "tokens.json")
	data, _ := json.Marshal(map[string]any{"tokens": f.Width.Tokens, "target": []float32{1}})
	if e = os.WriteFile(inp, data, 0600); e != nil {
		t.Fatal(e)
	}
	out := filepath.Join(dir, "head.safetensors")
	var result bytes.Buffer
	if e = run(context.Background(), []string{"-model", path, "-input", inp, "-mode", "train-head", "-head", "confidence", "-width", "8", "-steps", "2", "-out", out}, &result, new(bytes.Buffer)); e != nil {
		t.Fatal(e)
	}
	if e = run(context.Background(), []string{"-model", out, "-input", inp, "-mode", "confidence"}, new(bytes.Buffer), new(bytes.Buffer)); e != nil {
		t.Fatal(e)
	}
}

func TestToolModeAdmission(t *testing.T) {
	base := []string{"-model", "missing.cact", "-mode", "tools", "-text-file", "query.txt", "-tools", "tools.json"}
	for _, extra := range [][]string{{"-cached=false"}, {"-eos", "1"}, {"-width", "8"}, {"-max-new", "0"}, {"-max-new", "1025"}, {"-max-calls", "0"}, {"-steps", "1"}} {
		var out bytes.Buffer
		err := run(context.Background(), append(append([]string(nil), base...), extra...), &out, &out)
		if err == nil || !strings.Contains(err.Error(), "tools mode") || out.Len() != 0 {
			t.Fatalf("%v: output=%q err=%v", extra, out.String(), err)
		}
	}
	dir := t.TempDir()
	query := filepath.Join(dir, "query.txt")
	tools := filepath.Join(dir, "tools.json")
	if err := os.WriteFile(query, []byte("Turn it on."), 0600); err != nil {
		t.Fatal(err)
	}
	for _, schema := range []string{`[{"name":"x","parameters":{"type":"object"},"unknown":1}]`, `[] []`, `[{"name":"x","parameters":{"type":"object"}}]`} {
		if err := os.WriteFile(tools, []byte(schema), 0600); err != nil {
			t.Fatal(err)
		}
		var out bytes.Buffer
		err := run(context.Background(), []string{"-model", "../../loader/needle/testdata/needle3.cact", "-mode", "tools", "-tools", tools, "-text-file", query}, &out, &out)
		if err == nil || out.Len() != 0 {
			t.Fatalf("fixture admitted schema/tokenizer: %s %v", out.String(), err)
		}
	}
	var out bytes.Buffer
	if err := run(context.Background(), []string{"-model", "missing", "-input", "tokens.json", "-tools", tools}, &out, &out); err == nil || !strings.Contains(err.Error(), "requires tools mode") {
		t.Fatal(err)
	}
}

func TestNeedle2HeadCLI(t *testing.T) {
	raw, err := os.ReadFile("../../model/needle/testdata/needle2-heads.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Config  json.RawMessage              `json:"config"`
		Tensors map[string]checkpoint.Tensor `json:"tensors"`
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "v2.safetensors")
	out := filepath.Join(dir, "trained.safetensors")
	input := filepath.Join(dir, "tokens.json")
	if err = checkpoint.Save(src, &checkpoint.Checkpoint{FormatVersion: 2, Config: fixture.Config, Tensors: fixture.Tensors}); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(input, []byte(`{"tokens":[2,7,4,9,3],"target":[1,0,0,0]}`), 0600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if err = run(context.Background(), []string{"-model", src, "-input", input, "-mode", "train-head", "-head", "contrastive", "-steps", "2", "-out", out}, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	if err = run(context.Background(), []string{"-model", out, "-input", input, "-mode", "contrastive"}, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	var result struct {
		Head       string    `json:"head"`
		Values     []float32 `json:"values"`
		Calibrated bool      `json:"calibrated"`
	}
	if err = json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Head != "contrastive" || len(result.Values) != 4 || result.Calibrated {
		t.Fatalf("%s", stdout.String())
	}
}

func TestNeedle2CQCLI(t *testing.T) {
	raw, err := os.ReadFile("../../model/needle/testdata/needle2-cq.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Config  json.RawMessage              `json:"config"`
		Tensors map[string]checkpoint.Tensor `json:"tensors"`
	}
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "v2.safetensors")
	output := filepath.Join(dir, "trained.safetensors")
	input := filepath.Join(dir, "input.json")
	if err = checkpoint.Save(src, &checkpoint.Checkpoint{FormatVersion: 2, Config: f.Config, Tensors: f.Tensors}); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(input, []byte(`{"tokens":[2,7,4,9,3]}`), 0600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if err = run(context.Background(), []string{"-model", src, "-input", input, "-mode", "train", "-steps", "2", "-numerics", "needle2-cq4-a8-fp32kv", "-out", output}, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	if err = run(context.Background(), []string{"-model", output, "-input", input, "-numerics", "needle2-cq4-a8-fp32kv", "-max-new", "3"}, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	var result struct {
		Numerics string `json:"numerics"`
		IDs      []int  `json:"generated_ids"`
	}
	if err = json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Numerics != "needle2-cq4-a8-fp32kv" || len(result.IDs) != 3 {
		t.Fatal(stdout.String())
	}
	if err = run(context.Background(), []string{"-model", src, "-input", input, "-numerics", "needle3-cq4-a8-kv8"}, &stdout, &stderr); err == nil {
		t.Fatal("mismatched generation")
	}
}

func TestNeedle2ArchiveCLI(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "prompt.txt")
	if err := os.WriteFile(input, []byte("hello"), 0600); err != nil {
		t.Fatal(err)
	}
	base := []string{"-model", "../../loader/needle/testdata/needle2.cact", "-archive-config", "../../loader/needle/testdata/needle2-archive-config.json", "-text-file", input, "-max-new", "3"}
	var stdout, stderr bytes.Buffer
	if err := run(context.Background(), base, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	var result struct {
		Numerics string `json:"numerics"`
		IDs      []int  `json:"generated_ids"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Numerics != "archive-a8-fp32kv" || len(result.IDs) == 0 {
		t.Fatal(stdout.String())
	}
	stdout.Reset()
	if err := run(context.Background(), append(base, "-mode", "contrastive"), &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stdout.String(), `"calibrated":false`) {
		t.Fatal(stdout.String())
	}
	if err := run(context.Background(), []string{"-model", "../../loader/needle/testdata/needle2.cact", "-text-file", input}, &stdout, &stderr); err == nil {
		t.Fatal("no sidecar")
	}
}
