package main

import (
	"context"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	model "github.com/rcarmo/go-pherence/model/omnivoice"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestJoinChunkWaves(t *testing.T) {
	a := []float32{1, 1, 1, 1}
	b := []float32{.5, .5, .5, .5}
	out, err := joinChunkWaves([][]float32{a, b}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 1, 1, 0, 0, 0, 0, .5, .5, 0}
	if !reflect.DeepEqual(out, want) {
		t.Fatal(out)
	}
	if a[0] != 1 || b[0] != .5 {
		t.Fatal("mutated source")
	}
	for _, waves := range [][][]float32{nil, {nil}, {{float32(math.NaN())}}} {
		if _, err := joinChunkWaves(waves, 2, 2); err == nil {
			t.Fatal("accepted invalid chunks")
		}
	}
}

func TestRealChunkRunnerResizeParity(t *testing.T) {
	root := os.Getenv("GO_PHERENCE_REAL_OMNIVOICE")
	if root == "" {
		t.Skip("set GO_PHERENCE_REAL_OMNIVOICE")
	}
	w, err := loader.OpenWeights(root)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	cw, err := loader.LoadCodecDecoder(filepath.Join(root, "audio_tokenizer"))
	if err != nil {
		t.Fatal(err)
	}
	d, err := model.NewCodecDecoder(cw)
	if err != nil {
		t.Fatal(err)
	}
	makePrompt := func(frames int) loader.PreparedPrompt {
		n := frames + 1
		ids := make([]int, 8*n)
		ui := make([]int, 8*frames)
		mask := make([]bool, n)
		um := make([]bool, frames)
		for i := 1; i < n; i++ {
			mask[i] = true
		}
		for i := range um {
			um[i] = true
		}
		for b := 0; b < 8; b++ {
			ids[b*n] = 42
			for i := 1; i < n; i++ {
				ids[b*n+i] = w.Config.AudioMaskID
			}
		}
		for i := range ui {
			ui[i] = w.Config.AudioMaskID
		}
		return loader.PreparedPrompt{TargetFrames: frames, Conditional: loader.PreparedInput{Tokens: n, IDs: ids, AudioMask: mask}, Unconditional: loader.PreparedInput{Tokens: frames, IDs: ui, AudioMask: um}}
	}
	prompts := []loader.PreparedPrompt{makePrompt(2), makePrompt(1), makePrompt(2)}
	runner, err := newChunkRunner(w, d, prompts, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range prompts {
		got, err := runner.Generate(context.Background(), p, false)
		if err != nil {
			t.Fatal(err)
		}
		fresh, err := newChunkRunner(w, d, []loader.PreparedPrompt{p}, 2)
		if err != nil {
			t.Fatal(err)
		}
		want, err := fresh.Generate(context.Background(), p, false)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("chunk %d reuse differs", i)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := runner.Generate(ctx, prompts[0], false); err == nil {
		t.Fatal("cancel ignored")
	}
	if _, err := runner.Generate(context.Background(), prompts[0], false); err != nil {
		t.Fatal(err)
	}
}
