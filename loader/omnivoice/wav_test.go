package omnivoice

import (
	audio "github.com/rcarmo/go-pherence/loader/audio"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSyntheticWAV(t *testing.T) {
	p := filepath.Join(t.TempDir(), "synthetic.wav")
	gain, err := WriteSyntheticWAV(p, []float32{0, 2, -2, .5}, 24000)
	if err != nil {
		t.Fatal(err)
	}
	if gain != .49 {
		t.Fatal(gain)
	}
	samples, rate, err := audio.WAV(p)
	if err != nil || rate != 24000 || len(samples) != 4 {
		t.Fatalf("bad WAV %d %d %v", rate, len(samples), err)
	}
	if math.Abs(float64(samples[1])-.98) > .0001 {
		t.Fatal(samples)
	}
	original, _ := os.ReadFile(p)
	if _, err = WriteSyntheticWAV(p, []float32{.1}, 24000); err == nil {
		t.Fatal("overwrote")
	}
	after, _ := os.ReadFile(p)
	if string(after) != string(original) {
		t.Fatal("changed output")
	}
}
func TestSyntheticWAVRejectsNonfinite(t *testing.T) {
	p := filepath.Join(t.TempDir(), "invalid.wav")
	for _, s := range [][]float32{nil, {0}, {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, err := WriteSyntheticWAV(p, s, 24000); err == nil {
			t.Fatal("invalid audio accepted")
		}
	}
	if _, err := os.Stat(p); !os.IsNotExist(err) {
		t.Fatal("created invalid output")
	}
}
