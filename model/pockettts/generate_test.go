package pockettts

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestWritePCM16Mono24k(t *testing.T) {
	p := filepath.Join(t.TempDir(), "out.wav")
	if err := WritePCM16Mono24k(p, []float32{-1, -.5, 0, .5, 1}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 54 || string(data[:4]) != "RIFF" || binary.LittleEndian.Uint32(data[24:]) != SampleRate || int16(binary.LittleEndian.Uint16(data[44:])) != math.MinInt16 {
		t.Fatalf("bad WAV %d", len(data))
	}
	if err := WritePCM16Mono24k(p, []float32{0}); err == nil {
		t.Fatal("overwrote WAV")
	}
}
func TestWritePCM16Mono24kRejectsMalformed(t *testing.T) {
	for _, v := range [][]float32{nil, {float32(math.NaN())}, {1.1}} {
		p := filepath.Join(t.TempDir(), "bad.wav")
		if err := WritePCM16Mono24k(p, v); err == nil {
			t.Fatalf("accepted %v", v)
		}
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Fatalf("left output %v", err)
		}
	}
}
