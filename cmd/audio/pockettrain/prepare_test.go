//go:build linux

package main

import (
	"io"
	"math"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/model/pockettts"
)

func TestPreparedNormalization(t *testing.T) {
	out, err := normalizePrepared([]float32{3, 6, 5, 10}, []float32{1, 2}, []float32{2, 4})
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{1, 1, 2, 2} {
		if out[i] != want {
			t.Fatal(out)
		}
	}
	for _, std := range [][]float32{{0, 1}, {-1, 1}, {float32(math.NaN()), 1}} {
		if _, err := normalizePrepared([]float32{1, 2}, []float32{0, 0}, std); err == nil {
			t.Fatal("invalid std admitted")
		}
	}
}
func TestPreparedRowAssembly(t *testing.T) {
	wav := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	geometry := rowReport{SegmentStartSample: 1, CutSample: 3, SegmentEndSample: 7, TargetFrames: 2, VoiceFrames: 1}
	storage := make([]float32, 4)
	call := 0
	encode := func(samples []float32) ([]float32, int, error) {
		call++
		if call == 1 {
			if !reflect.DeepEqual(samples, []float32{1, 2}) {
				t.Fatal("voice slice", samples)
			}
			copy(storage, []float32{3, 6})
			return storage[:2], 1, nil
		}
		if !reflect.DeepEqual(samples, []float32{3, 4, 5, 6}) {
			t.Fatal("target slice", samples)
		}
		copy(storage, []float32{3, 6, 5, 10})
		return storage, 2, nil
	}
	tokens := []uint32{1, 2}
	row, err := assemblePreparedRow(wav, geometry, tokens, []float32{1, 2}, []float32{2, 4}, encode)
	if err != nil {
		t.Fatal(err)
	}
	want := pockettts.PreparedTrainingRow{Batch: pockettts.FlowLMTrainingBatch{Frames: 3, VoiceFrames: 1, NormalizedLatents: []float32{1, 1, 2, 2, 0, 0}, VoiceLatents: []float32{3, 6}, TextTokens: []uint32{1, 2}}, ValidFrames: 2}
	if !reflect.DeepEqual(row, want) {
		t.Fatalf("assembled row=%+v", row)
	}
	clear(storage)
	tokens[0] = 99
	if !reflect.DeepEqual(row, want) {
		t.Fatal("prepared row retains mutable codec/token storage")
	}
}
func TestPreparedRowAssemblyRejectsBeforeCodec(t *testing.T) {
	g := rowReport{SegmentStartSample: 0, CutSample: 2, SegmentEndSample: 4, TargetFrames: 375, VoiceFrames: 1}
	calls := 0
	encode := func([]float32) ([]float32, int, error) { calls++; return nil, 0, nil }
	if _, err := assemblePreparedRow(make([]float32, 4), g, []uint32{1}, []float32{0}, []float32{1}, encode); err == nil || calls != 0 {
		t.Fatal("375-frame EOS overflow reached codec")
	}
	g.TargetFrames = 1
	if _, err := assemblePreparedRow(make([]float32, 4), g, []uint32{1}, []float32{0}, []float32{1}, encode); err == nil {
		t.Fatal("codec frame mismatch admitted")
	}
}
func TestPreparedPreparationAdmission(t *testing.T) {
	if !prepareRequested([]string{"-mode", "prepare"}) || !prepareRequested([]string{"-mode=prepare"}) || prepareRequested([]string{"-mode=train"}) {
		t.Fatal("prepare mode")
	}
	for _, args := range [][]string{{"-backend=ptx"}, {"-prepared-output=/missing", "-max-prepared-bytes=0"}} {
		if err := prepareMain(args, io.Discard); err == nil {
			t.Fatal("invalid prepare admitted")
		}
	}
	w := preparedLimitWriter{writer: io.Discard, left: 2}
	if _, err := w.Write([]byte("abc")); err == nil {
		t.Fatal("output bound")
	}
}
