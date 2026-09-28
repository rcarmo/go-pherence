package pockettts

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

func binaryTrainingFixture(t *testing.T) FullTrainingState {
	t.Helper()
	oracle := loadTrainingStepOracle(t)
	lm, flow, weighting := trainingStepModelsFromOracle(t, oracle)
	trainer, err := NewFullTrainer(lm, flow, weighting, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
	samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
	_, gradients, err := PocketTrainingStep(lm, flow, weighting, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := trainer.Step(gradients); err != nil {
		t.Fatal(err)
	}
	state, err := trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	return state
}

func TestFullTrainingBinaryRoundTripAndResume(t *testing.T) {
	state := binaryTrainingFixture(t)
	path := filepath.Join(t.TempDir(), "full.bin")
	if err := SaveFullTrainingStateBinary(path, state); err != nil {
		t.Fatal(err)
	}
	encoded, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) < 32 || !bytes.Equal(encoded[:16], fullTrainingBinaryMagic[:]) {
		t.Fatal("missing binary checkpoint header")
	}
	loaded, err := LoadFullTrainingStateBinary(path, int64(len(encoded)))
	if err != nil || !reflect.DeepEqual(state, loaded) {
		t.Fatalf("binary round trip: err=%v equal=%v", err, reflect.DeepEqual(state, loaded))
	}
	again := filepath.Join(t.TempDir(), "again.bin")
	if err := SaveFullTrainingStateBinary(again, loaded); err != nil {
		t.Fatal(err)
	}
	identical, err := os.ReadFile(again)
	if err != nil || !bytes.Equal(encoded, identical) {
		t.Fatalf("binary checkpoint is not deterministic: %v", err)
	}
	if err := SaveFullTrainingStateBinary(path, loaded); err != nil {
		t.Fatalf("replace existing checkpoint: %v", err)
	}
	replaced, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(encoded, replaced) {
		t.Fatalf("replaced checkpoint differs: %v", err)
	}
	oracle := loadTrainingStepOracle(t)
	lm, flow, weighting := trainingStepModelsFromOracle(t, oracle)
	resumed, err := NewFullTrainer(lm, flow, weighting, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.LoadState(loaded); err != nil {
		t.Fatal(err)
	}
	got, err := resumed.State()
	if err != nil || !reflect.DeepEqual(state, got) {
		t.Fatalf("binary resume differs: %v", err)
	}
}

func TestFullTrainingBinaryLoadStateOwnsCallerValues(t *testing.T) {
	state := binaryTrainingFixture(t)
	path := filepath.Join(t.TempDir(), "full.bin")
	if err := SaveFullTrainingStateBinary(path, state); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadFullTrainingStateBinary(path, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	oracle := loadTrainingStepOracle(t)
	lm, flow, weighting := trainingStepModelsFromOracle(t, oracle)
	trainer, err := NewFullTrainer(lm, flow, weighting, DefaultAdamWConfig(), .999)
	if err != nil {
		t.Fatal(err)
	}
	if err = trainer.LoadState(loaded); err != nil {
		t.Fatal(err)
	}
	before, err := trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	loaded.Params[0].Values[0] += 10
	loaded.Buffers[0].Values[0] += 10
	loaded.Adam[0].M[0] += 10
	loaded.Adam[0].V[0] += 10
	loaded.EMA[0].Values[0] += 10
	loaded = FullTrainingState{}
	runtime.GC()
	after, err := trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatal("loaded checkpoint retains caller-owned parameter, buffer, moment or EMA storage")
	}
}

func TestFullTrainingBinaryRequiresExistingDirectory(t *testing.T) {
	parent := t.TempDir()
	path := filepath.Join(parent, "absent", "full.bin")
	if err := SaveFullTrainingStateBinary(path, binaryTrainingFixture(t)); err == nil {
		t.Fatal("accepted nonexistent checkpoint directory")
	}
	if _, err := os.Stat(filepath.Dir(path)); !os.IsNotExist(err) {
		t.Fatalf("checkpoint save unexpectedly created directory: %v", err)
	}
}

func TestFullTrainingBinaryRejectsHostileInput(t *testing.T) {
	state := binaryTrainingFixture(t)
	var out bytes.Buffer
	if err := writeFullTrainingBinary(&out, state); err != nil {
		t.Fatal(err)
	}
	good := out.Bytes()
	corrupt := append([]byte(nil), good...)
	corrupt[len(corrupt)-33] ^= 1
	trailing := append(append([]byte(nil), good...), 0)
	badMagic := append([]byte(nil), good...)
	badMagic[0] ^= 1
	hugeCount := append([]byte(nil), good...)
	binary.LittleEndian.PutUint32(hugeCount[52:56], maxBinaryTrainingTensors+1)
	hugeLength := append([]byte(nil), good...)
	nameLength := int(binary.LittleEndian.Uint16(hugeLength[56:58]))
	binary.LittleEndian.PutUint64(hugeLength[58+nameLength:66+nameLength], ^uint64(0))
	for name, data := range map[string][]byte{
		"empty": {}, "header": good[:12], "truncated": good[:len(good)-1],
		"corrupt": corrupt, "trailing": trailing, "bad magic": badMagic,
		"huge count": hugeCount, "huge length": hugeLength,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := ReadFullTrainingStateBinary(bytes.NewReader(data), int64(len(good)+1)); err == nil {
				t.Fatal("accepted malformed checkpoint")
			}
		})
	}
	for _, limit := range []int64{-1, 0, int64(len(good) - 1)} {
		if _, err := ReadFullTrainingStateBinary(bytes.NewReader(good), limit); err == nil {
			t.Fatalf("accepted limit %d", limit)
		}
	}
	path := filepath.Join(t.TempDir(), "full.bin")
	if err := os.WriteFile(path, good, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadFullTrainingStateBinary(path, int64(len(good)-1)); err == nil {
		t.Fatal("accepted over-limit file")
	}
	// A small forged file with a huge declared tensor must not allocate at
	// the caller's much larger ceiling.
	if err := os.WriteFile(path, hugeLength[:66+nameLength], 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadFullTrainingStateBinary(path, 2<<30); err == nil {
		t.Fatal("accepted forged file tensor length")
	}
	if err := os.WriteFile(path, good, 0600); err != nil {
		t.Fatal(err)
	}
	if err := SaveFullTrainingStateBinary(path, FullTrainingState{}); err == nil {
		t.Fatal("accepted invalid state")
	}
	untouched, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(untouched, good) {
		t.Fatalf("invalid write changed destination: %v", err)
	}
}

type shortDigestWriter struct {
	writes int
}

func (w *shortDigestWriter) Write(p []byte) (int, error) {
	w.writes++
	if len(p) == 32 {
		return 31, nil
	}
	return len(p), nil
}

func TestFullTrainingBinaryRejectsShortDigestWrite(t *testing.T) {
	writer := &shortDigestWriter{}
	if err := writeFullTrainingBinary(writer, binaryTrainingFixture(t)); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short checksum write: got %v want io.ErrShortWrite", err)
	}
	if writer.writes < 2 {
		t.Fatal("checkpoint body was not written")
	}
}
