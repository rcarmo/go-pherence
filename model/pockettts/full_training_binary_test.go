package pockettts

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
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

// Many separately owned tensors expose per-array decode buffer churn. Fixture
// preparation/serialization stays outside timed reads; returned state must own
// every value and preserve the existing checkpoint validation contract.
func manyTensorBinaryFixture(tb testing.TB) FullTrainingState {
	tb.Helper()
	state := FullTrainingState{Version: 1, Step: 1, AdamW: DefaultAdamWConfig(), EMADecay: .999, Buffers: []NamedTrainingTensor{{Name: "buffer", Values: []float32{1}}}}
	for i := 0; i < 128; i++ {
		name := fmt.Sprintf("parameter.%03d", i)
		values, m, v, ema := make([]float32, 2048), make([]float32, 2048), make([]float32, 2048), make([]float32, 2048)
		for j := range values {
			values[j] = float32(j%17-8) * .001
			ema[j] = values[j]
			m[j] = float32(j%5) * .001
			v[j] = m[j] * m[j]
		}
		state.Params = append(state.Params, NamedTrainingTensor{Name: name, Values: values})
		state.Adam = append(state.Adam, NamedAdamState{Name: name, M: m, V: v})
		state.EMA = append(state.EMA, NamedTrainingTensor{Name: name, Values: ema})
	}
	if err := state.Validate(); err != nil {
		tb.Fatal(err)
	}
	return state
}

func BenchmarkFullTrainingBinaryWriteManyTensors(b *testing.B) {
	state := manyTensorBinaryFixture(b)
	b.ReportAllocs()
	b.SetBytes(128 * 2048 * 4 * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := writeFullTrainingBinary(io.Discard, state); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFullTrainingBinaryReadManyTensors(b *testing.B) {
	state := manyTensorBinaryFixture(b)
	var out bytes.Buffer
	if err := writeFullTrainingBinary(&out, state); err != nil {
		b.Fatal(err)
	}
	data := out.Bytes()
	reader := bytes.NewReader(data)
	b.ReportAllocs()
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reader.Reset(data)
		loaded, err := ReadFullTrainingStateBinary(reader, int64(len(data)))
		if err != nil {
			b.Fatal(err)
		}
		if len(loaded.Params) != 128 || len(loaded.Adam) != 128 || len(loaded.EMA) != 128 {
			b.Fatal("read inventory")
		}
		runtime.KeepAlive(loaded)
	}
}

// This format oracle encodes fields with encoding/binary independently of
// the chunk writer. Exact bits are the checkpoint format contract, not an
// acceptance gate for generated audio or training signals.
func TestFullTrainingBinaryWriterIndependentFormat(t *testing.T) {
	state := manyTensorBinaryFixture(t)
	// Cross both the 8192-value chunk boundary and separately owned arrays.
	state.Params[0].Values = make([]float32, 8193)
	for i := range state.Params[0].Values {
		state.Params[0].Values[i] = float32(i%37-18) / 37
	}
	state.Adam[0].M = make([]float32, 8193)
	state.Adam[0].V = make([]float32, 8193)
	state.EMA[0].Values = append([]float32(nil), state.Params[0].Values...)
	state.Buffers[0].Values = []float32{math.Float32frombits(0x80000000), math.SmallestNonzeroFloat32, 1}
	if err := state.Validate(); err != nil {
		t.Fatal(err)
	}
	var oracle bytes.Buffer
	oracle.Write(fullTrainingBinaryMagic[:])
	put := func(v any) {
		t.Helper()
		if err := binary.Write(&oracle, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	put(uint32(state.Version))
	put(uint64(state.Step))
	for _, v := range []float32{state.AdamW.LearningRate, state.AdamW.Beta1, state.AdamW.Beta2, state.AdamW.Epsilon, state.AdamW.WeightDecay, state.EMADecay} {
		put(v)
	}
	putTensor := func(name string, arrays ...[]float32) {
		put(uint16(len(name)))
		oracle.WriteString(name)
		put(uint64(len(arrays[0])))
		for _, values := range arrays {
			put(values)
		}
	}
	for _, group := range [][]NamedTrainingTensor{state.Params, state.Buffers} {
		put(uint32(len(group)))
		for _, tensor := range group {
			putTensor(tensor.Name, tensor.Values)
		}
	}
	put(uint32(len(state.Adam)))
	for _, moment := range state.Adam {
		putTensor(moment.Name, moment.M, moment.V)
	}
	put(uint32(len(state.EMA)))
	for _, tensor := range state.EMA {
		putTensor(tensor.Name, tensor.Values)
	}
	digest := sha256.Sum256(oracle.Bytes())
	oracle.Write(digest[:])
	var encoded bytes.Buffer
	if err := writeFullTrainingBinary(&encoded, state); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded.Bytes(), oracle.Bytes()) {
		t.Fatal("checkpoint format differs from field oracle")
	}
	loaded, err := ReadFullTrainingStateBinary(bytes.NewReader(encoded.Bytes()), int64(encoded.Len()))
	if err != nil || !reflect.DeepEqual(state, loaded) {
		t.Fatal("chunked writer round trip", err)
	}
}

func BenchmarkFullTrainingBinaryWriteSavedReleased(b *testing.B) {
	path, pin := os.Getenv("GO_PHERENCE_POCKET_BINARY_WRITE_CHECKPOINT"), os.Getenv("GO_PHERENCE_POCKET_BINARY_WRITE_SHA256")
	if path == "" {
		b.Skip("set GO_PHERENCE_POCKET_BINARY_WRITE_CHECKPOINT and GO_PHERENCE_POCKET_BINARY_WRITE_SHA256 for read-only released checkpoint serialization")
	}
	f, err := os.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		b.Fatal(err)
	}
	h := sha256.New()
	_, hashErr := io.Copy(h, f)
	closeErr := f.Close()
	if hashErr != nil || closeErr != nil || len(pin) != 64 || fmt.Sprintf("%x", h.Sum(nil)) != pin {
		b.Fatal("released checkpoint provenance", hashErr, closeErr)
	}
	state, err := LoadFullTrainingStateBinary(path, 2<<30)
	if err != nil {
		b.Fatal(err)
	}
	var count int
	for _, tensor := range state.Params {
		count += len(tensor.Values)
	}
	if count != 89449730 {
		b.Fatal("released parameter inventory", count)
	}
	b.ReportAllocs()
	b.SetBytes(info.Size())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := writeFullTrainingBinary(io.Discard, state); err != nil {
			b.Fatal(err)
		}
	}
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

type binaryFailWriter struct {
	calls, failAt int
	err           error
}

func (w *binaryFailWriter) Write(data []byte) (int, error) {
	w.calls++
	if w.calls == w.failAt {
		return 0, w.err
	}
	return len(data), nil
}

func TestFullTrainingBinaryWriterFailureBoundaries(t *testing.T) {
	state := binaryTrainingFixture(t)
	probe := &binaryFailWriter{}
	if err := writeFullTrainingBinary(probe, state); err != nil {
		t.Fatal(err)
	}
	failure := errors.New("injected checkpoint writer failure")
	for call := 1; call <= probe.calls; call++ {
		writer := &binaryFailWriter{failAt: call, err: failure}
		if err := writeFullTrainingBinary(writer, state); !errors.Is(err, failure) || writer.calls != call {
			t.Fatalf("write %d continued or lost cause: calls=%d err=%v", call, writer.calls, err)
		}
	}
	for _, group := range []struct {
		name    string
		tensors []NamedTrainingTensor
		moments []NamedAdamState
	}{
		{"count", make([]NamedTrainingTensor, maxBinaryTrainingTensors+1), nil},
		{"name", []NamedTrainingTensor{{Name: "", Values: []float32{1}}}, nil},
		{"empty", []NamedTrainingTensor{{Name: "empty"}}, nil},
		{"moments", nil, []NamedAdamState{{Name: "moments", M: []float32{1}, V: []float32{1, 2}}}},
	} {
		t.Run(group.name, func(t *testing.T) {
			invalid := state
			invalid.Params, invalid.Adam = group.tensors, group.moments
			if err := writeFullTrainingBinary(io.Discard, invalid); err == nil {
				t.Fatal("accepted invalid writer inventory")
			}
		})
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
