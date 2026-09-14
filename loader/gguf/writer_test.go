package gguf

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestWriteTensorsRoundTrip(t *testing.T) {
	f32Vals := []float32{1.25, -2.5, 3.75, 4.5}
	f16Vals := []float32{0.125, -0.75, 1.5, -2.25, 3.5}
	q8Vals := make([]float32, 32)
	for i := range q8Vals {
		q8Vals[i] = float32((i%11)-5) * 0.1875
	}

	f32Raw := encodeF32Raw(f32Vals)
	f16Raw := encodeF16Raw(f16Vals)
	q8Raw := encodeQ8Raw(t, q8Vals)

	path := filepath.Join(t.TempDir(), "synthetic.gguf")
	err := WriteTensors(context.Background(), path,
		[]MetadataEntry{
			{Key: "general.architecture", Value: "omnivoice"},
			{Key: "omnivoice.hidden_size", Value: uint32(768)},
			{Key: "omnivoice.layers", Value: int64(12)},
		},
		[]ExportTensor{
			{Name: "decoder.embed.weight", Shape: []uint64{4}, QType: QuantF32, Raw: bytes.NewReader(f32Raw)},
			{Name: "decoder.norm.weight", Shape: []uint64{5}, QType: QuantF16, Raw: bytes.NewReader(f16Raw)},
			{Name: "decoder.proj.weight", Shape: []uint64{32}, QType: QuantQ8_0, Raw: bytes.NewReader(q8Raw)},
		},
	)
	if err != nil {
		t.Fatalf("WriteTensors: %v", err)
	}

	g, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer g.Close()

	if g.DataOffset%32 != 0 {
		t.Fatalf("DataOffset=%d not 32-byte aligned", g.DataOffset)
	}
	if got, ok := g.MetaString("general.architecture"); !ok || got != "omnivoice" {
		t.Fatalf("MetaString general.architecture = %q, %v", got, ok)
	}
	if got, ok := g.MetaUint32("omnivoice.hidden_size"); !ok || got != 768 {
		t.Fatalf("MetaUint32 omnivoice.hidden_size = %d, %v", got, ok)
	}
	if got, ok := g.Meta["omnivoice.layers"].(int64); !ok || got != 12 {
		t.Fatalf("metadata omnivoice.layers = %#v", g.Meta["omnivoice.layers"])
	}

	cases := []struct {
		name string
		raw  []byte
		qt   QuantType
		n    int
	}{
		{name: "decoder.embed.weight", raw: f32Raw, qt: QuantF32, n: len(f32Vals)},
		{name: "decoder.norm.weight", raw: f16Raw, qt: QuantF16, n: len(f16Vals)},
		{name: "decoder.proj.weight", raw: q8Raw, qt: QuantQ8_0, n: len(q8Vals)},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			tensor, ok := g.TensorByName(tc.name)
			if !ok {
				t.Fatalf("missing tensor %q", tc.name)
			}
			if tensor.Offset%32 != 0 {
				t.Fatalf("tensor offset=%d not 32-byte aligned", tensor.Offset)
			}
			if tensor.QType != tc.qt {
				t.Fatalf("QType=%v want %v", tensor.QType, tc.qt)
			}

			raw, err := g.Raw(tensor)
			if err != nil {
				t.Fatalf("Raw: %v", err)
			}
			if !bytes.Equal(raw, tc.raw) {
				t.Fatalf("raw mismatch\n got=%x\nwant=%x", raw, tc.raw)
			}

			got, err := g.DequantF32(tensor)
			if err != nil {
				t.Fatalf("DequantF32: %v", err)
			}
			want, err := DequantToF32(tc.raw, tc.qt, tc.n)
			if err != nil {
				t.Fatalf("DequantToF32 want: %v", err)
			}
			if len(got) != len(want) {
				t.Fatalf("len=%d want %d", len(got), len(want))
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("value[%d]=%v want %v", i, got[i], want[i])
				}
			}
		})
	}
}

func TestWriteV3RemovesPartialFileOnShortPayload(t *testing.T) {
	path := filepath.Join(t.TempDir(), "short.gguf")
	err := WriteTensors(context.Background(), path, nil, []ExportTensor{{
		Name:  "bad.q8",
		Shape: []uint64{32},
		QType: QuantQ8_0,
		Raw:   bytes.NewReader(make([]byte, 10)),
	}})
	if err == nil {
		t.Fatal("expected error")
	}
	if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
		t.Fatalf("partial file still present: %v", statErr)
	}
}

func TestWriteV3RemovesPartialFileOnCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	path := filepath.Join(t.TempDir(), "cancel.gguf")
	f32Raw := encodeF32Raw([]float32{1, 2, 3, 4})

	err := WriteV3(ctx, path, nil,
		[]TensorSpec{
			{Name: "first", Shape: []uint64{4}, QType: QuantF32},
			{Name: "second", Shape: []uint64{4}, QType: QuantF32},
		},
		func(_ context.Context, index int, _ TensorSpec) (io.Reader, error) {
			if index == 1 {
				cancel()
			}
			return bytes.NewReader(f32Raw), nil
		},
	)
	if err == nil {
		t.Fatal("expected cancellation error")
	}
	if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
		t.Fatalf("partial file still present: %v", statErr)
	}
}

func TestWriteV3Validation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "invalid.gguf")
	cases := []struct {
		name    string
		meta    []MetadataEntry
		tensors []TensorSpec
	}{
		{
			name: "duplicate metadata key",
			meta: []MetadataEntry{{Key: "x", Value: "a"}, {Key: "x", Value: "b"}},
		},
		{
			name:    "duplicate tensor name",
			tensors: []TensorSpec{{Name: "dup", Shape: []uint64{4}, QType: QuantF32}, {Name: "dup", Shape: []uint64{4}, QType: QuantF32}},
		},
		{
			name:    "q8 shape not block aligned",
			tensors: []TensorSpec{{Name: "q8", Shape: []uint64{31}, QType: QuantQ8_0}},
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			called := false
			err := WriteV3(context.Background(), path, tc.meta, tc.tensors, func(context.Context, int, TensorSpec) (io.Reader, error) {
				called = true
				return bytes.NewReader(nil), nil
			})
			if err == nil {
				t.Fatal("expected error")
			}
			if called {
				t.Fatal("reader callback should not be called for invalid input")
			}
		})
	}
}

func encodeF32Raw(vals []float32) []byte {
	raw := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(raw[i*4:], math32bits(v))
	}
	return raw
}

func encodeF16Raw(vals []float32) []byte {
	raw := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(raw[i*2:], half.F32ToF16(v))
	}
	return raw
}

func encodeQ8Raw(t testing.TB, vals []float32) []byte {
	t.Helper()
	blocks, err := QuantizeQ8_0(vals)
	if err != nil {
		t.Fatalf("QuantizeQ8_0: %v", err)
	}
	raw := make([]byte, len(blocks)*34)
	for i, blk := range blocks {
		off := i * 34
		binary.LittleEndian.PutUint16(raw[off:off+2], half.F32ToF16(blk.d))
		for j, q := range blk.qs {
			raw[off+2+j] = byte(q)
		}
	}
	return raw
}

func math32bits(v float32) uint32 { return math.Float32bits(v) }

func TestWriteV3ExclusiveCreate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "exists.gguf")
	want := []byte("keep")
	if err := os.WriteFile(path, want, 0o644); err != nil {
		t.Fatalf("seed file: %v", err)
	}
	err := WriteTensors(context.Background(), path, nil, []ExportTensor{{
		Name:  "x",
		Shape: []uint64{4},
		QType: QuantF32,
		Raw:   bytes.NewReader(encodeF32Raw([]float32{1, 2, 3, 4})),
	}})
	if err == nil {
		t.Fatal("expected create error")
	}
	got, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatalf("read existing file: %v", readErr)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("existing file changed: got %q want %q", got, want)
	}
}

func TestWriteV3RejectsConflictingAlignment(t *testing.T) {
	err := WriteV3(context.Background(), filepath.Join(t.TempDir(), "bad.gguf"), []MetadataEntry{{Key: "general.alignment", Value: uint32(64)}}, nil, func(context.Context, int, TensorSpec) (io.Reader, error) { return nil, nil })
	if err == nil {
		t.Fatal("conflicting alignment accepted")
	}
}
