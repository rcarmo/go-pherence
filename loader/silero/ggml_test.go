package silero

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type tensorLocation struct {
	header, name, payload int
	spec                  tensorSpec
}

func syntheticFile() ([]byte, []tensorLocation) {
	var b bytes.Buffer
	put := func(v int) { _ = binary.Write(&b, binary.LittleEndian, int32(v)) }
	put(0x67676d6c)
	put(10)
	b.WriteString("silero-16k")
	for _, v := range []int{6, 2, 0, 512, 64, 4, 129, 128, 3, 128, 64, 3, 64, 64, 3, 64, 128, 3, 128, 128, 128, 1} {
		put(v)
	}
	locations := make([]tensorLocation, 0, 15)
	for _, s := range specifications() {
		loc := tensorLocation{header: b.Len(), spec: s}
		put(len(s.shape))
		put(len(s.name))
		put(s.dtype)
		n := 1
		for _, d := range s.shape {
			put(d)
			n *= d
		}
		loc.name = b.Len()
		b.WriteString(s.name)
		loc.payload = b.Len()
		size := 4
		if s.dtype == 1 {
			size = 2
		}
		b.Write(make([]byte, n*size))
		locations = append(locations, loc)
	}
	return b.Bytes(), locations
}
func setInt(b []byte, offset, value int) { binary.LittleEndian.PutUint32(b[offset:], uint32(value)) }
func TestParseSyntheticInventoryAndOwnedValues(t *testing.T) {
	b, locations := syntheticFile()
	// Independent encoded constants: binary16 1.5; binary32 -2.5, scalar bias.
	binary.LittleEndian.PutUint16(b[locations[0].payload:], 0x3e00)
	binary.LittleEndian.PutUint32(b[locations[1].payload:], 0xc0200000)
	binary.LittleEndian.PutUint32(b[locations[13].payload:], 0x3f000000)
	m, err := Parse(context.Background(), b)
	if err != nil {
		t.Fatal(err)
	}
	if m.Window != 512 || m.Context != 64 || m.Version != [3]int{6, 2, 0} || len(m.Tensors) != 15 {
		t.Fatal("header/inventory")
	}
	for _, s := range specifications() {
		v := m.Tensors[s.name]
		n := 1
		for _, d := range s.shape {
			n *= d
		}
		if len(v.Values) != n || len(v.Shape) != len(s.shape) {
			t.Fatal("tensor footprint")
		}
	}
	if m.Tensors[locations[0].spec.name].Values[0] != 1.5 || m.Tensors[locations[1].spec.name].Values[0] != -2.5 || m.Tensors[locations[13].spec.name].Values[0] != 0.5 {
		t.Fatal("decoded types/scalar")
	}
	clear(b)
	if m.Tensors[locations[0].spec.name].Values[0] != 1.5 {
		t.Fatal("returned input alias")
	}
}
func TestParseRejectsMalformedHeaderAndInventory(t *testing.T) {
	valid, locations := syntheticFile()
	tests := []struct {
		name   string
		change func([]byte) []byte
	}{
		{"magic", func(b []byte) []byte { setInt(b, 0, 0); return b }},
		{"type-length", func(b []byte) []byte { setInt(b, 4, math.MaxInt32); return b }},
		{"type", func(b []byte) []byte { b[8] = 'X'; return b }},
		{"version", func(b []byte) []byte { setInt(b, 18, 7); return b }},
		{"window", func(b []byte) []byte { setInt(b, 30, 1024); return b }},
		{"rank", func(b []byte) []byte { setInt(b, locations[0].header, 4); return b }},
		{"negative-rank", func(b []byte) []byte { setInt(b, locations[0].header, -1); return b }},
		{"name-length", func(b []byte) []byte { setInt(b, locations[0].header+4, 129); return b }},
		{"empty-name", func(b []byte) []byte { setInt(b, locations[0].header+4, 0); return b }},
		{"dtype", func(b []byte) []byte { setInt(b, locations[0].header+8, 2); return b }},
		{"wrong-dtype", func(b []byte) []byte { setInt(b, locations[0].header+8, 0); return b }},
		{"zero-dim", func(b []byte) []byte { setInt(b, locations[0].header+12, 0); return b }},
		{"overflow-dim", func(b []byte) []byte { setInt(b, locations[0].header+12, math.MaxInt32); return b }},
		{"wrong-shape", func(b []byte) []byte { setInt(b, locations[0].header+12, 2); return b }},
		{"wrong-rank", func(b []byte) []byte { setInt(b, locations[13].header, 1); return b }},
		{"unknown-tensor", func(b []byte) []byte { b[locations[0].name] = 'X'; return b }},
		{"duplicate", func(b []byte) []byte { copy(b[locations[2].name:], locations[0].spec.name); return b }},
		{"half-inf", func(b []byte) []byte { binary.LittleEndian.PutUint16(b[locations[0].payload:], 0x7c00); return b }},
		{"half-nan", func(b []byte) []byte { binary.LittleEndian.PutUint16(b[locations[0].payload:], 0x7e00); return b }},
		{"float-nan", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[locations[1].payload:], 0x7fc00000); return b }},
		{"float-inf", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[locations[1].payload:], 0xff800000); return b }},
		{"trailing", func(b []byte) []byte { return append(b, 0) }},
		{"truncated-payload", func(b []byte) []byte { return b[:len(b)-1] }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			b := test.change(append([]byte(nil), valid...))
			if m, err := Parse(context.Background(), b); err == nil || m != nil {
				t.Fatal("malformed model returned", err)
			}
		})
	}
	for _, n := range []int{0, 1, 7, 8, 18, 30, 65, locations[0].name + 5, locations[0].payload + 7} {
		if m, err := Parse(context.Background(), valid[:n]); err == nil || m != nil {
			t.Fatal("truncation admitted", n)
		}
	}
	if m, err := Parse(context.Background(), make([]byte, MaxFileBytes+1)); err == nil || m != nil {
		t.Fatal("oversized file")
	}
	if _, err := Parse(nil, valid); err == nil {
		t.Fatal("nil context")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Parse(ctx, valid); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
func TestLoadPathPinsAndSize(t *testing.T) {
	valid, _ := syntheticFile()
	path := filepath.Join(t.TempDir(), "vad.bin")
	if err := os.WriteFile(path, valid, 0600); err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(valid)
	pin := hex.EncodeToString(hash[:])
	if m, err := LoadPath(context.Background(), path, pin); err != nil || len(m.Tensors) != 15 {
		t.Fatal(err)
	}
	for _, bad := range []string{"", "bad", string(bytes.ToUpper([]byte(pin))), RevisionSHA256} {
		if _, err := LoadPath(context.Background(), path, bad); err == nil {
			t.Fatal("invalid/mismatched pin")
		}
	}
	if _, err := LoadPath(nil, path, pin); err == nil {
		t.Fatal("nil context")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := LoadPath(ctx, path, pin); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := LoadPath(context.Background(), path+"-absent", pin); err == nil {
		t.Fatal("missing file")
	}
	if err := os.WriteFile(path, make([]byte, MaxFileBytes+1), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPath(context.Background(), path, pin); err == nil {
		t.Fatal("oversized file")
	}
}
func TestParseEveryStructuralPrefixAndMidDecodeCancellation(t *testing.T) {
	valid, locations := syntheticFile()
	for _, location := range locations {
		for _, offset := range []int{location.header, location.header + 4, location.header + 8, location.header + 10, location.name - 1, location.name + 1} {
			if m, err := Parse(context.Background(), valid[:offset]); err == nil || m != nil {
				t.Fatal("structural truncation admitted", offset)
			}
		}
	}
	ctx := &countedCancellation{at: 7}
	if m, err := Parse(ctx, valid); !errors.Is(err, context.Canceled) || m != nil {
		t.Fatal("mid-decode cancellation", err)
	}
}

type countedCancellation struct{ calls, at int }

func (c *countedCancellation) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *countedCancellation) Done() <-chan struct{}       { return nil }
func (c *countedCancellation) Value(any) any               { return nil }
func (c *countedCancellation) Err() error {
	c.calls++
	if c.calls >= c.at {
		return context.Canceled
	}
	return nil
}

func TestCursorBounds(t *testing.T) {
	r := cursor{data: []byte{1, 2, 3}}
	if _, err := r.take(-1); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal(err)
	}
	if _, err := r.integer(); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal(err)
	}
}
func FuzzParseBounded(f *testing.F) {
	valid, _ := syntheticFile()
	f.Add(valid[:64])
	f.Add([]byte("ggml"))
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > MaxFileBytes {
			t.Skip()
		}
		_, _ = Parse(context.Background(), b)
	})
}

// Metadata/decode gate only: no backend initialisation or neural inference.
func TestPinnedSileroFile(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_SILERO_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_SILERO_MODEL for pinned 6.2.0 file decode")
	}
	model, err := LoadPath(context.Background(), path, RevisionSHA256)
	if err != nil {
		t.Fatal(err)
	}
	if len(model.Tensors) != 15 || len(model.Tensors["_model.stft.forward_basis_buffer"].Values) != 256*258 {
		t.Fatal("pinned model footprint")
	}
	t.Logf("pinned Silero6.2.0 window=%d context=%d tensors=%d", model.Window, model.Context, len(model.Tensors))
}
