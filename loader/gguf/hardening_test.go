package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOpenRejectsMalformedHeaderAllocations(t *testing.T) {
	tests := []struct {
		name string
		data func() []byte
		want string
	}{
		{
			name: "metadata count cap",
			data: func() []byte {
				var buf bytes.Buffer
				writeTestGGUFHeader(&buf, 0, ggufMaxCollectionCount+1)
				return buf.Bytes()
			},
			want: "metadata count",
		},
		{
			name: "tensor count cap",
			data: func() []byte {
				var buf bytes.Buffer
				writeTestGGUFHeader(&buf, ggufMaxCollectionCount+1, 0)
				return buf.Bytes()
			},
			want: "tensor count",
		},
		{
			name: "oversized metadata key string",
			data: func() []byte {
				var buf bytes.Buffer
				writeTestGGUFHeader(&buf, 0, 1)
				writeTestU64(&buf, ggufMaxStringBytes+1)
				return buf.Bytes()
			},
			want: "string length",
		},
		{
			name: "oversized metadata value string",
			data: func() []byte {
				var buf bytes.Buffer
				writeTestGGUFHeader(&buf, 0, 1)
				writeTestString(&buf, "general.architecture")
				writeTestU32(&buf, uint32(GGUFTypeString))
				writeTestU64(&buf, ggufMaxStringBytes+1)
				return buf.Bytes()
			},
			want: "string length",
		},
		{
			name: "array count cap",
			data: func() []byte {
				var buf bytes.Buffer
				writeTestGGUFHeader(&buf, 0, 1)
				writeTestString(&buf, "tokenizer.ggml.tokens")
				writeTestU32(&buf, uint32(GGUFTypeArray))
				writeTestU32(&buf, uint32(GGUFTypeString))
				writeTestU64(&buf, ggufMaxCollectionCount+1)
				return buf.Bytes()
			},
			want: "array count",
		},
		{
			name: "nested arrays rejected",
			data: func() []byte {
				var buf bytes.Buffer
				writeTestGGUFHeader(&buf, 0, 1)
				writeTestString(&buf, "bad")
				writeTestU32(&buf, uint32(GGUFTypeArray))
				writeTestU32(&buf, uint32(GGUFTypeArray))
				writeTestU64(&buf, 1)
				return buf.Bytes()
			},
			want: "nested arrays",
		},
		{
			name: "tensor dims cap",
			data: func() []byte {
				var buf bytes.Buffer
				writeTestGGUFHeader(&buf, 1, 0)
				writeTestString(&buf, "tok_embeddings.weight")
				writeTestU32(&buf, ggufMaxTensorDims+1)
				return buf.Bytes()
			},
			want: "rank",
		},
		{
			name: "duplicate metadata key",
			data: func() []byte {
				var buf bytes.Buffer
				writeTestGGUFHeader(&buf, 0, 2)
				writeTestString(&buf, "general.architecture")
				writeTestU32(&buf, uint32(GGUFTypeString))
				writeTestString(&buf, "llama")
				writeTestString(&buf, "general.architecture")
				writeTestU32(&buf, uint32(GGUFTypeString))
				writeTestString(&buf, "qwen")
				return buf.Bytes()
			},
			want: "duplicate metadata key",
		},
		{
			name: "truncated metadata value",
			data: func() []byte {
				var buf bytes.Buffer
				writeTestGGUFHeader(&buf, 0, 1)
				writeTestString(&buf, "general.architecture")
				writeTestU32(&buf, uint32(GGUFTypeString))
				writeTestU64(&buf, 5)
				buf.WriteString("ll")
				return buf.Bytes()
			},
			want: "unexpected EOF",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := writeMalformedGGUF(t, tc.data())
			_, err := Open(path)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err, tc.want)
			}
		})
	}
}

func TestOpenRejectsTensorSpansPastEOF(t *testing.T) {
	tests := []struct {
		name string
		off  uint64
		data int
		want string
	}{
		{
			name: "short q8 payload",
			off:  0,
			data: 33,
			want: "exceeds GGUF data length",
		},
		{
			name: "overflowing absolute offset",
			off:  math.MaxUint64 - 15,
			data: 0,
			want: "absolute offset overflows",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			writeTestGGUFHeader(&buf, 1, 0)
			writeTestString(&buf, "tok_embeddings.weight")
			writeTestU32(&buf, 1)
			writeTestU64(&buf, 32)
			writeTestU32(&buf, uint32(QuantQ8_0))
			writeTestU64(&buf, tc.off)
			padToAlignment(&buf, 32)
			buf.Write(make([]byte, tc.data))

			path := writeMalformedGGUF(t, buf.Bytes())
			_, err := Open(path)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err, tc.want)
			}
		})
	}
}

func TestReaderReserveRejectsOversizedHeaderBudget(t *testing.T) {
	r := &reader{fileSize: ggufMaxHeaderBytes + 16, pos: ggufMaxHeaderBytes - 4, headerBytes: ggufMaxHeaderBytes - 4}
	if err := r.reserve(8); err == nil || !strings.Contains(err.Error(), "header exceeds limit") {
		t.Fatalf("reserve err=%v", err)
	}
}

func TestTensorReadHardening(t *testing.T) {
	path := filepath.Join(t.TempDir(), "raw.bin")
	if err := os.WriteFile(path, make([]byte, 34), 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open temp file: %v", err)
	}
	defer f.Close()

	g := &GGUF{f: f, fileSize: 34}

	if _, err := TensorRawBytes(QuantQ8_0, 31); err == nil || !strings.Contains(err.Error(), "multiple of 32") {
		t.Fatalf("TensorRawBytes short block err=%v", err)
	}
	if _, err := TensorRawBytes(QuantF32, ggufIntLimit()/4+1); err == nil || !strings.Contains(err.Error(), "exceeds int") {
		t.Fatalf("TensorRawBytes overflow err=%v", err)
	}

	if raw, err := g.Raw(TensorInfo{Name: "ok", Shape: []uint64{32}, QType: QuantQ8_0}); err != nil {
		t.Fatalf("Raw valid: %v", err)
	} else if len(raw) != 34 {
		t.Fatalf("Raw len=%d want 34", len(raw))
	}
	if vals, err := g.DequantF32(TensorInfo{Name: "ok", Shape: []uint64{32}, QType: QuantQ8_0}); err != nil {
		t.Fatalf("DequantF32 valid: %v", err)
	} else if len(vals) != 32 {
		t.Fatalf("DequantF32 len=%d want 32", len(vals))
	}

	if _, err := g.Raw(TensorInfo{Name: "shape-overflow", Shape: []uint64{1 << 63, 3}, QType: QuantF32}); err == nil || !strings.Contains(err.Error(), "overflows element count") {
		t.Fatalf("Raw shape overflow err=%v", err)
	}
	if _, err := g.Raw(TensorInfo{Name: "past-eof", Shape: []uint64{32}, QType: QuantQ8_0, Offset: 1}); err == nil || !strings.Contains(err.Error(), "exceeds GGUF data length") {
		t.Fatalf("Raw past eof err=%v", err)
	}
	g.DataOffset = 1
	if _, err := g.DequantF32(TensorInfo{Name: "offset-overflow", Shape: []uint64{32}, QType: QuantQ8_0, Offset: math.MaxUint64}); err == nil || !strings.Contains(err.Error(), "absolute offset overflows") {
		t.Fatalf("DequantF32 offset overflow err=%v", err)
	}
}

func TestMetaUint32RangeChecks(t *testing.T) {
	g := &GGUF{Meta: map[string]any{
		"u32":      uint32(7),
		"u64":      uint64(9),
		"int":      int(11),
		"int64":    int64(13),
		"negative": int32(-1),
		"too-big":  uint64(math.MaxUint32) + 1,
	}}

	for key, want := range map[string]uint32{
		"u32":   7,
		"u64":   9,
		"int":   11,
		"int64": 13,
	} {
		got, ok := g.MetaUint32(key)
		if !ok || got != want {
			t.Fatalf("MetaUint32(%q) = %d, %v want %d, true", key, got, ok, want)
		}
	}
	if got, ok := g.MetaUint32("negative"); ok || got != 0 {
		t.Fatalf("MetaUint32 negative = %d, %v want 0, false", got, ok)
	}
	if got, ok := g.MetaUint32("too-big"); ok || got != 0 {
		t.Fatalf("MetaUint32 too-big = %d, %v want 0, false", got, ok)
	}
}

func writeMalformedGGUF(t testing.TB, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "bad.gguf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write malformed gguf: %v", err)
	}
	return path
}

func writeTestGGUFHeader(buf *bytes.Buffer, nTensors, nKV uint64) {
	buf.WriteString("GGUF")
	writeTestU32(buf, 3)
	writeTestU64(buf, nTensors)
	writeTestU64(buf, nKV)
}

func writeTestU32(buf *bytes.Buffer, v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	buf.Write(b[:])
}

func writeTestU64(buf *bytes.Buffer, v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	buf.Write(b[:])
}

func writeTestString(buf *bytes.Buffer, s string) {
	writeTestU64(buf, uint64(len(s)))
	buf.WriteString(s)
}

func padToAlignment(buf *bytes.Buffer, align int) {
	if align <= 0 {
		return
	}
	pad := align - (buf.Len() % align)
	if pad == align {
		return
	}
	buf.Write(make([]byte, pad))
}
