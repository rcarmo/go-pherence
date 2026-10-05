package whisperggml

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func fixture(kind int) []byte {
	var b bytes.Buffer
	put := func(v int) { binary.Write(&b, binary.LittleEndian, int32(v)) }
	for _, v := range []int{0x67676d6c, 4, 2, 32, 4, 1, 8, 32, 4, 1, 1, 0, 1, 201} {
		put(v)
	}
	for i := 0; i < 201; i++ {
		binary.Write(&b, binary.LittleEndian, float32(i)/201)
	}
	put(4)
	for _, s := range []string{"a", "b", "c", "d"} {
		put(len(s))
		b.WriteString(s)
	}
	put(2)
	put(len("encoder.test"))
	put(kind)
	put(32)
	put(1)
	b.WriteString("encoder.test")
	switch kind {
	case 0:
		for i := 0; i < 32; i++ {
			binary.Write(&b, binary.LittleEndian, float32(i-16))
		}
	case 1:
		for i := 0; i < 32; i++ {
			binary.Write(&b, binary.LittleEndian, uint16(0x3c00))
		}
	case 6:
		b.Write([]byte{0x00, 0x3c, 0xff, 0xff, 0xff, 0xff})
		for i := 0; i < 16; i++ {
			b.WriteByte(byte(i | (15-i)<<4))
		}
	}
	return b.Bytes()
}
func TestLegacyWhisperGGMLPinnedTensorStorage(t *testing.T) {
	for _, kind := range []int{0, 1, 6} {
		data := fixture(kind)
		sum := sha256.Sum256(data)
		p := filepath.Join(t.TempDir(), "model.bin")
		if err := os.WriteFile(p, data, 0600); err != nil {
			t.Fatal(err)
		}
		f, err := Open(context.Background(), p, hex.EncodeToString(sum[:]))
		if err != nil {
			t.Fatal(err)
		}
		if f.Header().AudioState != 32 || f.SHA256() != hex.EncodeToString(sum[:]) {
			t.Fatal("header pin")
		}
		meta := f.Tensors()
		if len(meta) != 1 || meta[0].Type != kind || !reflect.DeepEqual(meta[0].Shape, []int{32, 1}) {
			t.Fatal(meta)
		}
		meta[0].Shape[0] = 99
		if f.Tensors()[0].Shape[0] != 32 {
			t.Fatal("metadata alias")
		}
		vocab := f.Vocabulary()
		vocab[0][0] = 'z'
		if string(f.Vocabulary()[0]) != "a" {
			t.Fatal("vocab alias")
		}
		filters := f.Filters()
		filters[1] = 9
		if f.Filters()[1] == 9 {
			t.Fatal("filter alias")
		}
		values, err := f.Float32(context.Background(), "encoder.test")
		if err != nil || len(values) != 32 {
			t.Fatal(values, err)
		}
		for i, v := range values {
			want := float32(i - 16)
			if kind == 1 {
				want = 1
			}
			if kind == 6 {
				if i < 16 {
					want = float32(i)
				} else {
					want = float32(31 - i)
				}
			}
			if v != want {
				t.Fatal("scalar known block", kind, i, v, want)
			}
		}
		if _, err := f.Float32(nil, "encoder.test"); err == nil {
			t.Fatal("nilctx")
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := f.Float32(ctx, "encoder.test"); !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
		if _, err := f.Float32(context.Background(), "missing"); err == nil {
			t.Fatal("missing")
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := f.Float32(context.Background(), "encoder.test"); !errors.Is(err, os.ErrClosed) {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := Open(context.Background(), p, "bad"); err == nil {
			t.Fatal("badpin")
		}
		if _, err := Open(context.Background(), p, string(bytes.Repeat([]byte{'0'}, 64))); err == nil {
			t.Fatal("wronghash")
		}
		if _, err := Open(nil, p, hex.EncodeToString(sum[:])); err == nil {
			t.Fatal("nilctx")
		}
	}
}
func TestLegacyWhisperGGMLRejectMalformed(t *testing.T) {
	data := fixture(6)
	for _, b := range [][]byte{nil, data[:47], append(append([]byte(nil), data...), 1), fixture(4), data[:len(data)-1]} {
		if _, err := parse(context.Background(), bytes.NewReader(b), int64(len(b))); err == nil {
			t.Fatal("malformed")
		}
	}
	for _, tc := range []struct {
		off   int
		value uint32
	}{{0, 0}, {4, 100001}, {8, 4097}, {12, 0}, {16, 0}, {20, 65}, {28, 33}, {40, 129}, {48, 2}, {52, 200}} {
		b := append([]byte(nil), data...)
		binary.LittleEndian.PutUint32(b[tc.off:], tc.value)
		if _, err := parse(context.Background(), bytes.NewReader(b), int64(len(b))); err == nil {
			t.Fatal("header", tc)
		}
	}
	b := fixture(0)
	binary.LittleEndian.PutUint32(b[56:], math.Float32bits(float32(math.NaN())))
	if _, err := parse(context.Background(), bytes.NewReader(b), int64(len(b))); err == nil {
		t.Fatal("nonfinite filter")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := parse(ctx, bytes.NewReader(data), int64(len(data))); !errors.Is(err, context.Canceled) {
		t.Fatal("cancel", err)
	}
}
func TestPinnedLegacyWhisperGGMLMetadata(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_WHISPER_GGML")
	if path == "" {
		t.Skip("pinned retained model opt-in")
	}
	f, err := Open(context.Background(), path, os.Getenv("GO_PHERENCE_WHISPER_GGML_SHA256"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	t.Logf("header=%+v tensors=%d", f.Header(), len(f.Tensors()))
}

func TestLegacyWhisperTensorAdmissionAndCancellation(t *testing.T) {
	raw := fixture(6)
	const tensorHeader = 48 + 8 + 201*4 + 4 + 4*5
	for _, tc := range []struct {
		offset int
		value  uint32
	}{{tensorHeader, 4}, {tensorHeader + 4, 257}, {tensorHeader + 8, 30}, {tensorHeader + 12, 31}, {tensorHeader + 16, 0}, {44, 1234}} {
		b := append([]byte(nil), raw...)
		binary.LittleEndian.PutUint32(b[tc.offset:], tc.value)
		if _, err := parse(context.Background(), bytes.NewReader(b), int64(len(b))); err == nil {
			t.Fatal("tensor admitted", tc)
		}
	}
	dup := append(append([]byte(nil), raw...), raw[tensorHeader:]...)
	if _, err := parse(context.Background(), bytes.NewReader(dup), int64(len(dup))); err == nil {
		t.Fatal("duplicate admitted")
	}
	badname := append([]byte(nil), raw...)
	badname[tensorHeader+20] = 0
	if _, err := parse(context.Background(), bytes.NewReader(badname), int64(len(badname))); err == nil {
		t.Fatal("NUL name")
	}
	data := fixture(0)
	sum := sha256.Sum256(data)
	p := filepath.Join(t.TempDir(), "finite.bin")
	if err := os.WriteFile(p, data, 0600); err != nil {
		t.Fatal(err)
	}
	f, err := Open(context.Background(), p, hex.EncodeToString(sum[:]))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	changed := append([]byte(nil), data...)
	binary.LittleEndian.PutUint32(changed[len(changed)-128:], math.Float32bits(float32(math.NaN())))
	if err := os.WriteFile(p, changed, 0600); err != nil {
		t.Fatal(err)
	}
	if v, err := f.Float32(context.Background(), "encoder.test"); err == nil || v != nil {
		t.Fatal("nonfinite payload success")
	}
	// A source cannot change after Open; detection on read must be an error.
	if err := os.Truncate(p, 50); err != nil {
		t.Fatal(err)
	}
	if v, err := f.Float32(context.Background(), "encoder.test"); err == nil || v != nil {
		t.Fatal("truncated payload success")
	}
	if _, err := Open(context.Background(), filepath.Dir(p), hex.EncodeToString(sum[:])); err == nil {
		t.Fatal("directory")
	}
}

type cancelReader struct {
	r         *bytes.Reader
	cancel    context.CancelFunc
	at, calls int
}

func (r *cancelReader) ReadAt(b []byte, off int64) (int, error) {
	r.calls++
	n, e := r.r.ReadAt(b, off)
	if r.calls == r.at {
		r.cancel()
	}
	return n, e
}
func TestLegacyWhisperParseEveryReadCancellation(t *testing.T) {
	raw := fixture(6)
	counter := &cancelReader{r: bytes.NewReader(raw), cancel: func() {}}
	if _, err := parse(context.Background(), counter, int64(len(raw))); err != nil {
		t.Fatal(err)
	}
	for at := 1; at <= counter.calls; at++ {
		ctx, cancel := context.WithCancel(context.Background())
		r := &cancelReader{r: bytes.NewReader(raw), at: at, cancel: cancel}
		f, err := parse(ctx, r, int64(len(raw)))
		cancel()
		if f != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("cancelled parse leaked", at, f, err)
		}
	}
}
