// Package whisperggml reads legacy whisper.cpp GGML model storage. It does not
// run inference, download assets or initialise a backend. Quantisation remains
// explicit; decoded Q5_0 values are the retained model's values, not HF weights.
package whisperggml

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	gguf "github.com/rcarmo/go-pherence/loader/gguf"
	"io"
	"math"
	"os"
	"sort"
	"sync"
)

const MaxFileBytes int64 = 3 << 30
const MaxTensorElements int64 = 100000000
const MaxDecodedBytes int64 = 4 << 30

type Header struct{ Vocab, AudioContext, AudioState, AudioHeads, AudioLayers, TextContext, TextState, TextHeads, TextLayers, Mels, FileType int }
type Tensor struct {
	Name                    string
	Shape                   []int
	Type                    int
	Offset, Bytes, Elements int64
}

// File holds one immutable pinned inode. Calls are serialised with Close; no
// partial tensor is returned after cancellation. Metadata has no borrowed slices.
type File struct {
	mu      sync.Mutex
	file    *os.File
	header  Header
	filters []float32
	vocab   [][]byte
	tensors map[string]Tensor
	sha     string
	size    int64
}

func Open(ctx context.Context, path, pin string) (*File, error) {
	if ctx == nil {
		return nil, fmt.Errorf("WhisperGGML: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	digest, err := hex.DecodeString(pin)
	if err != nil || len(digest) != 32 || pin != hex.EncodeToString(digest) {
		return nil, fmt.Errorf("WhisperGGML: require lowercase SHA256")
	}
	stat, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !stat.Mode().IsRegular() {
		return nil, fmt.Errorf("WhisperGGML: require regular file")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	good := false
	defer func() {
		if !good {
			f.Close()
		}
	}()
	info, err := f.Stat()
	if err != nil || info.Size() < 48 || info.Size() > MaxFileBytes {
		return nil, fmt.Errorf("WhisperGGML: file bound")
	}
	hash := sha256.New()
	buf := make([]byte, 65536)
	for offset := int64(0); offset < info.Size(); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n := int64(len(buf))
		if n > info.Size()-offset {
			n = info.Size() - offset
		}
		got, err := f.ReadAt(buf[:n], offset)
		if err != nil || int64(got) != n {
			return nil, fmt.Errorf("WhisperGGML: hash read: %w", err)
		}
		hash.Write(buf[:n])
		offset += n
	}
	if hex.EncodeToString(hash.Sum(nil)) != pin {
		return nil, fmt.Errorf("WhisperGGML: SHA256 mismatch")
	}
	meta, err := parse(ctx, f, info.Size())
	if err != nil {
		return nil, err
	}
	meta.file = f
	meta.sha = pin
	meta.size = info.Size()
	good = true
	return meta, nil
}
func (f *File) Close() error {
	if f == nil {
		return nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.file == nil {
		return nil
	}
	err := f.file.Close()
	f.file = nil
	return err
}
func (f *File) Header() Header     { return f.header }
func (f *File) SHA256() string     { return f.sha }
func (f *File) Filters() []float32 { return append([]float32(nil), f.filters...) }
func (f *File) Vocabulary() [][]byte {
	out := make([][]byte, len(f.vocab))
	for i, b := range f.vocab {
		out[i] = append([]byte(nil), b...)
	}
	return out
}
func (f *File) Tensors() []Tensor {
	names := make([]string, 0, len(f.tensors))
	for name := range f.tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]Tensor, 0, len(names))
	for _, name := range names {
		t := f.tensors[name]
		t.Shape = append([]int(nil), t.Shape...)
		out = append(out, t)
	}
	return out
}

type cursor struct {
	r            io.ReaderAt
	offset, size int64
}

func (c *cursor) take(n int64) ([]byte, error) {
	if n < 0 || n > c.size-c.offset || n > 1<<20 {
		return nil, io.ErrUnexpectedEOF
	}
	b := make([]byte, n)
	if n == 0 {
		return b, nil
	}
	got, err := c.r.ReadAt(b, c.offset)
	if err != nil || int64(got) != n {
		return nil, io.ErrUnexpectedEOF
	}
	c.offset += n
	return b, nil
}
func (c *cursor) integer() (int, error) {
	b, e := c.take(4)
	if e != nil {
		return 0, e
	}
	return int(int32(binary.LittleEndian.Uint32(b))), nil
}
func parse(ctx context.Context, r io.ReaderAt, size int64) (*File, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if ctx == nil || r == nil || size < 48 || size > MaxFileBytes {
		return nil, fmt.Errorf("WhisperGGML: invalid input")
	}
	c := cursor{r: r, size: size}
	magic, e := c.integer()
	if e != nil || magic != 0x67676d6c {
		return nil, fmt.Errorf("WhisperGGML: bad magic")
	}
	h := make([]int, 11)
	for i := range h {
		h[i], e = c.integer()
		if e != nil {
			return nil, e
		}
	}
	header := Header{h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8], h[9], h[10]}
	if h[0] < 1 || h[0] > 100000 || h[1] < 1 || h[1] > 4096 || h[2] < 1 || h[2] > 2048 || h[3] < 1 || h[3] > 32 || h[2]%h[3] != 0 || h[4] < 1 || h[4] > 64 || h[5] < 1 || h[5] > 2048 || h[6] != h[2] || h[7] < 1 || h[7] > 32 || h[6]%h[7] != 0 || h[8] < 1 || h[8] > 64 || h[9] < 1 || h[9] > 128 || (h[10] != 0 && h[10] != 1 && h[10] != 2008) {
		return nil, fmt.Errorf("WhisperGGML: unsupported header geometry")
	}
	mels, e := c.integer()
	if e != nil {
		return nil, e
	}
	bins, e := c.integer()
	if e != nil || mels != h[9] || bins != 201 {
		return nil, fmt.Errorf("WhisperGGML: filter geometry")
	}
	b, e := c.take(int64(mels * bins * 4))
	if e != nil {
		return nil, e
	}
	filters := make([]float32, mels*bins)
	for i := range filters {
		filters[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		if !finite(filters[i]) {
			return nil, fmt.Errorf("WhisperGGML: nonfinite filter")
		}
	}
	nv, e := c.integer()
	if e != nil || nv < 1 || nv > h[0] {
		return nil, fmt.Errorf("WhisperGGML: vocabulary geometry")
	}
	vocab := make([][]byte, nv)
	var vocabBytes int64
	for i := range vocab {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		n, e := c.integer()
		if e != nil || n < 0 || n > 65536 {
			return nil, fmt.Errorf("WhisperGGML: token length")
		}
		vocabBytes += int64(n)
		if vocabBytes > 16<<20 {
			return nil, fmt.Errorf("WhisperGGML: vocabulary bound")
		}
		vocab[i], e = c.take(int64(n))
		if e != nil {
			return nil, e
		}
	}
	f := &File{header: header, filters: filters, vocab: vocab, tensors: map[string]Tensor{}}
	var decoded int64
	for c.offset < c.size {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(f.tensors) >= 2000 {
			return nil, fmt.Errorf("WhisperGGML: tensor count")
		}
		rank, e := c.integer()
		if e != nil {
			return nil, e
		}
		length, e := c.integer()
		if e != nil {
			return nil, e
		}
		kind, e := c.integer()
		if e != nil || rank < 1 || rank > 3 || length < 1 || length > 256 || (kind != 0 && kind != 1 && kind != 6) {
			return nil, fmt.Errorf("WhisperGGML: tensor header")
		}
		shape := make([]int, rank)
		elements := int64(1)
		for i := range shape {
			d, e := c.integer()
			if e != nil || d < 1 || d > 100000 || elements > MaxTensorElements/int64(d) {
				return nil, fmt.Errorf("WhisperGGML: tensor dimensions")
			}
			shape[i] = d
			elements *= int64(d)
		}
		raw, e := c.take(int64(length))
		if e != nil {
			return nil, e
		}
		name := string(raw)
		for _, v := range raw {
			if v < 33 || v > 126 {
				return nil, fmt.Errorf("WhisperGGML: tensor name")
			}
		}
		if _, exists := f.tensors[name]; exists {
			return nil, fmt.Errorf("WhisperGGML: duplicate tensor")
		}
		bytes := elements * 4
		if kind == 1 {
			bytes = elements * 2
		}
		if kind == 6 {
			if shape[0]%32 != 0 {
				return nil, fmt.Errorf("WhisperGGML: Q5 block alignment")
			}
			bytes = elements / 32 * 22
		}
		if bytes > c.size-c.offset {
			return nil, io.ErrUnexpectedEOF
		}
		decoded += elements * 4
		if decoded > MaxDecodedBytes || decoded > int64(^uint(0)>>1) {
			return nil, fmt.Errorf("WhisperGGML: decoded footprint")
		}
		f.tensors[name] = Tensor{Name: name, Shape: shape, Type: kind, Offset: c.offset, Bytes: bytes, Elements: elements}
		c.offset += bytes
	}
	if len(f.tensors) == 0 {
		return nil, fmt.Errorf("WhisperGGML: missing tensors")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return f, nil
}
func finite(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }

// Float32 decodes one tensor in bounded chunks. Q5_0 is decoded with the existing
// GGML block decoder; original dtype remains in Tensor.Type. No requantisation.
func (f *File) Float32(ctx context.Context, name string) ([]float32, error) {
	if ctx == nil || f == nil {
		return nil, fmt.Errorf("WhisperGGML: nil request")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.file == nil {
		return nil, os.ErrClosed
	}
	t, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("WhisperGGML: missing tensor %q", name)
	}
	out := make([]float32, t.Elements)
	chunk := int64(16384)
	var offset int64
	for done := int64(0); done < t.Elements; {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n := chunk
		if n > t.Elements-done {
			n = t.Elements - done
		}
		bytes := n * 4
		if t.Type == 1 {
			bytes = n * 2
		} else if t.Type == 6 {
			bytes = n / 32 * 22
		}
		raw := make([]byte, bytes)
		got, err := f.file.ReadAt(raw, t.Offset+offset)
		if err != nil || int64(got) != bytes {
			return nil, errors.Join(io.ErrUnexpectedEOF, err)
		}
		values, err := gguf.DequantToF32(raw, gguf.QuantType(t.Type), int(n))
		if err != nil {
			return nil, err
		}
		for _, v := range values {
			if !finite(v) {
				return nil, fmt.Errorf("WhisperGGML: nonfinite tensor")
			}
		}
		copy(out[done:done+n], values)
		done += n
		offset += bytes
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
