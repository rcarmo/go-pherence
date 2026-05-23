// Package gguf implements a minimal GGUF v2/v3 reader that dequantizes
// tensor data to []float32.  It supports the quant types present in
// TinyLlama 1.1B: F32, F16, Q2_K, Q3_K, Q6_K, Q8_0.
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// GGUFType identifies the data type of a metadata value.
type GGUFType uint32

const (
	GGUFTypeU8     GGUFType = 0
	GGUFTypeI8     GGUFType = 1
	GGUFTypeU16    GGUFType = 2
	GGUFTypeI16    GGUFType = 3
	GGUFTypeU32    GGUFType = 4
	GGUFTypeI32    GGUFType = 5
	GGUFTypeF32    GGUFType = 6
	GGUFTypeBool   GGUFType = 7
	GGUFTypeString GGUFType = 8
	GGUFTypeArray  GGUFType = 9
	GGUFTypeU64    GGUFType = 10
	GGUFTypeI64    GGUFType = 11
	GGUFTypeF64    GGUFType = 12
)

// QuantType identifies the tensor quantization format.
type QuantType uint32

const (
	QuantF32  QuantType = 0
	QuantF16  QuantType = 1
	QuantQ4_0 QuantType = 2
	QuantQ4_1 QuantType = 3
	QuantQ8_0 QuantType = 8
	QuantQ2_K QuantType = 10
	QuantQ3_K QuantType = 11
	QuantQ4_K QuantType = 12
	QuantQ5_K QuantType = 13
	QuantQ6_K QuantType = 14
	QuantQ8_K QuantType = 15
)

// TensorInfo holds the index entry for one tensor.
type TensorInfo struct {
	Name   string
	Shape  []uint64 // innermost dimension first
	QType  QuantType
	Offset uint64 // offset from DataOffset, not from start of file
}

// GGUF is an open GGUF file.
type GGUF struct {
	Meta       map[string]any
	Tensors    []TensorInfo
	DataOffset int64 // byte offset in the file where tensor data begins
	f          *os.File
}

// Open reads the GGUF header, metadata, and tensor index.
// The file handle is kept open until Close().
func Open(path string) (*GGUF, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	r := &reader{r: f}

	// Magic
	magic := make([]byte, 4)
	if _, err := io.ReadFull(f, magic); err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if string(magic) != "GGUF" {
		f.Close()
		return nil, fmt.Errorf("gguf: bad magic %q", magic)
	}

	// Version
	version, err := r.u32()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: read version: %w", err)
	}
	if version != 2 && version != 3 {
		f.Close()
		return nil, fmt.Errorf("gguf: unsupported version %d", version)
	}

	nTensors, err := r.u64()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: n_tensors: %w", err)
	}
	nKV, err := r.u64()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: n_kv: %w", err)
	}

	// Metadata key-value pairs
	meta := make(map[string]any, nKV)
	for i := uint64(0); i < nKV; i++ {
		key, err := r.str()
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: kv[%d] key: %w", i, err)
		}
		vtype, err := r.u32()
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: kv[%d] type: %w", i, err)
		}
		val, err := r.value(GGUFType(vtype))
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: kv[%d] %q value: %w", i, key, err)
		}
		meta[key] = val
	}

	// Tensor info
	tensors := make([]TensorInfo, nTensors)
	for i := uint64(0); i < nTensors; i++ {
		name, err := r.str()
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: tensor[%d] name: %w", i, err)
		}
		ndims, err := r.u32()
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: tensor[%d] ndims: %w", i, err)
		}
		shape := make([]uint64, ndims)
		for d := uint32(0); d < ndims; d++ {
			shape[d], err = r.u64()
			if err != nil {
				f.Close()
				return nil, fmt.Errorf("gguf: tensor[%d] dim[%d]: %w", i, d, err)
			}
		}
		qtype, err := r.u32()
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: tensor[%d] qtype: %w", i, err)
		}
		offset, err := r.u64()
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("gguf: tensor[%d] offset: %w", i, err)
		}
		tensors[i] = TensorInfo{
			Name:   name,
			Shape:  shape,
			QType:  QuantType(qtype),
			Offset: offset,
		}
	}

	// Data offset is aligned to 32 bytes
	pos, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: seek current: %w", err)
	}
	dataOffset := ((pos + 31) / 32) * 32

	return &GGUF{
		Meta:       meta,
		Tensors:    tensors,
		DataOffset: dataOffset,
		f:          f,
	}, nil
}

// Close releases the underlying file.
func (g *GGUF) Close() { g.f.Close() }

// DequantF32 reads and dequantizes tensor t to a flat []float32.
func (g *GGUF) DequantF32(t TensorInfo) ([]float32, error) {
	n := int(TensorElements(t.Shape))
	if n == 0 {
		return nil, fmt.Errorf("gguf: tensor %q has zero elements", t.Name)
	}
	rawSize, err := TensorRawBytes(t.QType, n)
	if err != nil {
		return nil, fmt.Errorf("gguf: tensor %q: %w", t.Name, err)
	}
	if _, err := g.f.Seek(g.DataOffset+int64(t.Offset), io.SeekStart); err != nil {
		return nil, fmt.Errorf("gguf: tensor %q seek: %w", t.Name, err)
	}
	raw := make([]byte, rawSize)
	if _, err := io.ReadFull(g.f, raw); err != nil {
		return nil, fmt.Errorf("gguf: tensor %q read: %w", t.Name, err)
	}
	return dequantToF32(raw, t.QType, n)
}

// Raw reads the encoded tensor bytes without dequantizing them.
func (g *GGUF) Raw(t TensorInfo) ([]byte, error) {
	n := int(TensorElements(t.Shape))
	if n == 0 {
		return nil, fmt.Errorf("gguf: tensor %q has zero elements", t.Name)
	}
	rawSize, err := TensorRawBytes(t.QType, n)
	if err != nil {
		return nil, fmt.Errorf("gguf: tensor %q: %w", t.Name, err)
	}
	if _, err := g.f.Seek(g.DataOffset+int64(t.Offset), io.SeekStart); err != nil {
		return nil, fmt.Errorf("gguf: tensor %q seek: %w", t.Name, err)
	}
	raw := make([]byte, rawSize)
	if _, err := io.ReadFull(g.f, raw); err != nil {
		return nil, fmt.Errorf("gguf: tensor %q read: %w", t.Name, err)
	}
	return raw, nil
}

// MetaUint32 returns a uint32 metadata value, ok=false if missing or wrong type.
func (g *GGUF) MetaUint32(key string) (uint32, bool) {
	v, ok := g.Meta[key]
	if !ok {
		return 0, false
	}
	switch vv := v.(type) {
	case uint32:
		return vv, true
	case uint64:
		return uint32(vv), true
	case int32:
		return uint32(vv), true
	}
	return 0, false
}

// MetaFloat32 returns a float32/float64 metadata value.
func (g *GGUF) MetaFloat32(key string) (float32, bool) {
	v, ok := g.Meta[key]
	if !ok {
		return 0, false
	}
	switch vv := v.(type) {
	case float32:
		return vv, true
	case float64:
		return float32(vv), true
	}
	return 0, false
}

// MetaString returns a string metadata value.
func (g *GGUF) MetaString(key string) (string, bool) {
	v, ok := g.Meta[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// TensorByName returns the TensorInfo for the named tensor, or ok=false.
func (g *GGUF) TensorByName(name string) (TensorInfo, bool) {
	for _, t := range g.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return TensorInfo{}, false
}

// tensorElements returns the product of all shape dimensions.
func TensorElements(shape []uint64) uint64 {
	n := uint64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

// tensorRawBytes returns the number of raw bytes for n elements of the given quant type.
func TensorRawBytes(qt QuantType, n int) (int, error) {
	const qkK = 256
	switch qt {
	case QuantF32:
		return n * 4, nil
	case QuantF16:
		return n * 2, nil
	case QuantQ4_0:
		return (n / 32) * 18, nil
	case QuantQ4_1:
		return (n / 32) * 20, nil
	case QuantQ8_0:
		return (n / 32) * 34, nil
	case QuantQ2_K:
		return (n / qkK) * 84, nil
	case QuantQ3_K:
		return (n / qkK) * 110, nil
	case QuantQ4_K:
		return (n / qkK) * 144, nil
	case QuantQ5_K:
		return (n / qkK) * 176, nil
	case QuantQ6_K:
		return (n / qkK) * 210, nil
	case QuantQ8_K:
		return (n / qkK) * 292, nil
	default:
		return 0, fmt.Errorf("unsupported quant type %d", qt)
	}
}

// ── low-level binary reader ───────────────────────────────────────────────────

type reader struct{ r *os.File }

func (r *reader) u8() (uint8, error) {
	var buf [1]byte
	_, err := io.ReadFull(r.r, buf[:])
	return buf[0], err
}
func (r *reader) u16() (uint16, error) {
	var buf [2]byte
	_, err := io.ReadFull(r.r, buf[:])
	return binary.LittleEndian.Uint16(buf[:]), err
}
func (r *reader) u32() (uint32, error) {
	var buf [4]byte
	_, err := io.ReadFull(r.r, buf[:])
	return binary.LittleEndian.Uint32(buf[:]), err
}
func (r *reader) u64() (uint64, error) {
	var buf [8]byte
	_, err := io.ReadFull(r.r, buf[:])
	return binary.LittleEndian.Uint64(buf[:]), err
}
func (r *reader) i8() (int8, error)   { v, e := r.u8(); return int8(v), e }
func (r *reader) i16() (int16, error) { v, e := r.u16(); return int16(v), e }
func (r *reader) i32() (int32, error) { v, e := r.u32(); return int32(v), e }
func (r *reader) i64() (int64, error) { v, e := r.u64(); return int64(v), e }
func (r *reader) f32() (float32, error) {
	v, e := r.u32()
	if e != nil {
		return 0, e
	}
	var f [4]byte
	binary.LittleEndian.PutUint32(f[:], v)
	return math.Float32frombits(binary.LittleEndian.Uint32(f[:])), e
}
func (r *reader) f64() (float64, error) {
	var buf [8]byte
	_, err := io.ReadFull(r.r, buf[:])
	if err != nil {
		return 0, err
	}
	bits := binary.LittleEndian.Uint64(buf[:])
	return math.Float64frombits(bits), nil
}
func (r *reader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	buf := make([]byte, n)
	_, err = io.ReadFull(r.r, buf)
	return string(buf), err
}

func (r *reader) value(t GGUFType) (any, error) {
	switch t {
	case GGUFTypeU8:
		return r.u8()
	case GGUFTypeI8:
		return r.i8()
	case GGUFTypeU16:
		return r.u16()
	case GGUFTypeI16:
		return r.i16()
	case GGUFTypeU32:
		return r.u32()
	case GGUFTypeI32:
		return r.i32()
	case GGUFTypeF32:
		return r.f32()
	case GGUFTypeBool:
		v, err := r.u8()
		return v != 0, err
	case GGUFTypeString:
		return r.str()
	case GGUFTypeU64:
		return r.u64()
	case GGUFTypeI64:
		return r.i64()
	case GGUFTypeF64:
		return r.f64()
	case GGUFTypeArray:
		elemType, err := r.u32()
		if err != nil {
			return nil, err
		}
		count, err := r.u64()
		if err != nil {
			return nil, err
		}
		arr := make([]any, count)
		for i := uint64(0); i < count; i++ {
			arr[i], err = r.value(GGUFType(elemType))
			if err != nil {
				return nil, fmt.Errorf("array[%d]: %w", i, err)
			}
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unknown GGUFType %d", t)
	}
}
