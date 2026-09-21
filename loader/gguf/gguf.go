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

const (
	ggufMaxHeaderBytes     int64  = 128 << 20
	ggufMaxStringBytes     uint64 = 16 << 20
	ggufMaxCollectionCount uint64 = 1_000_000
	ggufMaxTensorDims      uint32 = 8
	ggufMaxArrayDepth             = 1
)

// QuantType identifies the tensor quantization format.
type QuantType uint32

const (
	QuantF32  QuantType = 0
	QuantF16  QuantType = 1
	QuantQ4_0 QuantType = 2
	QuantQ4_1 QuantType = 3
	QuantQ5_0 QuantType = 6
	QuantQ5_1 QuantType = 7
	QuantQ8_0 QuantType = 8
	QuantQ2_K QuantType = 10
	QuantQ3_K QuantType = 11
	QuantQ4_K QuantType = 12
	QuantQ5_K QuantType = 13
	QuantQ6_K QuantType = 14
	QuantQ8_K QuantType = 15
	QuantBF16 QuantType = 30
)

// String returns a human-readable quant type name.
func (qt QuantType) String() string {
	switch qt {
	case QuantF32:
		return "F32"
	case QuantF16:
		return "F16"
	case QuantQ4_0:
		return "Q4_0"
	case QuantQ4_1:
		return "Q4_1"
	case QuantQ5_0:
		return "Q5_0"
	case QuantQ5_1:
		return "Q5_1"
	case QuantQ8_0:
		return "Q8_0"
	case QuantQ2_K:
		return "Q2_K"
	case QuantQ3_K:
		return "Q3_K"
	case QuantQ4_K:
		return "Q4_K"
	case QuantQ5_K:
		return "Q5_K"
	case QuantQ6_K:
		return "Q6_K"
	case QuantQ8_K:
		return "Q8_K"
	case QuantBF16:
		return "BF16"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", qt)
	}
}

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
	fileSize   int64
}

// Open reads the GGUF header, metadata, and tensor index.
// The file handle is kept open until Close().
func Open(path string) (result *GGUF, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	// Use the returned error, including semantic validation errors constructed
	// below. A local err can remain nil when rejecting a successfully read header.
	defer func() {
		if err != nil {
			f.Close()
		}
	}()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("gguf: stat: %w", err)
	}
	if fi.Size() < 0 {
		return nil, fmt.Errorf("gguf: negative file size %d", fi.Size())
	}
	r := &reader{r: f, fileSize: fi.Size()}

	// Magic
	magic, err := r.bytes(4)
	if err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if string(magic) != "GGUF" {
		return nil, fmt.Errorf("gguf: bad magic %q", magic)
	}

	// Version
	version, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("gguf: read version: %w", err)
	}
	if version != 2 && version != 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", version)
	}

	nTensors, err := r.u64()
	if err != nil {
		return nil, fmt.Errorf("gguf: n_tensors: %w", err)
	}
	if nTensors > ggufMaxCollectionCount {
		return nil, fmt.Errorf("gguf: tensor count %d exceeds limit %d", nTensors, ggufMaxCollectionCount)
	}
	nKV, err := r.u64()
	if err != nil {
		return nil, fmt.Errorf("gguf: n_kv: %w", err)
	}
	if nKV > ggufMaxCollectionCount {
		return nil, fmt.Errorf("gguf: metadata count %d exceeds limit %d", nKV, ggufMaxCollectionCount)
	}

	metaCap, err := ggufU64ToInt(nKV, "metadata count")
	if err != nil {
		return nil, err
	}
	tensorCap, err := ggufU64ToInt(nTensors, "tensor count")
	if err != nil {
		return nil, err
	}

	// Metadata key-value pairs
	meta := make(map[string]any, metaCap)
	for i := uint64(0); i < nKV; i++ {
		key, err := r.str()
		if err != nil {
			return nil, fmt.Errorf("gguf: kv[%d] key: %w", i, err)
		}
		if _, dup := meta[key]; dup {
			return nil, fmt.Errorf("gguf: duplicate metadata key %q", key)
		}
		vtype, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("gguf: kv[%d] type: %w", i, err)
		}
		val, err := r.value(GGUFType(vtype))
		if err != nil {
			return nil, fmt.Errorf("gguf: kv[%d] %q value: %w", i, key, err)
		}
		meta[key] = val
	}

	// Tensor info
	tensors := make([]TensorInfo, tensorCap)
	for i := uint64(0); i < nTensors; i++ {
		name, err := r.str()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor[%d] name: %w", i, err)
		}
		ndims, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor[%d] ndims: %w", i, err)
		}
		if ndims == 0 || ndims > ggufMaxTensorDims {
			return nil, fmt.Errorf("gguf: tensor[%d] %q rank %d exceeds limit %d", i, name, ndims, ggufMaxTensorDims)
		}
		shape := make([]uint64, int(ndims))
		for d := uint32(0); d < ndims; d++ {
			shape[d], err = r.u64()
			if err != nil {
				return nil, fmt.Errorf("gguf: tensor[%d] dim[%d]: %w", i, d, err)
			}
		}
		qtype, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor[%d] qtype: %w", i, err)
		}
		offset, err := r.u64()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor[%d] offset: %w", i, err)
		}
		tensors[i] = TensorInfo{
			Name:   name,
			Shape:  shape,
			QType:  QuantType(qtype),
			Offset: offset,
		}
	}

	// Data offset is aligned to 32 bytes.
	dataOffsetU64, err := alignUp(uint64(r.pos), 32)
	if err != nil {
		return nil, fmt.Errorf("gguf: align data offset: %w", err)
	}
	if dataOffsetU64 > uint64(fi.Size()) {
		return nil, fmt.Errorf("gguf: data offset %d exceeds file size %d", dataOffsetU64, fi.Size())
	}
	dataOffset, err := ggufU64ToInt64(dataOffsetU64, "data offset")
	if err != nil {
		return nil, err
	}

	for i, t := range tensors {
		_, rawBytes, err := tensorEncodingSize(t.QType, t.Shape)
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor[%d] %q: %w", i, t.Name, err)
		}
		start, ok := ggufCheckedAdd(dataOffsetU64, t.Offset)
		if !ok {
			return nil, fmt.Errorf("gguf: tensor[%d] %q absolute offset overflows", i, t.Name)
		}
		end, ok := ggufCheckedAdd(start, rawBytes)
		if !ok {
			return nil, fmt.Errorf("gguf: tensor[%d] %q span overflows", i, t.Name)
		}
		if end > uint64(fi.Size()) {
			return nil, fmt.Errorf("gguf: tensor[%d] %q span [%d,%d) exceeds GGUF data length (file size %d)", i, t.Name, start, end, fi.Size())
		}
	}

	return &GGUF{
		Meta:       meta,
		Tensors:    tensors,
		DataOffset: dataOffset,
		f:          f,
		fileSize:   fi.Size(),
	}, nil
}

// Close releases the underlying file.
func (g *GGUF) Close() { g.f.Close() }

// DequantF32 reads and dequantizes tensor t to a flat []float32.
func (g *GGUF) DequantF32(t TensorInfo) ([]float32, error) {
	n, rawSize, fileOffset, err := g.tensorReadPlan(t)
	if err != nil {
		return nil, err
	}
	raw := make([]byte, rawSize)
	if _, err := g.f.ReadAt(raw, fileOffset); err != nil {
		return nil, fmt.Errorf("gguf: tensor %q read: %w", t.Name, err)
	}
	return dequantToF32(raw, t.QType, n)
}

// Raw reads the encoded tensor bytes without dequantizing them.
func (g *GGUF) Raw(t TensorInfo) ([]byte, error) {
	_, rawSize, fileOffset, err := g.tensorReadPlan(t)
	if err != nil {
		return nil, err
	}
	raw := make([]byte, rawSize)
	if _, err := g.f.ReadAt(raw, fileOffset); err != nil {
		return nil, fmt.Errorf("gguf: tensor %q read: %w", t.Name, err)
	}
	return raw, nil
}

func (g *GGUF) tensorReadPlan(t TensorInfo) (int, int, int64, error) {
	if g == nil || g.f == nil {
		return 0, 0, 0, fmt.Errorf("gguf: reader is nil")
	}
	if g.DataOffset < 0 {
		return 0, 0, 0, fmt.Errorf("gguf: negative data offset %d", g.DataOffset)
	}
	if g.fileSize < 0 {
		return 0, 0, 0, fmt.Errorf("gguf: negative file size %d", g.fileSize)
	}

	elements, rawBytes, err := tensorEncodingSize(t.QType, t.Shape)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("gguf: tensor %q: %w", t.Name, err)
	}
	if elements == 0 {
		return 0, 0, 0, fmt.Errorf("gguf: tensor %q has zero elements", t.Name)
	}
	if elements > uint64(ggufIntLimit()) {
		return 0, 0, 0, fmt.Errorf("gguf: tensor %q element count %d exceeds int", t.Name, elements)
	}
	if rawBytes > uint64(ggufIntLimit()) {
		return 0, 0, 0, fmt.Errorf("gguf: tensor %q raw byte count %d exceeds int", t.Name, rawBytes)
	}

	base := uint64(g.DataOffset)
	if base > uint64(g.fileSize) {
		return 0, 0, 0, fmt.Errorf("gguf: data offset %d exceeds file size %d", g.DataOffset, g.fileSize)
	}
	start, ok := ggufCheckedAdd(base, t.Offset)
	if !ok {
		return 0, 0, 0, fmt.Errorf("gguf: tensor %q absolute offset overflows", t.Name)
	}
	end, ok := ggufCheckedAdd(start, rawBytes)
	if !ok {
		return 0, 0, 0, fmt.Errorf("gguf: tensor %q span overflows", t.Name)
	}
	if end > uint64(g.fileSize) {
		return 0, 0, 0, fmt.Errorf("gguf: tensor %q span [%d,%d) exceeds GGUF data length (file size %d)", t.Name, start, end, g.fileSize)
	}

	return int(elements), int(rawBytes), int64(start), nil
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
		if vv <= uint64(^uint32(0)) {
			return uint32(vv), true
		}
	case uint:
		if uint64(vv) <= uint64(^uint32(0)) {
			return uint32(vv), true
		}
	case int:
		if vv >= 0 && uint64(vv) <= uint64(^uint32(0)) {
			return uint32(vv), true
		}
	case int32:
		if vv >= 0 {
			return uint32(vv), true
		}
	case int64:
		if vv >= 0 && uint64(vv) <= uint64(^uint32(0)) {
			return uint32(vv), true
		}
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
	if n < 0 {
		return 0, fmt.Errorf("negative element count %d", n)
	}
	if n == 0 {
		return 0, nil
	}

	elements := uint64(n)
	var rawBytes uint64
	var err error
	const qkK = 256
	switch qt {
	case QuantF32:
		rawBytes, err = ggufCheckedMul(elements, 4)
	case QuantF16, QuantBF16:
		rawBytes, err = ggufCheckedMul(elements, 2)
	case QuantQ4_0:
		rawBytes, err = ggufBlockRawBytes(elements, 32, 18, qt)
	case QuantQ4_1:
		rawBytes, err = ggufBlockRawBytes(elements, 32, 20, qt)
	case QuantQ5_0:
		rawBytes, err = ggufBlockRawBytes(elements, 32, 22, qt)
	case QuantQ5_1:
		rawBytes, err = ggufBlockRawBytes(elements, 32, 24, qt)
	case QuantQ8_0:
		rawBytes, err = ggufBlockRawBytes(elements, 32, 34, qt)
	case QuantQ2_K:
		rawBytes, err = ggufBlockRawBytes(elements, qkK, 84, qt)
	case QuantQ3_K:
		rawBytes, err = ggufBlockRawBytes(elements, qkK, 110, qt)
	case QuantQ4_K:
		rawBytes, err = ggufBlockRawBytes(elements, qkK, 144, qt)
	case QuantQ5_K:
		rawBytes, err = ggufBlockRawBytes(elements, qkK, 176, qt)
	case QuantQ6_K:
		rawBytes, err = ggufBlockRawBytes(elements, qkK, 210, qt)
	case QuantQ8_K:
		rawBytes, err = ggufBlockRawBytes(elements, qkK, 292, qt)
	default:
		return 0, fmt.Errorf("unsupported quant type %d", qt)
	}
	if err != nil {
		return 0, err
	}
	if rawBytes > uint64(ggufIntLimit()) {
		return 0, fmt.Errorf("raw byte count %d exceeds int", rawBytes)
	}
	return int(rawBytes), nil
}

// ── low-level binary reader ───────────────────────────────────────────────────

type reader struct {
	r           *os.File
	fileSize    int64
	pos         int64
	headerBytes int64
}

func (r *reader) bytes(n int) ([]byte, error) {
	buf := make([]byte, n)
	if err := r.readFull(buf); err != nil {
		return nil, err
	}
	return buf, nil
}

func (r *reader) readFull(buf []byte) error {
	if err := r.reserve(int64(len(buf))); err != nil {
		return err
	}
	if _, err := io.ReadFull(r.r, buf); err != nil {
		return err
	}
	r.pos += int64(len(buf))
	r.headerBytes += int64(len(buf))
	return nil
}

func (r *reader) reserve(n int64) error {
	if n < 0 {
		return fmt.Errorf("negative read size %d", n)
	}
	if n == 0 {
		return nil
	}
	if r.pos > r.fileSize-n {
		return io.ErrUnexpectedEOF
	}
	if r.headerBytes > ggufMaxHeaderBytes-n {
		return fmt.Errorf("header exceeds limit %d bytes", ggufMaxHeaderBytes)
	}
	return nil
}

func (r *reader) remainingFile() int64 { return r.fileSize - r.pos }

func (r *reader) u8() (uint8, error) {
	var buf [1]byte
	err := r.readFull(buf[:])
	return buf[0], err
}
func (r *reader) u16() (uint16, error) {
	var buf [2]byte
	err := r.readFull(buf[:])
	return binary.LittleEndian.Uint16(buf[:]), err
}
func (r *reader) u32() (uint32, error) {
	var buf [4]byte
	err := r.readFull(buf[:])
	return binary.LittleEndian.Uint32(buf[:]), err
}
func (r *reader) u64() (uint64, error) {
	var buf [8]byte
	err := r.readFull(buf[:])
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
	err := r.readFull(buf[:])
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
	if n > ggufMaxStringBytes {
		return "", fmt.Errorf("string length %d exceeds limit %d", n, ggufMaxStringBytes)
	}
	if n > uint64(r.remainingFile()) {
		return "", io.ErrUnexpectedEOF
	}
	length, err := ggufU64ToInt(n, "string length")
	if err != nil {
		return "", err
	}
	buf := make([]byte, length)
	if err := r.readFull(buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func (r *reader) value(t GGUFType) (any, error) {
	return r.valueDepth(t, 0)
}

func (r *reader) valueDepth(t GGUFType, depth int) (any, error) {
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
		if depth >= ggufMaxArrayDepth {
			return nil, fmt.Errorf("nested arrays exceed depth limit %d", ggufMaxArrayDepth)
		}
		elemType, err := r.u32()
		if err != nil {
			return nil, err
		}
		if GGUFType(elemType) == GGUFTypeArray {
			return nil, fmt.Errorf("nested arrays are unsupported")
		}
		count, err := r.u64()
		if err != nil {
			return nil, err
		}
		if count > ggufMaxCollectionCount {
			return nil, fmt.Errorf("array count %d exceeds limit %d", count, ggufMaxCollectionCount)
		}
		if minSize, ok := ggufMinValueSize(GGUFType(elemType)); ok {
			need, mulErr := ggufCheckedMul(count, minSize)
			if mulErr != nil {
				return nil, mulErr
			}
			if need > uint64(r.remainingFile()) {
				return nil, io.ErrUnexpectedEOF
			}
		}
		length, err := ggufU64ToInt(count, "array count")
		if err != nil {
			return nil, err
		}
		arr := make([]any, length)
		for i := 0; i < length; i++ {
			arr[i], err = r.valueDepth(GGUFType(elemType), depth+1)
			if err != nil {
				return nil, fmt.Errorf("array[%d]: %w", i, err)
			}
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unknown GGUFType %d", t)
	}
}

func ggufMinValueSize(t GGUFType) (uint64, bool) {
	switch t {
	case GGUFTypeU8, GGUFTypeI8, GGUFTypeBool:
		return 1, true
	case GGUFTypeU16, GGUFTypeI16:
		return 2, true
	case GGUFTypeU32, GGUFTypeI32, GGUFTypeF32:
		return 4, true
	case GGUFTypeU64, GGUFTypeI64, GGUFTypeF64, GGUFTypeString:
		return 8, true
	default:
		return 0, false
	}
}

func ggufCheckedAdd(a, b uint64) (uint64, bool) {
	if a > math.MaxUint64-b {
		return 0, false
	}
	return a + b, true
}

func ggufCheckedMul(a, b uint64) (uint64, error) {
	if a == 0 || b == 0 {
		return 0, nil
	}
	if a > math.MaxUint64/b {
		return 0, fmt.Errorf("uint64 overflow multiplying %d and %d", a, b)
	}
	return a * b, nil
}

func ggufBlockRawBytes(elements, blockElems, blockBytes uint64, qt QuantType) (uint64, error) {
	if elements%blockElems != 0 {
		return 0, fmt.Errorf("quant type %s requires element count multiple of %d, got %d", qt, blockElems, elements)
	}
	blocks := elements / blockElems
	return ggufCheckedMul(blocks, blockBytes)
}

func ggufU64ToInt(v uint64, what string) (int, error) {
	if v > uint64(ggufIntLimit()) {
		return 0, fmt.Errorf("gguf: %s %d exceeds int", what, v)
	}
	return int(v), nil
}

func ggufU64ToInt64(v uint64, what string) (int64, error) {
	if v > uint64(math.MaxInt64) {
		return 0, fmt.Errorf("gguf: %s %d exceeds int64", what, v)
	}
	return int64(v), nil
}

func ggufIntLimit() int { return int(^uint(0) >> 1) }
