package gguf

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	ggufWriterVersion = 3
	ggufAlignment     = 32
)

// MetadataEntry describes one GGUF metadata key/value pair.
//
// Supported value types are:
//   - string
//   - bool
//   - int8, int16, int32, int64, int
//   - uint8, uint16, uint32, uint64, uint
//   - float32, float64
//
// Arrays are intentionally unsupported to keep the writer bounded and simple.
type MetadataEntry struct {
	Key   string
	Value any
}

// TensorSpec describes one tensor to be written in GGUF dimension order
// (innermost dimension first), matching TensorInfo.Shape.
type TensorSpec struct {
	Name  string
	Shape []uint64
	QType QuantType
}

// TensorReaderFunc opens a streaming reader for one tensor payload. The reader
// must yield exactly the encoded raw byte count implied by the tensor shape and
// quantization type.
type TensorReaderFunc func(ctx context.Context, index int, tensor TensorSpec) (io.Reader, error)

// ExportTensor is a convenience adapter for callers that already have one raw
// reader per tensor.
type ExportTensor struct {
	Name  string
	Shape []uint64
	QType QuantType
	Raw   io.Reader
}

// WriteV3 writes a GGUF v3 file to path. The destination is created
// exclusively; if writing fails or ctx is canceled, any partial file is removed.
//
// Tensor payloads are streamed one at a time through open, so callers do not
// need to materialize an entire model in memory.
func WriteV3(ctx context.Context, path string, meta []MetadataEntry, tensors []TensorSpec, open TensorReaderFunc) (err error) {
	if ctx == nil {
		return fmt.Errorf("gguf writer: nil context")
	}
	if open == nil {
		return fmt.Errorf("gguf writer: nil tensor reader callback")
	}
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("gguf writer: %w", err)
	}

	plannedMeta, err := planMetadata(meta)
	if err != nil {
		return err
	}
	plannedTensors, headerSize, dataOffset, err := planTensors(tensors, plannedMeta)
	if err != nil {
		return err
	}
	if dataOffset > math.MaxInt64 {
		return fmt.Errorf("gguf writer: data offset %d exceeds int64", dataOffset)
	}

	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := f.Close(); err == nil && cerr != nil {
			err = cerr
		}
		if err != nil {
			_ = os.Remove(path)
		}
	}()

	bw := bufio.NewWriterSize(f, 1<<20)
	if err = writeHeader(bw, plannedMeta, plannedTensors); err != nil {
		return err
	}
	if err = writePadding(bw, headerSize, dataOffset); err != nil {
		return err
	}

	current := dataOffset
	for i, t := range plannedTensors {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("gguf writer: %w", err)
		}
		expected := dataOffset + t.Offset
		if current != expected {
			return fmt.Errorf("gguf writer: internal offset mismatch for %q: have %d want %d", t.Name, current, expected)
		}

		r, err := open(ctx, i, t.TensorSpec)
		if err != nil {
			return fmt.Errorf("gguf writer: tensor %q reader: %w", t.Name, err)
		}
		if r == nil {
			return fmt.Errorf("gguf writer: tensor %q reader is nil", t.Name)
		}

		copyErr := copyExact(ctx, bw, r, t.RawBytes)
		closeErr := closeIfPossible(r)
		if copyErr != nil {
			return fmt.Errorf("gguf writer: tensor %q payload: %w", t.Name, copyErr)
		}
		if closeErr != nil {
			return fmt.Errorf("gguf writer: tensor %q close: %w", t.Name, closeErr)
		}

		current += t.RawBytes
		if i+1 < len(plannedTensors) {
			next, err := alignUp(current, ggufAlignment)
			if err != nil {
				return err
			}
			if err := writePadding(bw, current, next); err != nil {
				return err
			}
			current = next
		}
	}

	if err = bw.Flush(); err != nil {
		return err
	}
	if err = f.Sync(); err != nil {
		return err
	}
	return nil
}

// WriteTensors is a convenience helper for exporting GGUF files from
// scalar metadata entries and per-tensor raw readers.
func WriteTensors(ctx context.Context, path string, meta []MetadataEntry, tensors []ExportTensor) error {
	specs := make([]TensorSpec, len(tensors))
	readers := make([]io.Reader, len(tensors))
	used := make([]bool, len(tensors))
	for i, t := range tensors {
		specs[i] = TensorSpec{Name: t.Name, Shape: append([]uint64(nil), t.Shape...), QType: t.QType}
		readers[i] = t.Raw
	}
	return WriteV3(ctx, path, meta, specs, func(_ context.Context, index int, tensor TensorSpec) (io.Reader, error) {
		if index < 0 || index >= len(readers) {
			return nil, fmt.Errorf("unexpected tensor index %d", index)
		}
		if used[index] {
			return nil, fmt.Errorf("tensor %q reader already consumed", tensor.Name)
		}
		if readers[index] == nil {
			return nil, fmt.Errorf("tensor %q reader is nil", tensor.Name)
		}
		used[index] = true
		return readers[index], nil
	})
}

type plannedMetadata struct {
	MetadataEntry
	Type GGUFType
	Size uint64
}

type plannedTensor struct {
	TensorSpec
	Elements uint64
	RawBytes uint64
	Offset   uint64
}

func planMetadata(meta []MetadataEntry) ([]plannedMetadata, error) {
	seen := make(map[string]struct{}, len(meta))
	planned := make([]plannedMetadata, len(meta))
	for i, kv := range meta {
		if kv.Key == "" {
			return nil, fmt.Errorf("gguf writer: metadata[%d] has empty key", i)
		}
		if _, dup := seen[kv.Key]; dup {
			return nil, fmt.Errorf("gguf writer: duplicate metadata key %q", kv.Key)
		}
		seen[kv.Key] = struct{}{}
		if kv.Key == "general.alignment" && kv.Value != uint32(32) {
			return nil, fmt.Errorf("gguf writer: general.alignment must be uint32(32)")
		}
		typ, size, err := metadataValuePlan(kv.Value)
		if err != nil {
			return nil, fmt.Errorf("gguf writer: metadata[%d] %q: %w", i, kv.Key, err)
		}
		planned[i] = plannedMetadata{MetadataEntry: kv, Type: typ, Size: size}
	}
	return planned, nil
}

func planTensors(tensors []TensorSpec, meta []plannedMetadata) ([]plannedTensor, uint64, uint64, error) {
	headerSize := uint64(4 + 4 + 8 + 8)
	for _, kv := range meta {
		var err error
		headerSize, err = checkedAdd(headerSize, stringFieldSize(kv.Key))
		if err != nil {
			return nil, 0, 0, err
		}
		headerSize, err = checkedAdd(headerSize, 4)
		if err != nil {
			return nil, 0, 0, err
		}
		headerSize, err = checkedAdd(headerSize, kv.Size)
		if err != nil {
			return nil, 0, 0, err
		}
	}

	seen := make(map[string]struct{}, len(tensors))
	planned := make([]plannedTensor, len(tensors))
	currentOffset := uint64(0)
	for i, t := range tensors {
		if t.Name == "" {
			return nil, 0, 0, fmt.Errorf("gguf writer: tensor[%d] has empty name", i)
		}
		if _, dup := seen[t.Name]; dup {
			return nil, 0, 0, fmt.Errorf("gguf writer: duplicate tensor name %q", t.Name)
		}
		seen[t.Name] = struct{}{}
		if len(t.Shape) == 0 {
			return nil, 0, 0, fmt.Errorf("gguf writer: tensor[%d] %q has empty shape", i, t.Name)
		}
		if len(t.Shape) > math.MaxUint32 {
			return nil, 0, 0, fmt.Errorf("gguf writer: tensor[%d] %q rank %d exceeds uint32", i, t.Name, len(t.Shape))
		}

		elements, rawBytes, err := tensorEncodingSize(t.QType, t.Shape)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("gguf writer: tensor[%d] %q: %w", i, t.Name, err)
		}

		planned[i] = plannedTensor{
			TensorSpec: TensorSpec{Name: t.Name, Shape: append([]uint64(nil), t.Shape...), QType: t.QType},
			Elements:   elements,
			RawBytes:   rawBytes,
			Offset:     currentOffset,
		}

		var hdrErr error
		headerSize, hdrErr = checkedAdd(headerSize, stringFieldSize(t.Name))
		if hdrErr != nil {
			return nil, 0, 0, hdrErr
		}
		headerSize, hdrErr = checkedAdd(headerSize, 4)
		if hdrErr != nil {
			return nil, 0, 0, hdrErr
		}
		headerSize, hdrErr = checkedAdd(headerSize, uint64(len(t.Shape))*8)
		if hdrErr != nil {
			return nil, 0, 0, hdrErr
		}
		headerSize, hdrErr = checkedAdd(headerSize, 4+8)
		if hdrErr != nil {
			return nil, 0, 0, hdrErr
		}

		currentOffset, err = checkedAdd(currentOffset, rawBytes)
		if err != nil {
			return nil, 0, 0, err
		}
		if i+1 < len(tensors) {
			currentOffset, err = alignUp(currentOffset, ggufAlignment)
			if err != nil {
				return nil, 0, 0, err
			}
		}
	}

	dataOffset, err := alignUp(headerSize, ggufAlignment)
	if err != nil {
		return nil, 0, 0, err
	}
	if dataOffset > math.MaxInt64 {
		return nil, 0, 0, fmt.Errorf("gguf writer: data offset %d exceeds int64", dataOffset)
	}
	for _, t := range planned {
		if t.Offset > math.MaxInt64 {
			return nil, 0, 0, fmt.Errorf("gguf writer: tensor %q offset %d exceeds int64", t.Name, t.Offset)
		}
		if _, err := checkedAdd(dataOffset, t.Offset); err != nil {
			return nil, 0, 0, err
		}
		if dataOffset+t.Offset > math.MaxInt64 {
			return nil, 0, 0, fmt.Errorf("gguf writer: tensor %q absolute offset exceeds int64", t.Name)
		}
	}
	return planned, headerSize, dataOffset, nil
}

func metadataValuePlan(v any) (GGUFType, uint64, error) {
	switch vv := v.(type) {
	case string:
		return GGUFTypeString, stringFieldSize(vv), nil
	case bool:
		return GGUFTypeBool, 1, nil
	case int8:
		return GGUFTypeI8, 1, nil
	case uint8:
		return GGUFTypeU8, 1, nil
	case int16:
		return GGUFTypeI16, 2, nil
	case uint16:
		return GGUFTypeU16, 2, nil
	case int32:
		return GGUFTypeI32, 4, nil
	case uint32:
		return GGUFTypeU32, 4, nil
	case float32:
		return GGUFTypeF32, 4, nil
	case int:
		if int64(vv) < math.MinInt64 || int64(vv) > math.MaxInt64 {
			return 0, 0, fmt.Errorf("int value out of range")
		}
		return GGUFTypeI64, 8, nil
	case uint:
		return GGUFTypeU64, 8, nil
	case int64:
		return GGUFTypeI64, 8, nil
	case uint64:
		return GGUFTypeU64, 8, nil
	case float64:
		return GGUFTypeF64, 8, nil
	default:
		return 0, 0, fmt.Errorf("unsupported metadata value type %T", v)
	}
}

func tensorEncodingSize(qt QuantType, shape []uint64) (elements uint64, rawBytes uint64, err error) {
	elements = 1
	for _, dim := range shape {
		if dim == 0 {
			return 0, 0, fmt.Errorf("zero dimension in shape %v", shape)
		}
		elements, err = checkedMul(elements, dim)
		if err != nil {
			return 0, 0, fmt.Errorf("shape %v overflows element count", shape)
		}
	}
	if elements == 0 {
		return 0, 0, fmt.Errorf("shape %v has zero elements", shape)
	}

	switch qt {
	case QuantF32:
		rawBytes, err = checkedMul(elements, 4)
	case QuantF16, QuantBF16:
		rawBytes, err = checkedMul(elements, 2)
	case QuantQ4_0:
		rawBytes, err = blockEncodedSize(elements, 32, 18, qt)
	case QuantQ4_1:
		rawBytes, err = blockEncodedSize(elements, 32, 20, qt)
	case QuantQ5_0:
		rawBytes, err = blockEncodedSize(elements, 32, 22, qt)
	case QuantQ5_1:
		rawBytes, err = blockEncodedSize(elements, 32, 24, qt)
	case QuantQ8_0:
		rawBytes, err = blockEncodedSize(elements, 32, 34, qt)
	case QuantQ2_K:
		rawBytes, err = blockEncodedSize(elements, 256, 84, qt)
	case QuantQ3_K:
		rawBytes, err = blockEncodedSize(elements, 256, 110, qt)
	case QuantQ4_K:
		rawBytes, err = blockEncodedSize(elements, 256, 144, qt)
	case QuantQ5_K:
		rawBytes, err = blockEncodedSize(elements, 256, 176, qt)
	case QuantQ6_K:
		rawBytes, err = blockEncodedSize(elements, 256, 210, qt)
	case QuantQ8_K:
		rawBytes, err = blockEncodedSize(elements, 256, 292, qt)
	default:
		return 0, 0, fmt.Errorf("unsupported quant type %s", qt)
	}
	if err != nil {
		return 0, 0, err
	}
	return elements, rawBytes, nil
}

func blockEncodedSize(elements, blockElems, blockBytes uint64, qt QuantType) (uint64, error) {
	if elements%blockElems != 0 {
		return 0, fmt.Errorf("quant type %s requires element count multiple of %d, got %d", qt, blockElems, elements)
	}
	blocks := elements / blockElems
	return checkedMul(blocks, blockBytes)
}

func writeHeader(w io.Writer, meta []plannedMetadata, tensors []plannedTensor) error {
	if _, err := io.WriteString(w, "GGUF"); err != nil {
		return err
	}
	if err := writeU32(w, ggufWriterVersion); err != nil {
		return err
	}
	if err := writeU64(w, uint64(len(tensors))); err != nil {
		return err
	}
	if err := writeU64(w, uint64(len(meta))); err != nil {
		return err
	}
	for _, kv := range meta {
		if err := writeString(w, kv.Key); err != nil {
			return err
		}
		if err := writeU32(w, uint32(kv.Type)); err != nil {
			return err
		}
		if err := writeMetadataValue(w, kv.Type, kv.Value); err != nil {
			return err
		}
	}
	for _, t := range tensors {
		if err := writeString(w, t.Name); err != nil {
			return err
		}
		if err := writeU32(w, uint32(len(t.Shape))); err != nil {
			return err
		}
		for _, dim := range t.Shape {
			if err := writeU64(w, dim); err != nil {
				return err
			}
		}
		if err := writeU32(w, uint32(t.QType)); err != nil {
			return err
		}
		if err := writeU64(w, t.Offset); err != nil {
			return err
		}
	}
	return nil
}

func writeMetadataValue(w io.Writer, typ GGUFType, value any) error {
	switch typ {
	case GGUFTypeString:
		return writeString(w, value.(string))
	case GGUFTypeBool:
		b := byte(0)
		if value.(bool) {
			b = 1
		}
		return writeBytes(w, []byte{b})
	case GGUFTypeU8:
		return writeBytes(w, []byte{value.(uint8)})
	case GGUFTypeI8:
		return writeBytes(w, []byte{byte(value.(int8))})
	case GGUFTypeU16:
		return writeU16(w, value.(uint16))
	case GGUFTypeI16:
		return writeU16(w, uint16(value.(int16)))
	case GGUFTypeU32:
		return writeU32(w, value.(uint32))
	case GGUFTypeI32:
		return writeU32(w, uint32(value.(int32)))
	case GGUFTypeF32:
		return writeU32(w, math.Float32bits(value.(float32)))
	case GGUFTypeU64:
		switch v := value.(type) {
		case uint:
			return writeU64(w, uint64(v))
		case uint64:
			return writeU64(w, v)
		default:
			panic(fmt.Sprintf("unsupported U64 value %T", value))
		}
	case GGUFTypeI64:
		switch v := value.(type) {
		case int:
			return writeU64(w, uint64(int64(v)))
		case int64:
			return writeU64(w, uint64(v))
		default:
			panic(fmt.Sprintf("unsupported I64 value %T", value))
		}
	case GGUFTypeF64:
		return writeU64(w, math.Float64bits(value.(float64)))
	default:
		return fmt.Errorf("unsupported metadata type %d", typ)
	}
}

func copyExact(ctx context.Context, dst io.Writer, src io.Reader, want uint64) error {
	buf := make([]byte, 32<<10)
	remaining := want
	for remaining > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		next := len(buf)
		if uint64(next) > remaining {
			next = int(remaining)
		}
		nr, er := src.Read(buf[:next])
		if nr > 0 {
			nw, ew := dst.Write(buf[:nr])
			if ew != nil {
				return ew
			}
			if nw != nr {
				return io.ErrShortWrite
			}
			remaining -= uint64(nw)
		}
		if er != nil {
			if er == io.EOF && remaining == 0 {
				break
			}
			if er == io.EOF {
				return io.ErrUnexpectedEOF
			}
			return er
		}
		if nr == 0 {
			return io.ErrNoProgress
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	var extra [1]byte
	n, er := src.Read(extra[:])
	if n > 0 {
		return fmt.Errorf("payload longer than expected %d bytes", want)
	}
	if er != nil && er != io.EOF {
		return er
	}
	return nil
}

func closeIfPossible(r io.Reader) error {
	closer, ok := r.(io.Closer)
	if !ok {
		return nil
	}
	return closer.Close()
}

func stringFieldSize(s string) uint64 { return 8 + uint64(len(s)) }

func writePadding(w io.Writer, from, to uint64) error {
	if to < from {
		return fmt.Errorf("gguf writer: invalid padding range %d..%d", from, to)
	}
	remaining := to - from
	if remaining == 0 {
		return nil
	}
	var zeros [ggufAlignment]byte
	for remaining > 0 {
		chunk := uint64(len(zeros))
		if chunk > remaining {
			chunk = remaining
		}
		if err := writeBytes(w, zeros[:chunk]); err != nil {
			return err
		}
		remaining -= chunk
	}
	return nil
}

func checkedAdd(a, b uint64) (uint64, error) {
	if a > math.MaxUint64-b {
		return 0, fmt.Errorf("gguf writer: uint64 overflow adding %d and %d", a, b)
	}
	return a + b, nil
}

func checkedMul(a, b uint64) (uint64, error) {
	if a == 0 || b == 0 {
		return 0, nil
	}
	if a > math.MaxUint64/b {
		return 0, fmt.Errorf("gguf writer: uint64 overflow multiplying %d and %d", a, b)
	}
	return a * b, nil
}

func alignUp(v, align uint64) (uint64, error) {
	if align == 0 {
		return 0, fmt.Errorf("gguf writer: zero alignment")
	}
	rem := v % align
	if rem == 0 {
		return v, nil
	}
	return checkedAdd(v, align-rem)
}

func writeBytes(w io.Writer, b []byte) error {
	n, err := w.Write(b)
	if err != nil {
		return err
	}
	if n != len(b) {
		return io.ErrShortWrite
	}
	return nil
}

func writeString(w io.Writer, s string) error {
	if err := writeU64(w, uint64(len(s))); err != nil {
		return err
	}
	return writeBytes(w, []byte(s))
}

func writeU16(w io.Writer, v uint16) error {
	var buf [2]byte
	binary.LittleEndian.PutUint16(buf[:], v)
	return writeBytes(w, buf[:])
}

func writeU32(w io.Writer, v uint32) error {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], v)
	return writeBytes(w, buf[:])
}

func writeU64(w io.Writer, v uint64) error {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], v)
	return writeBytes(w, buf[:])
}
