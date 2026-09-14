// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT
// Package numpy reads a bounded numeric-only subset of NPZ/NPY, without pickle.
package numpy

import (
	"archive/zip"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// Array owns float64 values widened from plain endian-marked f4/f8 C-order NPY.
// No source buffers/files are retained. Shape has one or two positive axes.
type Array struct {
	Shape  []int
	Values []float64
}

// ReadNPZ reads an exact caller-supplied name/shape schema. Names omit .npy and
// contain only ASCII letters/digits/underscore. Limits: 1..16 members, each
// dimension1..512, rank1..2, archive<=16MiB, expanded total<=8MiB, header<=4096.
// ZIP store/deflate, NPY1/2/3, endian < or > f4/f8 only. Object/structured dtype,
// Fortran order, paths, encryption, extras/duplicates and unknown keys fail.
// Metadata/shape checks precede float-array allocation. Headers are parsed as
// a small literal grammar, NEVER eval/pickle. CRC and exact payload lengths are
// verified. Inputs must remain immutable during both metadata/payload passes;
// caller retains reader ownership. Cancellation returns no partial map.
func ReadNPZ(ctx context.Context, reader io.ReaderAt, size int64, schema map[string][]int) (map[string]Array, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if reader == nil || size < 22 || size > 16<<20 || len(schema) < 1 || len(schema) > 16 {
		return nil, fmt.Errorf("invalid NPZ bounds/schema")
	}
	for name, shape := range schema {
		if !validName(name) || len(shape) < 1 || len(shape) > 2 {
			return nil, fmt.Errorf("invalid NPZ schema")
		}
		for _, dim := range shape {
			if dim < 1 || dim > 512 {
				return nil, fmt.Errorf("invalid NPZ schema dimension")
			}
		}
	}
	if err := boundedDirectory(reader, size); err != nil {
		return nil, err
	}
	archive, err := zip.NewReader(reader, size)
	if err != nil {
		return nil, fmt.Errorf("NPZ ZIP: %w", err)
	}
	if len(archive.File) != len(schema) {
		return nil, fmt.Errorf("unexpected NPZ inventory")
	}
	type member struct {
		file        *zip.File
		shape       []int
		dtype       string
		headerBytes int
	}
	members := make([]member, 0, len(schema))
	seen := make(map[string]bool)
	var total uint64
	for _, file := range archive.File {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		name := strings.TrimSuffix(file.Name, ".npy")
		shape, ok := schema[name]
		if !ok || file.Name != name+".npy" || seen[name] || file.Flags&1 != 0 || (file.Method != zip.Store && file.Method != zip.Deflate) {
			return nil, fmt.Errorf("invalid NPZ member %q", file.Name)
		}
		seen[name] = true
		if file.UncompressedSize64 > 8<<20 || total+file.UncompressedSize64 > 8<<20 {
			return nil, fmt.Errorf("NPZ expanded limit")
		}
		total += file.UncompressedSize64
		stream, err := file.Open()
		if err != nil {
			return nil, err
		}
		dtype, actual, headerBytes, err := readHeader(stream)
		closeErr := stream.Close()
		if err != nil {
			return nil, err
		}
		if closeErr != nil {
			return nil, closeErr
		}
		if len(actual) != len(shape) {
			return nil, fmt.Errorf("NPZ rank mismatch %s", name)
		}
		count := 1
		for i, dim := range shape {
			if actual[i] != dim {
				return nil, fmt.Errorf("NPZ shape mismatch %s", name)
			}
			count *= dim
		}
		width := 4
		if dtype[1:] == "f8" {
			width = 8
		}
		if uint64(headerBytes+count*width) != file.UncompressedSize64 {
			return nil, fmt.Errorf("NPZ extent mismatch %s", name)
		}
		members = append(members, member{file, append([]int(nil), shape...), dtype, headerBytes})
	}
	output := make(map[string]Array, len(members))
	for _, m := range members {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		stream, err := m.file.Open()
		if err != nil {
			return nil, err
		}
		values, err := readValues(ctx, stream, m.headerBytes, m.dtype, m.shape)
		closeErr := stream.Close()
		if err != nil {
			return nil, err
		}
		if closeErr != nil {
			return nil, closeErr
		}
		output[strings.TrimSuffix(m.file.Name, ".npy")] = Array{m.shape, values}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// Check the classic EOCD before archive/zip can allocate its file inventory.
// ZIP64 archives are unnecessary at these bounds and are rejected. NumPy's
// force_zip64 local headers are accepted when the directory itself is classic.
func boundedDirectory(reader io.ReaderAt, size int64) error {
	tail := make([]byte, min(size, 65557))
	if _, err := reader.ReadAt(tail, size-int64(len(tail))); err != nil {
		return err
	}
	for i := len(tail) - 22; i >= 0; i-- {
		if string(tail[i:i+4]) != "PK\x05\x06" || i+22+int(binary.LittleEndian.Uint16(tail[i+20:])) != len(tail) {
			continue
		}
		disk, cdDisk := binary.LittleEndian.Uint16(tail[i+4:]), binary.LittleEndian.Uint16(tail[i+6:])
		local, count := binary.LittleEndian.Uint16(tail[i+8:]), binary.LittleEndian.Uint16(tail[i+10:])
		bytes, offset := uint64(binary.LittleEndian.Uint32(tail[i+12:])), uint64(binary.LittleEndian.Uint32(tail[i+16:]))
		if disk != 0 || cdDisk != 0 || local != count || count < 1 || count > 16 || bytes > 16384 || offset+bytes != uint64(size-int64(len(tail))+int64(i)) {
			return fmt.Errorf("NPZ central directory bound")
		}
		// archive/zip scans directory records rather than trusting the count.
		// Verify the bounded directory itself before handing it to that parser.
		directory := make([]byte, int(bytes))
		if _, err := reader.ReadAt(directory, int64(offset)); err != nil {
			return err
		}
		position := 0
		for record := 0; record < int(count); record++ {
			if position+46 > len(directory) || string(directory[position:position+4]) != "PK\x01\x02" {
				return fmt.Errorf("invalid NPZ directory record")
			}
			h := directory[position:]
			length := 46 + int(binary.LittleEndian.Uint16(h[28:])) + int(binary.LittleEndian.Uint16(h[30:])) + int(binary.LittleEndian.Uint16(h[32:]))
			if length > len(directory)-position {
				return fmt.Errorf("truncated NPZ directory record")
			}
			position += length
		}
		if position != len(directory) {
			return fmt.Errorf("NPZ directory count mismatch")
		}
		return nil
	}
	return fmt.Errorf("missing classic NPZ end record")
}
func validName(name string) bool {
	if len(name) < 1 || len(name) > 64 {
		return false
	}
	for _, c := range name {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_') {
			return false
		}
	}
	return true
}
func readValues(ctx context.Context, r io.Reader, header int, dtype string, shape []int) ([]float64, error) {
	if _, err := io.CopyN(io.Discard, r, int64(header)); err != nil {
		return nil, err
	}
	count := 1
	for _, dim := range shape {
		count *= dim
	}
	width := 4
	if dtype[1:] == "f8" {
		width = 8
	}
	var order binary.ByteOrder = binary.LittleEndian
	if dtype[0] == '>' {
		order = binary.BigEndian
	}
	out := make([]float64, count)
	var buffer [8192]byte
	for start := 0; start < count; {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n := min(count-start, len(buffer)/width)
		if _, err := io.ReadFull(r, buffer[:n*width]); err != nil {
			return nil, err
		}
		for i := 0; i < n; i++ {
			var v float64
			if width == 4 {
				v = float64(math.Float32frombits(order.Uint32(buffer[i*width:])))
			} else {
				v = math.Float64frombits(order.Uint64(buffer[i*width:]))
			}
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("nonfinite NPY value")
			}
			out[start+i] = v
		}
		start += n
	}
	var extra [1]byte
	n, err := r.Read(extra[:])
	if n != 0 || err != io.EOF {
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("NPY trailing payload")
	}
	return out, ctx.Err()
}
func readHeader(r io.Reader) (string, []int, int, error) {
	var fixed [12]byte
	if _, err := io.ReadFull(r, fixed[:8]); err != nil {
		return "", nil, 0, err
	}
	if string(fixed[:6]) != "\x93NUMPY" || fixed[7] != 0 || fixed[6] < 1 || fixed[6] > 3 {
		return "", nil, 0, fmt.Errorf("unsupported NPY magic/version")
	}
	prefix := 10
	length := uint32(0)
	if fixed[6] == 1 {
		if _, err := io.ReadFull(r, fixed[8:10]); err != nil {
			return "", nil, 0, err
		}
		length = uint32(binary.LittleEndian.Uint16(fixed[8:10]))
	} else {
		prefix = 12
		if _, err := io.ReadFull(r, fixed[8:12]); err != nil {
			return "", nil, 0, err
		}
		length = binary.LittleEndian.Uint32(fixed[8:12])
	}
	if length < 1 || length > 4096 {
		return "", nil, 0, fmt.Errorf("NPY header bound")
	}
	header := make([]byte, int(length))
	if _, err := io.ReadFull(r, header); err != nil {
		return "", nil, 0, err
	}
	if header[len(header)-1] != '\n' {
		return "", nil, 0, fmt.Errorf("NPY header missing newline")
	}
	p := headerParser{text: string(header)}
	dtype, shape, err := p.parse()
	return dtype, shape, prefix + int(length), err
}

// Literal grammar accepts key reordering and single/double quotes, but never
// escapes, expressions, duplicate keys, nested descriptors or unknown fields.
type headerParser struct {
	text string
	pos  int
}

func (p *headerParser) space() {
	for p.pos < len(p.text) && (p.text[p.pos] == ' ' || p.text[p.pos] == '\n' || p.text[p.pos] == '\t' || p.text[p.pos] == '\r') {
		p.pos++
	}
}
func (p *headerParser) take(c byte) bool {
	p.space()
	if p.pos < len(p.text) && p.text[p.pos] == c {
		p.pos++
		return true
	}
	return false
}
func (p *headerParser) quoted() (string, bool) {
	p.space()
	if p.pos >= len(p.text) || (p.text[p.pos] != '\'' && p.text[p.pos] != '"') {
		return "", false
	}
	q := p.text[p.pos]
	p.pos++
	start := p.pos
	for p.pos < len(p.text) && p.text[p.pos] != q {
		c := p.text[p.pos]
		if c < 32 || c > 126 || c == '\\' {
			return "", false
		}
		p.pos++
	}
	if p.pos == len(p.text) {
		return "", false
	}
	value := p.text[start:p.pos]
	p.pos++
	return value, true
}
func (p *headerParser) parse() (string, []int, error) {
	fail := fmt.Errorf("unsupported NPY literal header")
	if !p.take('{') {
		return "", nil, fail
	}
	seen := make(map[string]bool)
	dtype := ""
	var shape []int
	for {
		key, ok := p.quoted()
		if !ok || seen[key] || !p.take(':') {
			return "", nil, fail
		}
		seen[key] = true
		switch key {
		case "descr":
			dtype, ok = p.quoted()
			if !ok || (dtype != "<f4" && dtype != "<f8" && dtype != ">f4" && dtype != ">f8") {
				return "", nil, fail
			}
		case "fortran_order":
			p.space()
			if !strings.HasPrefix(p.text[p.pos:], "False") {
				return "", nil, fail
			}
			p.pos += 5
		case "shape":
			if !p.take('(') {
				return "", nil, fail
			}
			for {
				p.space()
				start := p.pos
				for p.pos < len(p.text) && p.text[p.pos] >= '0' && p.text[p.pos] <= '9' {
					p.pos++
				}
				if start == p.pos {
					return "", nil, fail
				}
				dim, err := strconv.Atoi(p.text[start:p.pos])
				if err != nil || dim < 1 || dim > 512 || len(shape) >= 2 {
					return "", nil, fail
				}
				shape = append(shape, dim)
				if p.take(')') {
					if len(shape) == 1 {
						return "", nil, fail
					}
					break
				}
				if !p.take(',') {
					return "", nil, fail
				}
				if p.take(')') {
					break
				}
			}
		default:
			return "", nil, fail
		}
		if p.take('}') {
			break
		}
		if !p.take(',') {
			return "", nil, fail
		}
		if p.take('}') {
			break
		}
	}
	p.space()
	if p.pos != len(p.text) || len(seen) != 3 || len(shape) < 1 || dtype == "" {
		return "", nil, fail
	}
	return dtype, shape, nil
}
