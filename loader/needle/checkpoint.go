package needle

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/rcarmo/go-pherence/half"
)

const (
	maxHeaderBytes      int64 = 16 << 20
	maxFileBytes        int64 = 2 << 30
	maxTotalTensorBytes int64 = 2 << 30
	maxRank                   = 8
)

// Tensor stores a decoded checkpoint tensor as float32 values.
type Tensor struct {
	Shape []int
	Data  []float32
}

// Checkpoint is the non-executable Needle safetensors checkpoint payload.
type Checkpoint struct {
	Config        json.RawMessage
	FormatVersion int
	Tensors       map[string]Tensor
}

type rawTensorInfo struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

type tensorSpec struct {
	Name     string
	DType    string
	Shape    []int
	Elements int
	Start    int64
	End      int64
	ByteLen  int64
}

// Load reads a Needle checkpoint from a safetensors file.
func Load(path string) (*Checkpoint, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("needle: open %s: %w", path, err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("needle: stat %s: %w", path, err)
	}
	size := fi.Size()
	if size < 8 {
		return nil, fmt.Errorf("needle: file too short")
	}
	if size > maxFileBytes {
		return nil, fmt.Errorf("needle: file size %d exceeds %d", size, maxFileBytes)
	}

	var lenBuf [8]byte
	if _, err := io.ReadFull(f, lenBuf[:]); err != nil {
		return nil, fmt.Errorf("needle: read header length: %w", err)
	}
	headerSize := binary.LittleEndian.Uint64(lenBuf[:])
	if headerSize > uint64(maxHeaderBytes) {
		return nil, fmt.Errorf("needle: header size %d exceeds %d", headerSize, maxHeaderBytes)
	}
	headerLen := int64(headerSize)
	if headerLen > maxHeaderBytes {
		return nil, fmt.Errorf("needle: header size %d exceeds %d", headerLen, maxHeaderBytes)
	}
	if headerLen > size-8 {
		return nil, fmt.Errorf("needle: header size %d exceeds file size", headerLen)
	}
	if headerLen > int64(maxInt()) {
		return nil, fmt.Errorf("needle: header size %d exceeds platform limit", headerLen)
	}

	header := make([]byte, int(headerLen))
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, fmt.Errorf("needle: read header: %w", err)
	}

	config, formatVersion, specs, err := parseHeader(header, size-8-headerLen)
	if err != nil {
		return nil, err
	}

	checkpoint := &Checkpoint{
		Config:        config,
		FormatVersion: formatVersion,
		Tensors:       make(map[string]Tensor, len(specs)),
	}
	dataBase := 8 + headerLen
	for _, spec := range specs {
		if spec.ByteLen > int64(maxInt()) {
			return nil, fmt.Errorf("needle: tensor %q byte size %d exceeds platform limit", spec.Name, spec.ByteLen)
		}
		raw := make([]byte, int(spec.ByteLen))
		if _, err := f.ReadAt(raw, dataBase+spec.Start); err != nil {
			return nil, fmt.Errorf("needle: read tensor %q: %w", spec.Name, err)
		}
		data, err := decodeFloat32(spec.Name, spec.DType, raw)
		if err != nil {
			return nil, err
		}
		if err := validateFinite(spec.Name, data); err != nil {
			return nil, err
		}
		shape := append([]int{}, spec.Shape...)
		checkpoint.Tensors[spec.Name] = Tensor{Shape: shape, Data: data}
	}
	return checkpoint, nil
}

// Save writes a Needle checkpoint as a safetensors file.
func Save(path string, c *Checkpoint) error {
	if c == nil {
		return fmt.Errorf("needle: nil checkpoint")
	}
	names, header, totalTensorBytes, err := buildSaveHeader(c)
	if err != nil {
		return err
	}

	headerLen := int64(len(header))
	fileSize, ok := checkedAddInt64(8, headerLen)
	if !ok {
		return fmt.Errorf("needle: file size overflow")
	}
	fileSize, ok = checkedAddInt64(fileSize, totalTensorBytes)
	if !ok {
		return fmt.Errorf("needle: file size overflow")
	}
	if fileSize > maxFileBytes {
		return fmt.Errorf("needle: file size %d exceeds %d", fileSize, maxFileBytes)
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("needle: mkdir %s: %w", dir, err)
	}
	f, err := os.CreateTemp(dir, ".needle-*.tmp")
	if err != nil {
		return fmt.Errorf("needle: create temp file: %w", err)
	}
	tmp := f.Name()
	defer os.Remove(tmp)

	bw := bufio.NewWriterSize(f, 1<<20)
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(header)))
	if _, err := bw.Write(lenBuf[:]); err != nil {
		f.Close()
		return fmt.Errorf("needle: write header length: %w", err)
	}
	if _, err := bw.Write(header); err != nil {
		f.Close()
		return fmt.Errorf("needle: write header: %w", err)
	}
	chunk := make([]byte, 4096*4)
	for _, name := range names {
		tensor := c.Tensors[name]
		for off := 0; off < len(tensor.Data); {
			n := len(tensor.Data) - off
			if n > 4096 {
				n = 4096
			}
			for i := 0; i < n; i++ {
				binary.LittleEndian.PutUint32(chunk[i*4:], math.Float32bits(tensor.Data[off+i]))
			}
			if _, err := bw.Write(chunk[:n*4]); err != nil {
				f.Close()
				return fmt.Errorf("needle: write tensor %q: %w", name, err)
			}
			off += n
		}
	}
	if err := bw.Flush(); err != nil {
		f.Close()
		return fmt.Errorf("needle: flush %s: %w", tmp, err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("needle: sync %s: %w", tmp, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("needle: close %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("needle: rename %s to %s: %w", tmp, path, err)
	}
	return nil
}

func buildSaveHeader(c *Checkpoint) ([]string, []byte, int64, error) {
	if c.FormatVersion < 0 {
		return nil, nil, 0, fmt.Errorf("needle: negative format_version %d", c.FormatVersion)
	}
	config := append(json.RawMessage(nil), c.Config...)
	if len(config) == 0 {
		config = json.RawMessage(`{}`)
	}
	if !json.Valid(config) {
		return nil, nil, 0, fmt.Errorf("needle: invalid config JSON")
	}

	names := make([]string, 0, len(c.Tensors))
	for name := range c.Tensors {
		if name == "__metadata__" {
			return nil, nil, 0, fmt.Errorf("needle: reserved tensor name __metadata__")
		}
		names = append(names, name)
	}
	sort.Strings(names)

	type saveEntry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}

	entries := make(map[string]saveEntry, len(names))
	var offset int64
	for _, name := range names {
		if name == "" {
			return nil, nil, 0, fmt.Errorf("needle: empty tensor name")
		}
		tensor := c.Tensors[name]
		shape, elements, err := validateShapeForSave(name, tensor.Shape)
		if err != nil {
			return nil, nil, 0, err
		}
		if len(tensor.Data) != elements {
			return nil, nil, 0, fmt.Errorf("needle: tensor %q has %d values, want %d", name, len(tensor.Data), elements)
		}
		if err := validateFinite(name, tensor.Data); err != nil {
			return nil, nil, 0, err
		}
		byteLen, ok := checkedMulInt64(int64(elements), 4)
		if !ok {
			return nil, nil, 0, fmt.Errorf("needle: tensor %q byte size overflow", name)
		}
		if byteLen > maxTotalTensorBytes {
			return nil, nil, 0, fmt.Errorf("needle: tensor %q byte size %d exceeds %d", name, byteLen, maxTotalTensorBytes)
		}
		end, ok := checkedAddInt64(offset, byteLen)
		if !ok {
			return nil, nil, 0, fmt.Errorf("needle: tensor %q offsets overflow", name)
		}
		if end > maxTotalTensorBytes {
			return nil, nil, 0, fmt.Errorf("needle: tensor bytes %d exceed %d", end, maxTotalTensorBytes)
		}
		if end > int64(maxInt()) {
			return nil, nil, 0, fmt.Errorf("needle: tensor %q offsets exceed platform limit", name)
		}
		entries[name] = saveEntry{
			DType:       "F32",
			Shape:       shape,
			DataOffsets: [2]int{int(offset), int(end)},
		}
		offset = end
	}

	metadata, err := json.Marshal(map[string]string{
		"config":         string(config),
		"format_version": strconv.Itoa(c.FormatVersion),
	})
	if err != nil {
		return nil, nil, 0, fmt.Errorf("needle: marshal metadata: %w", err)
	}

	var buf bytes.Buffer
	buf.WriteByte('{')
	buf.WriteString(`"__metadata__":`)
	buf.Write(metadata)
	for _, name := range names {
		key, err := json.Marshal(name)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("needle: marshal tensor name %q: %w", name, err)
		}
		value, err := json.Marshal(entries[name])
		if err != nil {
			return nil, nil, 0, fmt.Errorf("needle: marshal tensor %q: %w", name, err)
		}
		buf.WriteByte(',')
		buf.Write(key)
		buf.WriteByte(':')
		buf.Write(value)
	}
	buf.WriteByte('}')
	// Align tensor data for native safetensors readers.
	for buf.Len()%8 != 0 {
		buf.WriteByte(' ')
	}
	if int64(buf.Len()) > maxHeaderBytes {
		return nil, nil, 0, fmt.Errorf("needle: header size %d exceeds %d", buf.Len(), maxHeaderBytes)
	}
	return names, buf.Bytes(), offset, nil
}

func parseHeader(header []byte, dataLen int64) (json.RawMessage, int, []tensorSpec, error) {
	var raw map[string]json.RawMessage
	if err := decodeJSONStrict(header, &raw); err != nil {
		return nil, 0, nil, fmt.Errorf("needle: parse header: %w", err)
	}

	metadataRaw, ok := raw["__metadata__"]
	if !ok {
		return nil, 0, nil, fmt.Errorf("needle: missing __metadata__")
	}
	delete(raw, "__metadata__")

	var metadata map[string]string
	if err := decodeJSONStrict(metadataRaw, &metadata); err != nil {
		return nil, 0, nil, fmt.Errorf("needle: parse metadata: %w", err)
	}
	formatText, ok := metadata["format_version"]
	if !ok || formatText == "" {
		return nil, 0, nil, fmt.Errorf("needle: missing format_version metadata")
	}
	formatVersion, err := strconv.Atoi(formatText)
	if err != nil || formatVersion < 0 {
		return nil, 0, nil, fmt.Errorf("needle: invalid format_version %q", formatText)
	}
	configText, ok := metadata["config"]
	if !ok {
		return nil, 0, nil, fmt.Errorf("needle: missing config metadata")
	}
	if !json.Valid([]byte(configText)) {
		return nil, 0, nil, fmt.Errorf("needle: invalid config JSON")
	}
	config := append(json.RawMessage(nil), []byte(configText)...)

	specs := make([]tensorSpec, 0, len(raw))
	var totalBytes int64
	for name, rawInfo := range raw {
		if name == "" {
			return nil, 0, nil, fmt.Errorf("needle: empty tensor name")
		}
		info, err := parseRawTensorInfo(rawInfo)
		if err != nil {
			return nil, 0, nil, fmt.Errorf("needle: parse tensor %q: %w", name, err)
		}
		spec, err := validateTensorInfo(name, info, dataLen)
		if err != nil {
			return nil, 0, nil, err
		}
		totalBytes, err = addTensorBytes(totalBytes, spec.ByteLen)
		if err != nil {
			return nil, 0, nil, err
		}
		specs = append(specs, spec)
	}
	sort.Slice(specs, func(i, j int) bool {
		if specs[i].Start != specs[j].Start {
			return specs[i].Start < specs[j].Start
		}
		return specs[i].Name < specs[j].Name
	})
	for i := 1; i < len(specs); i++ {
		if specs[i].Start < specs[i-1].End {
			return nil, 0, nil, fmt.Errorf("needle: tensors %q and %q overlap", specs[i-1].Name, specs[i].Name)
		}
	}
	sort.Slice(specs, func(i, j int) bool { return specs[i].Name < specs[j].Name })
	return config, formatVersion, specs, nil
}

func addTensorBytes(total, n int64) (int64, error) {
	next, ok := checkedAddInt64(total, n)
	if !ok {
		return 0, fmt.Errorf("needle: total tensor bytes overflow")
	}
	if next > maxTotalTensorBytes {
		return 0, fmt.Errorf("needle: tensor bytes %d exceed %d", next, maxTotalTensorBytes)
	}
	return next, nil
}

func validateTensorInfo(name string, info rawTensorInfo, dataLen int64) (tensorSpec, error) {
	elemSize, ok := dtypeSize(info.DType)
	if !ok {
		return tensorSpec{}, fmt.Errorf("needle: tensor %q unsupported dtype %q", name, info.DType)
	}
	shape, elements, err := validateShape(name, info.Shape)
	if err != nil {
		return tensorSpec{}, err
	}
	start, end := info.DataOffsets[0], info.DataOffsets[1]
	if start < 0 || end < start || end > dataLen {
		return tensorSpec{}, fmt.Errorf("needle: tensor %q invalid data offsets [%d,%d] for data length %d", name, start, end, dataLen)
	}
	wantBytes, ok := checkedMulInt64(int64(elements), int64(elemSize))
	if !ok {
		return tensorSpec{}, fmt.Errorf("needle: tensor %q byte size overflow", name)
	}
	if wantBytes > maxTotalTensorBytes {
		return tensorSpec{}, fmt.Errorf("needle: tensor %q byte size %d exceeds %d", name, wantBytes, maxTotalTensorBytes)
	}
	if end-start != wantBytes {
		return tensorSpec{}, fmt.Errorf("needle: tensor %q shape %v dtype %s expects %d bytes, got %d", name, shape, info.DType, wantBytes, end-start)
	}
	return tensorSpec{
		Name:     name,
		DType:    info.DType,
		Shape:    shape,
		Elements: elements,
		Start:    start,
		End:      end,
		ByteLen:  wantBytes,
	}, nil
}

func validateShape(name string, raw []int64) ([]int, int, error) {
	if len(raw) > maxRank {
		return nil, 0, fmt.Errorf("needle: tensor %q rank %d exceeds %d", name, len(raw), maxRank)
	}
	if len(raw) == 0 {
		return []int{}, 1, nil
	}
	shape := make([]int, len(raw))
	elements := 1
	limit := maxInt()
	for i, dim64 := range raw {
		if dim64 <= 0 {
			return nil, 0, fmt.Errorf("needle: tensor %q has invalid dimension %d at axis %d", name, dim64, i)
		}
		if dim64 > int64(limit) {
			return nil, 0, fmt.Errorf("needle: tensor %q dimension %d exceeds platform limit", name, dim64)
		}
		dim := int(dim64)
		if elements > limit/dim {
			return nil, 0, fmt.Errorf("needle: tensor %q shape %v overflows element count", name, raw)
		}
		elements *= dim
		shape[i] = dim
	}
	return shape, elements, nil
}

func parseRawTensorInfo(raw []byte) (rawTensorInfo, error) {
	var fields map[string]json.RawMessage
	if err := decodeJSONStrict(raw, &fields); err != nil {
		return rawTensorInfo{}, err
	}
	for key := range fields {
		switch key {
		case "dtype", "shape", "data_offsets":
		default:
			return rawTensorInfo{}, fmt.Errorf("unknown field %q", key)
		}
	}
	for _, key := range []string{"dtype", "shape", "data_offsets"} {
		rawValue, ok := fields[key]
		if !ok {
			return rawTensorInfo{}, fmt.Errorf("missing %s", key)
		}
		if bytes.Equal(bytes.TrimSpace(rawValue), []byte("null")) {
			return rawTensorInfo{}, fmt.Errorf("null %s", key)
		}
	}
	var info rawTensorInfo
	if err := decodeJSONStrict(raw, &info); err != nil {
		return rawTensorInfo{}, err
	}
	return info, nil
}

func validateShapeForSave(name string, shape []int) ([]int, int, error) {
	if len(shape) > maxRank {
		return nil, 0, fmt.Errorf("needle: tensor %q rank %d exceeds %d", name, len(shape), maxRank)
	}
	if len(shape) == 0 {
		return []int{}, 1, nil
	}
	out := append([]int{}, shape...)
	elements := 1
	limit := maxInt()
	for i, dim := range out {
		if dim <= 0 {
			return nil, 0, fmt.Errorf("needle: tensor %q has invalid dimension %d at axis %d", name, dim, i)
		}
		if elements > limit/dim {
			return nil, 0, fmt.Errorf("needle: tensor %q shape %v overflows element count", name, shape)
		}
		elements *= dim
	}
	return out, elements, nil
}

func decodeFloat32(name, dtype string, raw []byte) ([]float32, error) {
	switch dtype {
	case "F32":
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("needle: tensor %q F32 byte length %d is not divisible by 4", name, len(raw))
		}
		n := len(raw) / 4
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, nil
	case "F16":
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("needle: tensor %q F16 byte length %d is not divisible by 2", name, len(raw))
		}
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = half.F16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, nil
	case "BF16":
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("needle: tensor %q BF16 byte length %d is not divisible by 2", name, len(raw))
		}
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("needle: tensor %q unsupported dtype %q", name, dtype)
	}
}

func validateFinite(name string, data []float32) error {
	for _, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("needle: tensor %q contains non-finite value", name)
		}
	}
	return nil
}

func dtypeSize(dtype string) (int, bool) {
	switch dtype {
	case "F32":
		return 4, true
	case "F16", "BF16":
		return 2, true
	default:
		return 0, false
	}
}

func decodeJSONStrict(raw []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("trailing data")
		}
		return err
	}
	return nil
}

func checkedAddInt64(a, b int64) (int64, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a > math.MaxInt64-b {
		return 0, false
	}
	return a + b, true
}

func checkedMulInt64(a, b int64) (int64, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a == 0 || b == 0 {
		return 0, true
	}
	if a > math.MaxInt64/b {
		return 0, false
	}
	return a * b, true
}

func maxInt() int {
	return int(^uint(0) >> 1)
}
