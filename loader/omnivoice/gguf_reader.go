package omnivoice

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sort"

	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const ggufTensorAlignment = 32

type ggufReader struct {
	names  []string
	infos  map[string]safetensors.TensorInfo
	raws   map[string][]byte
	closed bool
}

func openGGUFWeights(path string) (*Weights, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("omnivoice gguf: open %s: %w", path, err)
	}
	defer g.Close()

	cfg, err := ggufOmniVoiceConfig(g)
	if err != nil {
		return nil, err
	}
	r, err := newGGUFReader(path, g, cfg)
	if err != nil {
		return nil, err
	}
	w := &Weights{file: r, Config: cfg}
	if err := w.CheckCodebookOffsets(); err != nil {
		_ = r.Close()
		return nil, err
	}
	return w, nil
}

func ggufOmniVoiceConfig(g *gguf.GGUF) (Config, error) {
	if _, exists := g.Meta["general.alignment"]; exists {
		alignment, ok := g.MetaUint32("general.alignment")
		if !ok || alignment != 32 {
			return Config{}, fmt.Errorf("omnivoice gguf: only 32-byte alignment supported")
		}
	}
	arch, ok := g.MetaString("general.architecture")
	if !ok {
		return Config{}, fmt.Errorf("omnivoice gguf: missing metadata general.architecture")
	}
	if arch != "omnivoice" {
		return Config{}, fmt.Errorf("omnivoice gguf: general.architecture=%q, want %q", arch, "omnivoice")
	}
	schema, ok := g.MetaUint32("omnivoice.schema_version")
	if !ok {
		return Config{}, fmt.Errorf("omnivoice gguf: missing metadata omnivoice.schema_version")
	}
	if schema != 1 {
		return Config{}, fmt.Errorf("omnivoice gguf: omnivoice.schema_version=%d, want 1", schema)
	}
	rawConfig, ok := g.MetaString("omnivoice.config_json")
	if !ok {
		return Config{}, fmt.Errorf("omnivoice gguf: missing metadata omnivoice.config_json")
	}
	var cfg Config
	if err := json.Unmarshal([]byte(rawConfig), &cfg); err != nil {
		return Config{}, fmt.Errorf("omnivoice gguf: parse omnivoice.config_json: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, fmt.Errorf("omnivoice gguf: validate omnivoice.config_json: %w", err)
	}
	// Bound model-derived allocation independently of header byte limits.
	if cfg.LLMConfig.NumHiddenLayers > 4096 || cfg.NumAudioCodebook > 4096 {
		return Config{}, fmt.Errorf("omnivoice gguf: layer/codebook count exceeds supported limit 4096")
	}
	return cfg, nil
}

func newGGUFReader(path string, g *gguf.GGUF, cfg Config) (*ggufReader, error) {
	specs, err := expectedTensorSpecs(cfg)
	if err != nil {
		return nil, fmt.Errorf("omnivoice gguf: expected schema: %w", err)
	}
	fi, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("omnivoice gguf: stat %s: %w", path, err)
	}
	if g.DataOffset < 0 {
		return nil, fmt.Errorf("omnivoice gguf: negative data offset %d", g.DataOffset)
	}
	if g.DataOffset%ggufTensorAlignment != 0 {
		return nil, fmt.Errorf("omnivoice gguf: data offset %d is not %d-byte aligned", g.DataOffset, ggufTensorAlignment)
	}
	if g.DataOffset > fi.Size() {
		return nil, fmt.Errorf("omnivoice gguf: data offset %d exceeds file size %d", g.DataOffset, fi.Size())
	}
	dataLen := fi.Size() - g.DataOffset

	infos := make(map[string]safetensors.TensorInfo, len(g.Tensors)+1)
	byName := make(map[string]gguf.TensorInfo, len(g.Tensors))
	spans := make([]ggufTensorSpan, 0, len(g.Tensors))
	for _, t := range g.Tensors {
		if t.Name == "" {
			return nil, fmt.Errorf("omnivoice gguf: empty tensor name")
		}
		if _, dup := byName[t.Name]; dup {
			return nil, fmt.Errorf("omnivoice gguf: duplicate tensor name %q", t.Name)
		}
		byName[t.Name] = t

		dtype, elemSize, err := ggufTensorDType(t.QType)
		if err != nil {
			return nil, fmt.Errorf("omnivoice gguf: tensor %q: %w", t.Name, err)
		}
		if t.Name == "codebook_layer_offsets" {
			return nil, fmt.Errorf("omnivoice gguf: tensor %q must be omitted from GGUF and synthesized from config", t.Name)
		}

		rawBytes, err := ggufTensorByteLen(t.Shape, elemSize)
		if err != nil {
			return nil, fmt.Errorf("omnivoice gguf: tensor %q: %w", t.Name, err)
		}
		if t.Offset%ggufTensorAlignment != 0 {
			return nil, fmt.Errorf("omnivoice gguf: tensor %q offset %d is not %d-byte aligned", t.Name, t.Offset, ggufTensorAlignment)
		}
		start, end, err := ggufTensorBounds(t.Offset, rawBytes, dataLen)
		if err != nil {
			return nil, fmt.Errorf("omnivoice gguf: tensor %q: %w", t.Name, err)
		}
		spans = append(spans, ggufTensorSpan{name: t.Name, start: start, end: end})

		spec, ok := specs[t.Name]
		if !ok {
			infos[t.Name] = safetensors.TensorInfo{DType: dtype, DataOffsets: [2]int{0, int(rawBytes)}}
			continue
		}
		info := safetensors.TensorInfo{DType: dtype, DataOffsets: [2]int{0, int(rawBytes)}}
		if len(t.Shape) == len(spec.Shape) {
			shape := make([]int, len(spec.Shape))
			for i := range spec.Shape {
				dim := t.Shape[len(t.Shape)-1-i]
				if dim == 0 {
					return nil, fmt.Errorf("omnivoice gguf: tensor %q has zero dimension at index %d", t.Name, len(t.Shape)-1-i)
				}
				if dim > uint64(int(^uint(0)>>1)) {
					return nil, fmt.Errorf("omnivoice gguf: tensor %q dimension %d exceeds int", t.Name, dim)
				}
				shape[i] = int(dim)
			}
			info.Shape = shape
		}
		infos[t.Name] = info
	}
	if err := validateNonOverlappingGGUFSpans(spans); err != nil {
		return nil, err
	}

	codebookRaw, codebookInfo, err := synthesizeCodebookOffsets(cfg)
	if err != nil {
		return nil, err
	}
	infos["codebook_layer_offsets"] = codebookInfo

	meta := ValidateTensorInfos(cfg, infos)
	if !meta.Valid {
		return nil, fmt.Errorf("omnivoice gguf: checkpoint tensor layout mismatch: %+v", meta)
	}

	raws := make(map[string][]byte, len(specs))
	for name := range specs {
		if name == "codebook_layer_offsets" {
			raws[name] = codebookRaw
			continue
		}
		t := byName[name]
		raw, err := g.Raw(t)
		if err != nil {
			return nil, fmt.Errorf("omnivoice gguf: tensor %q raw: %w", name, err)
		}
		if len(raw) != infos[name].DataOffsets[1] {
			return nil, fmt.Errorf("omnivoice gguf: tensor %q raw length %d, want %d", name, len(raw), infos[name].DataOffsets[1])
		}
		raws[name] = raw
	}

	names := make([]string, 0, len(infos))
	for name := range infos {
		names = append(names, name)
	}
	sort.Strings(names)
	return &ggufReader{names: names, infos: infos, raws: raws}, nil
}

func validateNonOverlappingGGUFSpans(spans []ggufTensorSpan) error {
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].start == spans[j].start {
			return spans[i].name < spans[j].name
		}
		return spans[i].start < spans[j].start
	})
	for i := 1; i < len(spans); i++ {
		if spans[i].start < spans[i-1].end {
			return fmt.Errorf("omnivoice gguf: tensors %q [%d,%d) and %q [%d,%d) overlap", spans[i-1].name, spans[i-1].start, spans[i-1].end, spans[i].name, spans[i].start, spans[i].end)
		}
	}
	return nil
}

func ggufTensorDType(qt gguf.QuantType) (string, int64, error) {
	switch qt {
	case gguf.QuantF32:
		return "F32", 4, nil
	case gguf.QuantF16:
		return "F16", 2, nil
	default:
		return "", 0, fmt.Errorf("unsupported tensor type %s; only F32/F16 are supported", qt)
	}
}

func ggufTensorByteLen(shape []uint64, elemSize int64) (int64, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("empty shape")
	}
	numel := uint64(1)
	maxUint64 := ^uint64(0)
	for i, dim := range shape {
		if dim == 0 {
			return 0, fmt.Errorf("zero dimension at index %d", i)
		}
		if numel > maxUint64/dim {
			return 0, fmt.Errorf("shape %v element count overflow", shape)
		}
		numel *= dim
	}
	if numel > uint64(^uint(0)>>1)/uint64(elemSize) {
		return 0, fmt.Errorf("shape %v byte size overflow", shape)
	}
	return int64(numel) * elemSize, nil
}

func ggufTensorBounds(offset uint64, rawBytes, dataLen int64) (int64, int64, error) {
	if rawBytes < 0 {
		return 0, 0, fmt.Errorf("negative raw byte length %d", rawBytes)
	}
	if offset > uint64(^uint64(0)>>1) {
		return 0, 0, fmt.Errorf("offset %d exceeds int64", offset)
	}
	start := int64(offset)
	end, ok := ggufCheckedAddInt64(start, rawBytes)
	if !ok {
		return 0, 0, fmt.Errorf("offset %d + raw length %d overflows", offset, rawBytes)
	}
	if end > dataLen {
		return 0, 0, fmt.Errorf("span [%d,%d) exceeds GGUF data length %d", start, end, dataLen)
	}
	return start, end, nil
}

type ggufTensorSpan struct {
	name       string
	start, end int64
}

func synthesizeCodebookOffsets(cfg Config) ([]byte, safetensors.TensorInfo, error) {
	if cfg.NumAudioCodebook <= 0 {
		return nil, safetensors.TensorInfo{}, fmt.Errorf("omnivoice gguf: invalid num_audio_codebook %d", cfg.NumAudioCodebook)
	}
	if cfg.AudioVocabSize <= 0 {
		return nil, safetensors.TensorInfo{}, fmt.Errorf("omnivoice gguf: invalid audio_vocab_size %d", cfg.AudioVocabSize)
	}
	raw := make([]byte, cfg.NumAudioCodebook*8)
	for i := 0; i < cfg.NumAudioCodebook; i++ {
		v, ok := checkedMulInt64(int64(i), int64(cfg.AudioVocabSize))
		if !ok {
			return nil, safetensors.TensorInfo{}, fmt.Errorf("omnivoice gguf: codebook offset overflow at %d", i)
		}
		binary.LittleEndian.PutUint64(raw[i*8:], uint64(v))
	}
	return raw, safetensors.TensorInfo{
		DType:       "I64",
		Shape:       []int{cfg.NumAudioCodebook},
		DataOffsets: [2]int{0, len(raw)},
	}, nil
}

func (r *ggufReader) Close() error {
	if r == nil || r.closed {
		return nil
	}
	r.closed = true
	r.names = nil
	r.infos = nil
	r.raws = nil
	return nil
}

func (r *ggufReader) Names() []string {
	if r == nil || r.closed {
		return nil
	}
	out := make([]string, len(r.names))
	copy(out, r.names)
	return out
}

func (r *ggufReader) TensorInfos() map[string]safetensors.TensorInfo {
	if r == nil || r.closed {
		return nil
	}
	out := make(map[string]safetensors.TensorInfo, len(r.infos))
	for name, info := range r.infos {
		shape := append([]int(nil), info.Shape...)
		out[name] = safetensors.TensorInfo{DType: info.DType, Shape: shape, DataOffsets: info.DataOffsets}
	}
	return out
}

func (r *ggufReader) GetRaw(name string) ([]byte, string, []int, error) {
	if r == nil || r.closed {
		return nil, "", nil, fmt.Errorf("omnivoice gguf: reader closed")
	}
	info, ok := r.infos[name]
	if !ok {
		return nil, "", nil, fmt.Errorf("omnivoice gguf: tensor %q not found", name)
	}
	raw, ok := r.raws[name]
	if !ok {
		return nil, "", nil, fmt.Errorf("omnivoice gguf: tensor %q bytes not loaded", name)
	}
	// Borrowed immutable metadata, matching safetensors.GetRaw. Copying here
	// would allocate for every streamed layer tensor and matrix row access.
	return raw, info.DType, info.Shape, nil
}

func (r *ggufReader) GetFloat32(name string) ([]float32, []int, error) {
	raw, dtype, shape, err := r.GetRaw(name)
	if err != nil {
		return nil, nil, err
	}
	switch dtype {
	case "I64":
		if len(raw)%8 != 0 {
			return nil, nil, fmt.Errorf("omnivoice gguf: tensor %q I64 byte length %d is not divisible by 8", name, len(raw))
		}
		out := make([]float32, len(raw)/8)
		for i := range out {
			out[i] = float32(int64(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out, shape, nil
	case "F16", "F32":
		count := 1
		for _, dim := range shape {
			if dim <= 0 || count > int(^uint(0)>>1)/dim {
				return nil, nil, fmt.Errorf("omnivoice gguf: tensor %q invalid shape %v", name, shape)
			}
			count *= dim
		}
		out := make([]float32, count)
		if err := convertInto(out, raw, dtype); err != nil {
			return nil, nil, err
		}
		return out, shape, nil
	default:
		return nil, nil, fmt.Errorf("omnivoice gguf: unsupported dtype %q for tensor %q", dtype, name)
	}
}

func ggufCheckedAddInt64(a, b int64) (int64, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a > int64(^uint64(0)>>1)-b {
		return 0, false
	}
	return a + b, true
}
