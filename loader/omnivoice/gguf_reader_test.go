package omnivoice

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestOpenGGUFWeightsRoundTripBackboneFixture(t *testing.T) {
	sourceDir := filepath.Join("..", "..", "testdata", "omnivoice", "backbone")
	source, err := OpenWeights(sourceDir)
	if err != nil {
		t.Fatalf("OpenWeights source: %v", err)
	}
	defer source.Close()

	configJSON, err := os.ReadFile(filepath.Join(sourceDir, "config.json"))
	if err != nil {
		t.Fatalf("read config.json: %v", err)
	}

	infos := source.file.TensorInfos()
	names := source.file.Names()
	tensors := make([]ggufTensorFixture, 0, len(names)-1)
	for _, name := range names {
		if name == "codebook_layer_offsets" {
			continue
		}
		info := infos[name]
		raw, dtype, shape, err := source.file.GetRaw(name)
		if err != nil {
			t.Fatalf("source raw %s: %v", name, err)
		}
		qtype, err := testQTypeForDType(dtype)
		if err != nil {
			t.Fatalf("qtype %s: %v", name, err)
		}
		tensors = append(tensors, ggufTensorFixture{
			spec: gguf.TensorSpec{Name: name, Shape: reverseIntsToUint64(shape), QType: qtype},
			raw:  append([]byte(nil), raw...),
		})
		if info.DType != dtype {
			t.Fatalf("info dtype %s=%s raw dtype=%s", name, info.DType, dtype)
		}
	}

	ggufPath := filepath.Join(t.TempDir(), "backbone.gguf")
	writeGGUFFixture(t, ggufPath, []gguf.MetadataEntry{
		{Key: "general.architecture", Value: "omnivoice"},
		{Key: "omnivoice.schema_version", Value: uint32(1)},
		{Key: "omnivoice.config_json", Value: string(configJSON)},
	}, tensors)

	got, err := openGGUFWeights(ggufPath)
	if err != nil {
		t.Fatalf("openGGUFWeights: %v", err)
	}
	defer got.Close()

	if !reflect.DeepEqual(got.Config, source.Config) {
		t.Fatalf("config mismatch\n got=%+v\nwant=%+v", got.Config, source.Config)
	}
	if err := got.CheckCodebookOffsets(); err != nil {
		t.Fatalf("CheckCodebookOffsets: %v", err)
	}
	if meta := ValidateTensorInfos(got.Config, got.file.TensorInfos()); !meta.Valid {
		t.Fatalf("ValidateTensorInfos: %+v", meta)
	}
	if !reflect.DeepEqual(got.file.Names(), source.file.Names()) {
		t.Fatalf("names mismatch\n got=%v\nwant=%v", got.file.Names(), source.file.Names())
	}

	for _, name := range source.file.Names() {
		wantRaw, wantDType, wantShape, err := source.file.GetRaw(name)
		if err != nil {
			t.Fatalf("source GetRaw %s: %v", name, err)
		}
		gotRaw, gotDType, gotShape, err := got.file.GetRaw(name)
		if err != nil {
			t.Fatalf("got GetRaw %s: %v", name, err)
		}
		if gotDType != wantDType {
			t.Fatalf("%s dtype=%s want %s", name, gotDType, wantDType)
		}
		if !reflect.DeepEqual(gotShape, wantShape) {
			t.Fatalf("%s shape=%v want %v", name, gotShape, wantShape)
		}
		if !bytes.Equal(gotRaw, wantRaw) {
			t.Fatalf("%s raw mismatch", name)
		}
		wantF32, wantF32Shape, err := source.file.GetFloat32(name)
		if err != nil {
			t.Fatalf("source GetFloat32 %s: %v", name, err)
		}
		gotF32, gotF32Shape, err := got.file.GetFloat32(name)
		if err != nil {
			t.Fatalf("got GetFloat32 %s: %v", name, err)
		}
		if !reflect.DeepEqual(gotF32Shape, wantF32Shape) {
			t.Fatalf("%s float32 shape=%v want %v", name, gotF32Shape, wantF32Shape)
		}
		if !reflect.DeepEqual(gotF32, wantF32) {
			t.Fatalf("%s float32 mismatch", name)
		}
	}

	if err := got.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, _, _, err := got.file.GetRaw("llm.norm.weight"); err == nil {
		t.Fatal("GetRaw accepted closed reader")
	}
}

func TestOpenGGUFWeightsRejectsMetadataAndSchema(t *testing.T) {
	cfg := sampleConfig(t)
	baseMeta, baseTensors := buildTinyOmniVoiceGGUFFixture(t, cfg)

	t.Run("wrong architecture", func(t *testing.T) {
		meta := append([]gguf.MetadataEntry(nil), baseMeta...)
		meta[0].Value = "not-omnivoice"
		path := filepath.Join(t.TempDir(), "bad-arch.gguf")
		writeGGUFFixture(t, path, meta, baseTensors)
		assertOpenGGUFWeightsError(t, path, "general.architecture")
	})

	t.Run("missing schema version", func(t *testing.T) {
		meta := []gguf.MetadataEntry{baseMeta[0], baseMeta[2]}
		path := filepath.Join(t.TempDir(), "missing-schema.gguf")
		writeGGUFFixture(t, path, meta, baseTensors)
		assertOpenGGUFWeightsError(t, path, "omnivoice.schema_version")
	})

	t.Run("invalid config json", func(t *testing.T) {
		meta := append([]gguf.MetadataEntry(nil), baseMeta...)
		meta[2].Value = "{"
		path := filepath.Join(t.TempDir(), "bad-config-json.gguf")
		writeGGUFFixture(t, path, meta, baseTensors)
		assertOpenGGUFWeightsError(t, path, "parse omnivoice.config_json")
	})

	t.Run("invalid config payload", func(t *testing.T) {
		badCfg := cfg
		badCfg.ModelType = "wrong"
		raw, err := json.Marshal(badCfg)
		if err != nil {
			t.Fatal(err)
		}
		meta := append([]gguf.MetadataEntry(nil), baseMeta...)
		meta[2].Value = string(raw)
		path := filepath.Join(t.TempDir(), "bad-config.gguf")
		writeGGUFFixture(t, path, meta, baseTensors)
		assertOpenGGUFWeightsError(t, path, "validate omnivoice.config_json")
	})

	t.Run("missing tensor", func(t *testing.T) {
		tensors := append([]ggufTensorFixture(nil), baseTensors...)
		tensors = tensors[:len(tensors)-1]
		path := filepath.Join(t.TempDir(), "missing-tensor.gguf")
		writeGGUFFixture(t, path, baseMeta, tensors)
		assertOpenGGUFWeightsError(t, path, "checkpoint tensor layout mismatch")
	})

	t.Run("extra tensor", func(t *testing.T) {
		tensors := append([]ggufTensorFixture(nil), baseTensors...)
		tensors = append(tensors, ggufTensorFixture{
			spec: gguf.TensorSpec{Name: "unexpected.weight", Shape: []uint64{1}, QType: gguf.QuantF16},
			raw:  makeTestTensorRaw(t, gguf.QuantF16, []uint64{1}, 99),
		})
		path := filepath.Join(t.TempDir(), "extra-tensor.gguf")
		writeGGUFFixture(t, path, baseMeta, tensors)
		assertOpenGGUFWeightsError(t, path, "checkpoint tensor layout mismatch")
	})

	t.Run("unsupported dtype", func(t *testing.T) {
		tensors := cloneGGUFTensors(baseTensors)
		for i := range tensors {
			if tensors[i].spec.Name == "llm.norm.weight" {
				tensors[i].spec.QType = gguf.QuantBF16
				tensors[i].raw = makeTestTensorRaw(t, gguf.QuantBF16, tensors[i].spec.Shape, 7)
				break
			}
		}
		path := filepath.Join(t.TempDir(), "bad-dtype.gguf")
		writeGGUFFixture(t, path, baseMeta, tensors)
		assertOpenGGUFWeightsError(t, path, "only F32/F16 are supported")
	})

	t.Run("wrong shape", func(t *testing.T) {
		tensors := cloneGGUFTensors(baseTensors)
		for i := range tensors {
			if tensors[i].spec.Name == "llm.norm.weight" {
				tensors[i].spec.Shape = []uint64{5}
				tensors[i].raw = makeTestTensorRaw(t, tensors[i].spec.QType, tensors[i].spec.Shape, 5)
				break
			}
		}
		path := filepath.Join(t.TempDir(), "bad-shape.gguf")
		writeGGUFFixture(t, path, baseMeta, tensors)
		assertOpenGGUFWeightsError(t, path, "checkpoint tensor layout mismatch")
	})
}

func TestOpenGGUFWeightsRejectsMalformedTensorSpans(t *testing.T) {
	cfg := sampleConfig(t)
	meta, tensors := buildTinyOmniVoiceGGUFFixture(t, cfg)
	path := filepath.Join(t.TempDir(), "valid.gguf")
	writeGGUFFixture(t, path, meta, tensors)
	base, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read base gguf: %v", err)
	}
	headers, err := parseGGUFTensorHeaders(base)
	if err != nil {
		t.Fatalf("parse headers: %v", err)
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("open base gguf: %v", err)
	}
	dataLen := fileSize(t, path) - g.DataOffset
	g.Close()

	t.Run("duplicate name", func(t *testing.T) {
		data := append([]byte(nil), base...)
		patchTensorName(t, data, headers, "llm.embed_tokens.weight", "audio_embeddings.weight")
		bad := filepath.Join(t.TempDir(), "dup-name.gguf")
		if err := os.WriteFile(bad, data, 0o644); err != nil {
			t.Fatal(err)
		}
		assertOpenGGUFWeightsError(t, bad, "duplicate tensor name")
	})

	t.Run("unaligned offset", func(t *testing.T) {
		data := append([]byte(nil), base...)
		patchTensorOffset(t, data, headers, "llm.norm.weight", 1)
		bad := filepath.Join(t.TempDir(), "unaligned.gguf")
		if err := os.WriteFile(bad, data, 0o644); err != nil {
			t.Fatal(err)
		}
		assertOpenGGUFWeightsError(t, bad, "not 32-byte aligned")
	})

	t.Run("out of file", func(t *testing.T) {
		data := append([]byte(nil), base...)
		patchTensorOffset(t, data, headers, "llm.norm.weight", uint64((dataLen+31)/32*32))
		bad := filepath.Join(t.TempDir(), "oob.gguf")
		if err := os.WriteFile(bad, data, 0o644); err != nil {
			t.Fatal(err)
		}
		assertOpenGGUFWeightsError(t, bad, "exceeds GGUF data length")
	})

	t.Run("overlap", func(t *testing.T) {
		data := append([]byte(nil), base...)
		first := findTensorHeader(t, headers, "audio_embeddings.weight")
		patchTensorOffset(t, data, headers, "audio_heads.weight", readU64At(data, first.offsetPos))
		bad := filepath.Join(t.TempDir(), "overlap.gguf")
		if err := os.WriteFile(bad, data, 0o644); err != nil {
			t.Fatal(err)
		}
		assertOpenGGUFWeightsError(t, bad, "overlap")
	})
}

type ggufTensorFixture struct {
	spec gguf.TensorSpec
	raw  []byte
}

func buildTinyOmniVoiceGGUFFixture(t testing.TB, cfg Config) ([]gguf.MetadataEntry, []ggufTensorFixture) {
	t.Helper()
	specs, err := expectedTensorSpecs(cfg)
	if err != nil {
		t.Fatalf("expectedTensorSpecs: %v", err)
	}
	rawCfg, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	names := make([]string, 0, len(specs)-1)
	for name := range specs {
		if name != "codebook_layer_offsets" {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	tensors := make([]ggufTensorFixture, 0, len(names))
	for i, name := range names {
		spec := specs[name]
		qtype := gguf.QuantF16
		if name == "llm.norm.weight" || name == "llm.layers.0.self_attn.q_norm.weight" {
			qtype = gguf.QuantF32
		}
		shape := reverseInt64sToUint64(spec.Shape)
		tensors = append(tensors, ggufTensorFixture{
			spec: gguf.TensorSpec{Name: name, Shape: shape, QType: qtype},
			raw:  makeTestTensorRaw(t, qtype, shape, i+1),
		})
	}
	return []gguf.MetadataEntry{
		{Key: "general.architecture", Value: "omnivoice"},
		{Key: "omnivoice.schema_version", Value: uint32(1)},
		{Key: "omnivoice.config_json", Value: string(rawCfg)},
	}, tensors
}

func cloneGGUFTensors(in []ggufTensorFixture) []ggufTensorFixture {
	out := make([]ggufTensorFixture, len(in))
	for i, t := range in {
		out[i] = ggufTensorFixture{
			spec: gguf.TensorSpec{Name: t.spec.Name, Shape: append([]uint64(nil), t.spec.Shape...), QType: t.spec.QType},
			raw:  append([]byte(nil), t.raw...),
		}
	}
	return out
}

func writeGGUFFixture(t testing.TB, path string, meta []gguf.MetadataEntry, tensors []ggufTensorFixture) {
	t.Helper()
	specs := make([]gguf.TensorSpec, len(tensors))
	for i, tensor := range tensors {
		specs[i] = tensor.spec
	}
	if err := gguf.WriteV3(context.Background(), path, meta, specs, func(_ context.Context, index int, tensor gguf.TensorSpec) (io.Reader, error) {
		if index < 0 || index >= len(tensors) {
			return nil, fmt.Errorf("unexpected tensor index %d", index)
		}
		if tensor.Name != tensors[index].spec.Name {
			return nil, fmt.Errorf("tensor[%d] name=%q want %q", index, tensor.Name, tensors[index].spec.Name)
		}
		return bytes.NewReader(tensors[index].raw), nil
	}); err != nil {
		t.Fatalf("WriteV3 %s: %v", path, err)
	}
}

func assertOpenGGUFWeightsError(t testing.TB, path, want string) {
	t.Helper()
	_, err := openGGUFWeights(path)
	if err == nil {
		t.Fatalf("openGGUFWeights(%s) succeeded, want error containing %q", path, want)
	}
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("openGGUFWeights(%s) error=%q, want substring %q", path, err, want)
	}
}

func testQTypeForDType(dtype string) (gguf.QuantType, error) {
	switch dtype {
	case "F16":
		return gguf.QuantF16, nil
	case "F32":
		return gguf.QuantF32, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %q", dtype)
	}
}

func reverseIntsToUint64(shape []int) []uint64 {
	out := make([]uint64, len(shape))
	for i := range shape {
		out[i] = uint64(shape[len(shape)-1-i])
	}
	return out
}

func reverseInt64sToUint64(shape []int64) []uint64 {
	out := make([]uint64, len(shape))
	for i := range shape {
		out[i] = uint64(shape[len(shape)-1-i])
	}
	return out
}

func makeTestTensorRaw(t testing.TB, qtype gguf.QuantType, shape []uint64, seed int) []byte {
	t.Helper()
	n := int(testNumel(t, shape))
	switch qtype {
	case gguf.QuantF16:
		raw := make([]byte, n*2)
		for i := 0; i < n; i++ {
			v := float32(seed) + float32(i+1)/16
			binary.LittleEndian.PutUint16(raw[i*2:], half.F32ToF16(v))
		}
		return raw
	case gguf.QuantF32:
		raw := make([]byte, n*4)
		for i := 0; i < n; i++ {
			v := float32(seed) + float32(i+1)/8
			binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
		}
		return raw
	case gguf.QuantBF16:
		return make([]byte, n*2)
	default:
		t.Fatalf("unsupported qtype %s", qtype)
	}
	return nil
}

func testNumel(t testing.TB, shape []uint64) uint64 {
	t.Helper()
	n := uint64(1)
	for _, dim := range shape {
		if dim == 0 || n > ^uint64(0)/dim {
			t.Fatalf("bad shape %v", shape)
		}
		n *= dim
	}
	return n
}

type ggufTensorHeader struct {
	name      string
	namePos   int
	nameLen   int
	offsetPos int
}

func parseGGUFTensorHeaders(data []byte) ([]ggufTensorHeader, error) {
	if len(data) < 24 || string(data[:4]) != "GGUF" {
		return nil, fmt.Errorf("bad gguf header")
	}
	off := 4
	version, off, err := readU32(data, off)
	if err != nil {
		return nil, err
	}
	if version != 2 && version != 3 {
		return nil, fmt.Errorf("unsupported version %d", version)
	}
	nTensors, off, err := readU64(data, off)
	if err != nil {
		return nil, err
	}
	nKV, off, err := readU64(data, off)
	if err != nil {
		return nil, err
	}
	for i := uint64(0); i < nKV; i++ {
		_, off, err = readString(data, off)
		if err != nil {
			return nil, err
		}
		vtype, next, err := readU32(data, off)
		if err != nil {
			return nil, err
		}
		off = next
		off, err = skipGGUFValue(data, off, gguf.GGUFType(vtype))
		if err != nil {
			return nil, err
		}
	}
	headers := make([]ggufTensorHeader, 0, nTensors)
	for i := uint64(0); i < nTensors; i++ {
		nameLenPos := off
		name, next, err := readString(data, off)
		if err != nil {
			return nil, err
		}
		off = next
		ndims, next, err := readU32(data, off)
		if err != nil {
			return nil, err
		}
		off = next + int(ndims)*8
		if off+12 > len(data) {
			return nil, fmt.Errorf("tensor %q header truncated", name)
		}
		off += 4
		headers = append(headers, ggufTensorHeader{name: name, namePos: nameLenPos + 8, nameLen: len(name), offsetPos: off})
		off += 8
	}
	return headers, nil
}

func skipGGUFValue(data []byte, off int, typ gguf.GGUFType) (int, error) {
	switch typ {
	case gguf.GGUFTypeU8, gguf.GGUFTypeI8, gguf.GGUFTypeBool:
		return requireBytes(data, off, 1)
	case gguf.GGUFTypeU16, gguf.GGUFTypeI16:
		return requireBytes(data, off, 2)
	case gguf.GGUFTypeU32, gguf.GGUFTypeI32, gguf.GGUFTypeF32:
		return requireBytes(data, off, 4)
	case gguf.GGUFTypeU64, gguf.GGUFTypeI64, gguf.GGUFTypeF64:
		return requireBytes(data, off, 8)
	case gguf.GGUFTypeString:
		_, next, err := readString(data, off)
		return next, err
	case gguf.GGUFTypeArray:
		elemType, next, err := readU32(data, off)
		if err != nil {
			return 0, err
		}
		count, next, err := readU64(data, next)
		if err != nil {
			return 0, err
		}
		off = next
		for i := uint64(0); i < count; i++ {
			off, err = skipGGUFValue(data, off, gguf.GGUFType(elemType))
			if err != nil {
				return 0, err
			}
		}
		return off, nil
	default:
		return 0, fmt.Errorf("unsupported metadata type %d", typ)
	}
}

func requireBytes(data []byte, off, n int) (int, error) {
	if off < 0 || n < 0 || off > len(data)-n {
		return 0, fmt.Errorf("buffer underflow off=%d n=%d len=%d", off, n, len(data))
	}
	return off + n, nil
}

func readU32(data []byte, off int) (uint32, int, error) {
	next, err := requireBytes(data, off, 4)
	if err != nil {
		return 0, 0, err
	}
	return binary.LittleEndian.Uint32(data[off:next]), next, nil
}

func readU64(data []byte, off int) (uint64, int, error) {
	next, err := requireBytes(data, off, 8)
	if err != nil {
		return 0, 0, err
	}
	return binary.LittleEndian.Uint64(data[off:next]), next, nil
}

func readString(data []byte, off int) (string, int, error) {
	n, next, err := readU64(data, off)
	if err != nil {
		return "", 0, err
	}
	if n > uint64(len(data)-next) {
		return "", 0, fmt.Errorf("string length %d exceeds remaining %d", n, len(data)-next)
	}
	end := next + int(n)
	return string(data[next:end]), end, nil
}

func patchTensorName(t testing.TB, data []byte, headers []ggufTensorHeader, target, replacement string) {
	t.Helper()
	h := findTensorHeader(t, headers, target)
	if len(replacement) != h.nameLen {
		t.Fatalf("replacement name %q len=%d want %d", replacement, len(replacement), h.nameLen)
	}
	copy(data[h.namePos:h.namePos+h.nameLen], []byte(replacement))
}

func patchTensorOffset(t testing.TB, data []byte, headers []ggufTensorHeader, name string, offset uint64) {
	t.Helper()
	h := findTensorHeader(t, headers, name)
	binary.LittleEndian.PutUint64(data[h.offsetPos:h.offsetPos+8], offset)
}

func findTensorHeader(t testing.TB, headers []ggufTensorHeader, name string) ggufTensorHeader {
	t.Helper()
	for _, h := range headers {
		if h.name == name {
			return h
		}
	}
	t.Fatalf("tensor header %q not found", name)
	return ggufTensorHeader{}
}

func readU64At(data []byte, off int) uint64 {
	return binary.LittleEndian.Uint64(data[off : off+8])
}

func fileSize(t testing.TB, path string) int64 {
	t.Helper()
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	return fi.Size()
}
