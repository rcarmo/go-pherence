package needle

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

type fixtureTensor struct {
	DType string
	Shape []int
	Data  []byte
}

func TestRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "model.safetensors")
	want := &Checkpoint{
		Config:        json.RawMessage(`{"arch":"tiny","layers":2}`),
		FormatVersion: 3,
		Tensors: map[string]Tensor{
			"z":      {Shape: []int{2}, Data: []float32{1.25, -2.5}},
			"a":      {Shape: []int{}, Data: []float32{7}},
			"middle": {Shape: []int{1, 2}, Data: []float32{3.5, 4.5}},
		},
	}
	if err := Save(path, want); err != nil {
		t.Fatalf("Save: %v", err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.FormatVersion != want.FormatVersion {
		t.Fatalf("FormatVersion=%d want %d", got.FormatVersion, want.FormatVersion)
	}
	if string(got.Config) != string(want.Config) {
		t.Fatalf("Config=%s want %s", got.Config, want.Config)
	}
	if !reflect.DeepEqual(got.Tensors, want.Tensors) {
		t.Fatalf("Tensors=%#v want %#v", got.Tensors, want.Tensors)
	}
	header := readHeaderString(t, path)
	for _, token := range []string{`"__metadata__":`, `"a":{"dtype":`, `"middle":{"dtype":`, `"z":{"dtype":`} {
		if !strings.Contains(header, token) {
			t.Fatalf("header %q missing %q", header, token)
		}
	}
	if !(strings.Index(header, `"__metadata__":`) < strings.Index(header, `"a":{"dtype":`) &&
		strings.Index(header, `"a":{"dtype":`) < strings.Index(header, `"middle":{"dtype":`) &&
		strings.Index(header, `"middle":{"dtype":`) < strings.Index(header, `"z":{"dtype":`)) {
		t.Fatalf("header order not deterministic: %s", header)
	}
}

func TestLoadMetadataAndDTypes(t *testing.T) {
	path := writeFixture(t, map[string]string{
		"format_version": "2",
		"config":         `{"backend":"cpu"}`,
		"step":           "17",
		"run":            `{"seed":1}`,
	}, map[string]fixtureTensor{
		"bf16":   {DType: "BF16", Shape: []int{2}, Data: bf16Bytes(1.5, -2.25)},
		"f16":    {DType: "F16", Shape: []int{2}, Data: f16Bytes(2.0, -4.0)},
		"scalar": {DType: "F32", Shape: []int{}, Data: f32Bytes(9.25)},
	})
	got, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.FormatVersion != 2 {
		t.Fatalf("FormatVersion=%d want 2", got.FormatVersion)
	}
	if string(got.Config) != `{"backend":"cpu"}` {
		t.Fatalf("Config=%s", got.Config)
	}
	want := map[string]Tensor{
		"bf16":   {Shape: []int{2}, Data: []float32{1.5, -2.25}},
		"f16":    {Shape: []int{2}, Data: []float32{2.0, -4.0}},
		"scalar": {Shape: []int{}, Data: []float32{9.25}},
	}
	if !reflect.DeepEqual(got.Tensors, want) {
		t.Fatalf("Tensors=%#v want %#v", got.Tensors, want)
	}
}

func TestLoadRejectsMalformedMetadataAndShapes(t *testing.T) {
	cases := []struct {
		name string
		path func(*testing.T) string
		want string
	}{
		{
			name: "missing metadata",
			path: func(t *testing.T) string {
				return writeFixture(t, nil, map[string]fixtureTensor{
					"x": {DType: "F32", Shape: []int{1}, Data: f32Bytes(1)},
				})
			},
			want: "missing __metadata__",
		},
		{
			name: "missing format version",
			path: func(t *testing.T) string {
				return writeFixture(t, map[string]string{"config": `{}`}, map[string]fixtureTensor{
					"x": {DType: "F32", Shape: []int{1}, Data: f32Bytes(1)},
				})
			},
			want: "missing format_version",
		},
		{
			name: "bad format version",
			path: func(t *testing.T) string {
				return writeFixture(t, map[string]string{"format_version": "x", "config": `{}`}, map[string]fixtureTensor{
					"x": {DType: "F32", Shape: []int{1}, Data: f32Bytes(1)},
				})
			},
			want: "invalid format_version",
		},
		{
			name: "bad config json",
			path: func(t *testing.T) string {
				return writeFixture(t, map[string]string{"format_version": "1", "config": "{"}, map[string]fixtureTensor{
					"x": {DType: "F32", Shape: []int{1}, Data: f32Bytes(1)},
				})
			},
			want: "invalid config JSON",
		},
		{
			name: "unsupported dtype",
			path: func(t *testing.T) string {
				return writeFixture(t, map[string]string{"format_version": "1", "config": `{}`}, map[string]fixtureTensor{
					"x": {DType: "I32", Shape: []int{1}, Data: []byte{0, 0, 0, 1}},
				})
			},
			want: "unsupported dtype",
		},
		{
			name: "zero dimension",
			path: func(t *testing.T) string {
				return writeFixture(t, map[string]string{"format_version": "1", "config": `{}`}, map[string]fixtureTensor{
					"x": {DType: "F32", Shape: []int{0}, Data: nil},
				})
			},
			want: "invalid dimension 0",
		},
		{
			name: "rank too large",
			path: func(t *testing.T) string {
				return writeFixture(t, map[string]string{"format_version": "1", "config": `{}`}, map[string]fixtureTensor{
					"x": {DType: "F32", Shape: []int{1, 1, 1, 1, 1, 1, 1, 1, 1}, Data: f32Bytes(1)},
				})
			},
			want: "rank 9 exceeds 8",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Load(tc.path(t)); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Load err=%v want substring %q", err, tc.want)
			}
		})
	}
}

func TestLoadRejectsBoundsAndTruncation(t *testing.T) {
	cases := []struct {
		name string
		path func(*testing.T) string
		want string
	}{
		{
			name: "truncated tensor data",
			path: func(t *testing.T) string {
				return writeRaw(t, `{"__metadata__":{"format_version":"1","config":"{}"},"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, f32Bytes(1))
			},
			want: "invalid data offsets",
		},
		{
			name: "negative offsets",
			path: func(t *testing.T) string {
				return writeRaw(t, `{"__metadata__":{"format_version":"1","config":"{}"},"x":{"dtype":"F32","shape":[1],"data_offsets":[-1,3]}}`, f32Bytes(1))
			},
			want: "invalid data offsets",
		},
		{
			name: "overlap",
			path: func(t *testing.T) string {
				return writeRaw(t, `{"__metadata__":{"format_version":"1","config":"{}"},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F16","shape":[2],"data_offsets":[2,6]}}`, append(f32Bytes(1), []byte{0, 0}...))
			},
			want: "overlap",
		},
		{
			name: "huge product",
			path: func(t *testing.T) string {
				return writeRaw(t, `{"__metadata__":{"format_version":"1","config":"{}"},"x":{"dtype":"F32","shape":[1073741824,2],"data_offsets":[0,8]}}`, f32Bytes(1, 2))
			},
			want: "exceeds",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Load(tc.path(t)); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Load err=%v want substring %q", err, tc.want)
			}
		})
	}
}

func TestLoadRejectsOversizeHeaderAndFile(t *testing.T) {
	headerPath := filepath.Join(t.TempDir(), "huge-header.safetensors")
	f, err := os.Create(headerPath)
	if err != nil {
		t.Fatal(err)
	}
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(maxHeaderBytes+1))
	if _, err := f.Write(lenBuf[:]); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Truncate(8 + maxHeaderBytes + 1); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(headerPath); err == nil || !strings.Contains(err.Error(), "header size") {
		t.Fatalf("Load err=%v want header size error", err)
	}

	filePath := filepath.Join(t.TempDir(), "huge-file.safetensors")
	f, err = os.Create(filePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(maxFileBytes + 1); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(filePath); err == nil || !strings.Contains(err.Error(), "file size") {
		t.Fatalf("Load err=%v want file size error", err)
	}
}

func TestLoadAndSaveRejectNonFinite(t *testing.T) {
	loadPath := writeFixture(t, map[string]string{"format_version": "1", "config": `{}`}, map[string]fixtureTensor{
		"x": {DType: "F32", Shape: []int{1}, Data: f32Bytes(float32(math.Inf(1)))},
	})
	if _, err := Load(loadPath); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("Load err=%v want non-finite", err)
	}

	savePath := filepath.Join(t.TempDir(), "bad.safetensors")
	if err := Save(savePath, &Checkpoint{
		Config:        json.RawMessage(`{}`),
		FormatVersion: 1,
		Tensors: map[string]Tensor{
			"x": {Shape: []int{1}, Data: []float32{float32(math.NaN())}},
		},
	}); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("Save err=%v want non-finite", err)
	}
}

func TestSaveRejectsInvalidCheckpoint(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.safetensors")
	cases := []struct {
		name string
		cp   *Checkpoint
		want string
	}{
		{
			name: "nil checkpoint",
			cp:   nil,
			want: "nil checkpoint",
		},
		{
			name: "bad config",
			cp: &Checkpoint{
				Config:        json.RawMessage(`{"`),
				FormatVersion: 1,
				Tensors:       map[string]Tensor{},
			},
			want: "invalid config JSON",
		},
		{
			name: "shape mismatch",
			cp: &Checkpoint{
				Config:        json.RawMessage(`{}`),
				FormatVersion: 1,
				Tensors: map[string]Tensor{
					"x": {Shape: []int{2}, Data: []float32{1}},
				},
			},
			want: "has 1 values, want 2",
		},
		{
			name: "zero dim",
			cp: &Checkpoint{
				Config:        json.RawMessage(`{}`),
				FormatVersion: 1,
				Tensors: map[string]Tensor{
					"x": {Shape: []int{0}, Data: nil},
				},
			},
			want: "invalid dimension 0",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := Save(path, tc.cp); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Save err=%v want substring %q", err, tc.want)
			}
		})
	}
}

func writeFixture(t *testing.T, metadata map[string]string, tensors map[string]fixtureTensor) string {
	t.Helper()
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sortStrings(names)
	header := make(map[string]any, len(tensors)+1)
	if metadata != nil {
		header["__metadata__"] = metadata
	}
	data := make([]byte, 0)
	offset := 0
	for _, name := range names {
		tensor := tensors[name]
		shape := append([]int{}, tensor.Shape...)
		start := offset
		offset += len(tensor.Data)
		header[name] = map[string]any{
			"dtype":        tensor.DType,
			"shape":        shape,
			"data_offsets": [2]int{start, offset},
		}
		data = append(data, tensor.Data...)
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("Marshal header: %v", err)
	}
	return writeRaw(t, string(raw), data)
}

func writeRaw(t *testing.T, header string, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.safetensors")
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(header)))
	payload := append(append(lenBuf[:], []byte(header)...), data...)
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	return path
}

func readHeaderString(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if len(data) < 8 {
		t.Fatalf("short file: %d", len(data))
	}
	n := binary.LittleEndian.Uint64(data[:8])
	if int(8+n) > len(data) {
		t.Fatalf("short header: %d", n)
	}
	return string(data[8 : 8+n])
}

func f32Bytes(values ...float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func f16Bytes(values ...float32) []byte {
	out := make([]byte, len(values)*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(out[i*2:], half.F32ToF16(v))
	}
	return out
}

func bf16Bytes(values ...float32) []byte {
	out := make([]byte, len(values)*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(out[i*2:], half.F32ToBF16(v))
	}
	return out
}

func sortStrings(values []string) {
	for i := 0; i < len(values); i++ {
		for j := i + 1; j < len(values); j++ {
			if values[j] < values[i] {
				values[i], values[j] = values[j], values[i]
			}
		}
	}
}

func TestUnsignedHeaderCannotWrap(t *testing.T) {
	path := filepath.Join(t.TempDir(), "wrapped.safetensors")
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], math.MaxUint64)
	if err := os.WriteFile(path, b[:], 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("wrapped header accepted")
	}
}
