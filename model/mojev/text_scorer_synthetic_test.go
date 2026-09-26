package mojev

import (
	"bytes"
	"errors"
	"math"
	"os"
	"reflect"
	"runtime"
	"testing"
	"weak"
)

func TestTextScorerLoaderFailures(t *testing.T) {
	config, err := os.ReadFile("testdata/mojev_config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, data := range [][]byte{nil, []byte(`{}`), config} {
		if got, err := LoadTextScorer(nil, data); err == nil || got != nil {
			t.Fatal("nil source accepted")
		}
	}
	src := &textTestSource{err: errors.New("missing tensor")}
	if got, err := LoadTextScorer(src, []byte(`{}`)); err == nil || got != nil {
		t.Fatal("bad config accepted")
	}
	if got, err := LoadTextScorer(src, config); err == nil || got != nil {
		t.Fatal("missing tensors accepted")
	}
	for _, bad := range [][]byte{
		bytes.Replace(config, []byte(`"rms_norm_eps": 1e-06`), []byte(`"rms_norm_eps": 0`), 1),
		bytes.Replace(config, []byte(`"rope_theta": 10000000`), []byte(`"rope_theta": 0`), 1),
	} {
		if bytes.Equal(bad, config) {
			t.Fatal("config mutation missed")
		}
		if got, err := LoadTextScorer(src, bad); err == nil || got != nil {
			t.Fatal("invalid arithmetic accepted")
		}
	}
	var scorer *TextScorer
	if got, err := scorer.ScoreEncoded(repairedTextRow()); err == nil || got != nil {
		t.Fatal("nil scorer accepted")
	}
	if got, err := (&TextScorer{}).ScoreEncoded(repairedTextRow()); err == nil || got != nil {
		t.Fatal("empty scorer accepted")
	}
}

// Observe the converted allocation without extending its lifetime. Closing
// the source invalidates its own storage, not the conversion buffer it owns
// and transfers to its caller via weights.Source.GetFloat32.
type trackedTextSource struct {
	*textTestSource
	converted weak.Pointer[float32]
}

func (s *trackedTextSource) GetFloat32(name string) ([]float32, []int, error) {
	data, shape, err := s.textTestSource.GetFloat32(name)
	if len(data) > 0 {
		s.converted = weak.Make(&data[0])
	}
	return data, shape, err
}

func (s *trackedTextSource) Close() error {
	clear(s.data)
	s.shape[0] = 0
	return nil
}

func TestMoJevTextTensorOwnedTransfer(t *testing.T) {
	src := &trackedTextSource{textTestSource: &textTestSource{dtype: "BF16", shape: []int{2}, data: []float32{1, 2}}}
	shape := []int{2}
	x, err := (mojevTextSource{src}).Get("model.norm.weight", shape)
	if err != nil {
		t.Fatal(err)
	}
	if src.converted.Value() != &x.Data()[0] {
		t.Fatal("owned conversion buffer copied")
	}
	shape[0] = 0
	if err := src.Close(); err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	if got := x.Data(); len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatal("source close changed tensor", got)
	}
	if got := x.Shape(); len(got) != 1 || got[0] != 2 {
		t.Fatal("shape aliases caller/source", got)
	}
	if src.converted.Value() != &x.Data()[0] {
		t.Fatal("tensor lost converted backing")
	}
	runtime.KeepAlive(x)
}

type textTestSource struct {
	testHeadSource
	data  []float32
	shape []int
	dtype string
	err   error
}

func (s *textTestSource) GetRaw(string) ([]byte, string, []int, error) {
	return nil, s.dtype, s.shape, s.err
}
func (s *textTestSource) GetFloat32(string) ([]float32, []int, error) {
	return append([]float32(nil), s.data...), s.shape, s.err
}

func TestMoJevTextTensorSource(t *testing.T) {
	src := &textTestSource{data: []float32{1, 2, 3, 4}, shape: []int{2, 2}, dtype: "BF16"}
	wrapped := mojevTextSource{src}
	got, err := wrapped.Get("model.layers.0.weight", []int{2, 2})
	if err != nil {
		t.Fatal(err)
	}
	src.data[0] = 99
	if !reflect.DeepEqual(got.Data(), []float32{1, 2, 3, 4}) {
		t.Fatal("borrowed tensor")
	}
	for _, tc := range []struct {
		name, dtype string
		shape       []int
		data        []float32
		err         error
	}{
		{"bad.prefix", "BF16", []int{2, 2}, src.data, nil},
		{"model.layers.0.weight", "I8", []int{2, 2}, src.data, nil},
		{"model.layers.0.weight", "BF16", []int{4}, src.data, nil},
		{"model.layers.0.weight", "BF16", []int{2, 2}, []float32{1}, nil},
		{"model.layers.0.weight", "BF16", []int{2, 2}, []float32{1, 2, 3, float32(math.Inf(1))}, nil},
		{"model.layers.0.weight", "BF16", []int{2, 2}, src.data, errors.New("read error")},
	} {
		src.dtype, src.shape, src.data, src.err = tc.dtype, tc.shape, tc.data, tc.err
		if got, err := wrapped.Get(tc.name, []int{2, 2}); err == nil || got != nil {
			t.Errorf("invalid source accepted: %+v", tc)
		}
	}
}
