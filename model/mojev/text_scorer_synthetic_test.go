package mojev

import (
	"bytes"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"
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
