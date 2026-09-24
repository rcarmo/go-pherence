package mojev

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

type headTensor struct {
	data []float32
	dims []int
}
type testHeadSource struct {
	tensors map[string]headTensor
	bad     string
	reads   []string
}

func (s *testHeadSource) GetFloat32(name string) ([]float32, []int, error) {
	s.reads = append(s.reads, name)
	if name == s.bad {
		return nil, nil, errors.New("injected failure")
	}
	v, ok := s.tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing %s", name)
	}
	return v.data, v.dims, nil
}
func (s *testHeadSource) GetInt32(string) ([]int32, []int, error) { panic("unexpected int32 access") }
func (s *testHeadSource) GetRaw(string) ([]byte, string, []int, error) {
	panic("unexpected raw access")
}
func (s *testHeadSource) Close() error { return nil }

func smallHeadSource() *testHeadSource {
	return &testHeadSource{tensors: map[string]headTensor{
		"norm.weight":         {[]float32{1, 1}, []int{2}},
		"norm.bias":           {[]float32{0, 0}, []int{2}},
		"context_proj.weight": {[]float32{1, 0}, []int{1, 2}},
		"option_proj.weight":  {[]float32{1, 0}, []int{1, 2}},
	}}
}

func TestLoadHeadOwnsSourceValues(t *testing.T) {
	src := smallHeadSource()
	head, err := LoadHead(src, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(src.reads, []string{"norm.weight", "norm.bias", "context_proj.weight", "option_proj.weight"}) {
		t.Fatalf("read order %v", src.reads)
	}
	if err := src.Close(); err != nil {
		t.Fatal(err)
	}
	for _, name := range src.reads {
		data := src.tensors[name]
		for i := range data.data {
			data.data[i] = 99
		}
	}
	if head.NormWeight[0] != 1 || head.NormBias[0] != 0 || head.ContextProjection[0] != 1 || head.OptionProjection[0] != 1 {
		t.Fatal("head aliases source data")
	}
	state := []bool{true, false, false}
	q := [][]bool{{false, true, false}}
	c := [][][]bool{{{false, false, true}}}
	got, err := head.ScoreHidden([]float32{2, 0, 2, 0, 2, 0}, state, q, c, [][]bool{{true}})
	if err != nil || len(got) != 1 || got[0][0] < 1.9 || got[0][0] > 2.1 {
		t.Fatalf("head after source mutation: %v %v", got, err)
	}
}

func TestLoadHeadRejectsMalformed(t *testing.T) {
	if v, err := LoadHead(nil, 2, 1); v != nil || err == nil {
		t.Fatal("accepted nil source")
	}
	for _, dims := range [][2]int{{0, 1}, {2, 0}, {4097, 1}, {2, 4097}} {
		src := smallHeadSource()
		v, err := LoadHead(src, dims[0], dims[1])
		if v != nil || err == nil || len(src.reads) != 0 {
			t.Fatal("invalid dimensions read source", dims)
		}
	}
	for _, name := range []string{"norm.weight", "norm.bias", "context_proj.weight", "option_proj.weight"} {
		t.Run("missing/"+name, func(t *testing.T) {
			s := smallHeadSource()
			delete(s.tensors, name)
			v, err := LoadHead(s, 2, 1)
			if v != nil || err == nil || !strings.Contains(err.Error(), name) {
				t.Fatalf("accepted missing %s: %v %v", name, v, err)
			}
		})
		t.Run("error/"+name, func(t *testing.T) {
			s := smallHeadSource()
			s.bad = name
			v, err := LoadHead(s, 2, 1)
			if v != nil || err == nil || !strings.Contains(err.Error(), name) {
				t.Fatalf("accepted failed %s: %v %v", name, v, err)
			}
		})
		t.Run("shape/"+name, func(t *testing.T) {
			s := smallHeadSource()
			entry := s.tensors[name]
			entry.dims = []int{2, 1}
			s.tensors[name] = entry
			v, err := LoadHead(s, 2, 1)
			if v != nil || err == nil || !strings.Contains(err.Error(), name) {
				t.Fatalf("accepted wrong shape %s: %v %v", name, v, err)
			}
		})
		t.Run("value/"+name, func(t *testing.T) {
			s := smallHeadSource()
			entry := s.tensors[name]
			entry.data = entry.data[:1]
			s.tensors[name] = entry
			v, err := LoadHead(s, 2, 1)
			if v != nil || err == nil {
				t.Fatalf("accepted short data %s: %v %v", name, v, err)
			}
		})
	}
}
