package pockettts

import (
	"math"
	"reflect"
	"testing"
)

func TestTrainingReadOnlyStateValidationOwnership(t *testing.T) {
	expected := map[string][]float32{"x": make([]float32, 3), "y": make([]float32, 1)}
	list := []NamedTrainingTensor{{Name: "y", Values: []float32{4}}, {Name: "x", Values: []float32{1, 2, 3}}}
	adam := []NamedAdamState{{Name: "y", M: []float32{1}, V: []float32{2}}, {Name: "x", M: []float32{1, 2, 3}, V: []float32{4, 5, 6}}}
	borrowed, err := validateTensorList(list, expected, "test")
	if err != nil {
		t.Fatal(err)
	}
	owned, err := tensorListToMap(list, expected, "test")
	if err != nil {
		t.Fatal(err)
	}
	moments, err := adamListToMap(adam, expected, true)
	if err != nil {
		t.Fatal(err)
	}
	refs, err := validateAdamList(adam, expected, true)
	if err != nil {
		t.Fatal(err)
	}
	if !f32SlicesOverlap(borrowed["x"], list[1].Values) || !f32SlicesOverlap(refs["x"].M, adam[1].M) {
		t.Fatal("not borrowed views")
	}
	if f32SlicesOverlap(owned["x"], list[1].Values) || f32SlicesOverlap(moments["x"].M, adam[1].M) || f32SlicesOverlap(moments["x"].V, adam[1].V) {
		t.Fatal("owned conversion aliases")
	}
	list[1].Values[0] = 99
	adam[1].M[0] = 88
	adam[1].V[0] = 77
	if owned["x"][0] != 1 || moments["x"].M[0] != 1 || moments["x"].V[0] != 4 {
		t.Fatal("owned conversion changed")
	}
	owned["x"][1] = 9
	moments["x"].M[1] = 8
	if list[1].Values[1] != 2 || adam[1].M[1] != 2 {
		t.Fatal("input mutated")
	}
	if _, err = validateAdamList(nil, expected, false); err != nil {
		t.Fatal("optional zero-step moments", err)
	}
}
func TestTrainingReadOnlyStateValidationAdmission(t *testing.T) {
	expected := map[string][]float32{"x": make([]float32, 2)}
	for _, list := range [][]NamedTrainingTensor{nil, {{Name: "missing", Values: []float32{1, 2}}}, {{Name: "x", Values: []float32{1}}}, {{Name: "x", Values: []float32{1, 2}}, {Name: "x", Values: []float32{1, 2}}}, {{Name: "x", Values: []float32{float32(math.NaN()), 2}}}, {{Name: "x", Values: []float32{1, float32(math.Inf(1))}}}} {
		if _, err := validateTensorList(list, expected, "test"); err == nil {
			t.Fatal("borrowed malformed tensor")
		}
		if _, err := tensorListToMap(list, expected, "test"); err == nil {
			t.Fatal("owned malformed tensor")
		}
	}
	for _, list := range [][]NamedAdamState{nil, {{Name: "missing", M: []float32{1, 2}, V: []float32{1, 2}}}, {{Name: "x", M: []float32{1}, V: []float32{1, 2}}}, {{Name: "x", M: []float32{1, 2}, V: []float32{1}}}, {{Name: "x", M: []float32{float32(math.NaN()), 2}, V: []float32{1, 2}}}, {{Name: "x", M: []float32{1, 2}, V: []float32{float32(math.Inf(-1)), 2}}}} {
		if _, err := validateAdamList(list, expected, true); err == nil {
			t.Fatal("borrowed malformed moments")
		}
		if _, err := adamListToMap(list, expected, true); err == nil {
			t.Fatal("owned malformed moments")
		}
	}
	// Exactly matched inventory count with duplicate names must still reject.
	two := map[string][]float32{"x": make([]float32, 2), "y": make([]float32, 2)}
	if _, err := validateAdamList([]NamedAdamState{{Name: "x", M: []float32{1, 2}, V: []float32{1, 2}}, {Name: "x", M: []float32{1, 2}, V: []float32{1, 2}}}, two, true); err == nil {
		t.Fatal("duplicate Adam")
	}
}
func validationStateFixture(n int) FullTrainingState {
	c := DefaultAdamWConfig()
	return FullTrainingState{Version: 1, Step: 1, AdamW: c, EMADecay: .9, Params: []NamedTrainingTensor{{Name: "x", Values: make([]float32, n)}}, Buffers: []NamedTrainingTensor{{Name: "buf", Values: []float32{1}}}, Adam: []NamedAdamState{{Name: "x", M: make([]float32, n), V: make([]float32, n)}}, EMA: []NamedTrainingTensor{{Name: "x", Values: make([]float32, n)}}}
}
func TestTrainingReadOnlyStateValidationStorage(t *testing.T) {
	s := validationStateFixture(32768)
	// Inputs and all state ownership remain unchanged by repeated validation.
	before := validationStateFixture(32768)
	for i := 0; i < 3; i++ {
		if err := s.Validate(); err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(s, before) {
		t.Fatal("validation mutated owned state")
	}
	var borrowed, owned float64
	borrowed = testing.AllocsPerRun(10, func() {
		if err := s.Validate(); err != nil {
			panic(err)
		}
	})
	owned = testing.AllocsPerRun(10, func() {
		if _, err := tensorListToMap(s.EMA, map[string][]float32{"x": s.Params[0].Values}, "EMA"); err != nil {
			panic(err)
		}
		if _, err := adamListToMap(s.Adam, map[string][]float32{"x": s.Params[0].Values}, true); err != nil {
			panic(err)
		}
	})
	t.Logf("Validate borrowed allocations/op=%g; standalone owned moment+EMA conversion allocations/op=%g (byte savings measured in benchmark)", borrowed, owned)
}
func BenchmarkTrainingReadOnlyStateValidation(b *testing.B) {
	s := validationStateFixture(32768)
	b.Run("read-only", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if err := s.Validate(); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("prior-discarded-copies", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			specs := map[string][]float32{}
			for _, p := range s.Params {
				specs[p.Name] = make([]float32, len(p.Values))
			}
			if _, err := adamListToMap(s.Adam, specs, true); err != nil {
				b.Fatal(err)
			}
			if _, err := tensorListToMap(s.EMA, specs, "EMA"); err != nil {
				b.Fatal(err)
			}
		}
	})
}
