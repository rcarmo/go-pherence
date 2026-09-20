package lfm2

import (
	"math"
	"path/filepath"
	"testing"
)

func TestSizingRejectsOverflowAndNegativeFactors(t *testing.T) {
	max := int(^uint(0) >> 1)
	for _, v := range [][]int{{max, 2}, {-1, 0}, {0, -1}, {max, 3, 0}} {
		if sizeProduct(v...) != -1 {
			t.Fatal(v)
		}
	}
	if sizeSum(max, 1) != -1 || sizeSum(-1, 1) != -1 {
		t.Fatal("sum accepted overflow")
	}
	if n, err := sizeBytes(max, 3); err == nil && uint64(max) > math.MaxInt64/3 {
		t.Fatal(n)
	}
	if _, err := sizeBytes(-1, 0); err == nil {
		t.Fatal("negative bytes")
	}
	if n, err := sizeBytes(0, 8); err != nil || n != 0 {
		t.Fatal(n, err)
	}
}
func TestLayoutConstructorsAndForgedCountsFailClosed(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	cfg := meta.Config
	max := int(^uint(0) >> 1)
	tests := map[string]func() error{
		"conv state": func() error {
			c := cfg
			c.ConvLCache = max
			_, err := NewConvStateLayout(c, LayerSchedule{})
			return err
		},
		"conv weights": func() error {
			c := cfg
			c.ConvLCache = max
			_, err := NewConvProjectionLayout(c, LayerSchedule{})
			return err
		},
		"router":    func() error { c := cfg; c.NumExperts = max; _, err := NewRouterLayout(c, ExecutionPlan{}); return err },
		"embedding": func() error { c := cfg; c.VocabSize = max; _, err := NewEmbeddingLayout(c); return err },
		"ffn": func() error {
			c := cfg
			c.IntermediateSize = max
			_, err := NewFFNLayout(c, ExecutionPlan{})
			return err
		},
		"attention": func() error {
			c := cfg
			c.HiddenSize = max - 1
			c.HeadDim = (max - 1) / 2
			c.NumAttentionHeads = 2
			c.NumKeyValueHeads = 2
			_, err := NewAttentionProjectionLayout(c, LayerSchedule{})
			return err
		},
	}
	for name, fn := range tests {
		t.Run(name, func(t *testing.T) {
			if err := fn(); err == nil {
				t.Fatal("overflow accepted")
			}
		})
	}
	// A wrapped declared count must not match the same wrapped expected count.
	l := EmbeddingLayout{VocabSize: max, HiddenSize: 3, TieWordEmbeddings: true, OutputSharesInput: true, EmbeddingFloats: max * 3, TotalUntiedFloats: max * 3}
	if l.Validate() == nil {
		t.Fatal("wrapped embedding accepted")
	}
	p, err := NewRuntimePlan(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for name, fn := range map[string]func() error{
		"conv bytes":      func() error { _, e := p.ConvStateBytes(max); return e },
		"kv bytes":        func() error { _, e := p.KVBytes(max, max); return e },
		"embedding bytes": func() error { _, e := p.EmbeddingLayout.Bytes(max); return e },
		"router scratch":  func() error { _, e := p.RouterLayout.ScratchFloats(max); return e },
		"norm scratch":    func() error { _, e := p.NormLayout.ScratchFloats(max); return e },
	} {
		t.Run(name, func(t *testing.T) {
			if fn() == nil {
				t.Fatal("overflow admitted")
			}
		})
	}
	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		n := p.NormLayout
		n.Epsilon = bad
		if n.Validate() == nil {
			t.Fatal("bad epsilon")
		}
		r := p.RoPELayout
		r.Theta = bad
		if r.Validate() == nil {
			t.Fatal("bad theta")
		}
		route := p.Routing
		route.RoutedScalingFactor = bad
		if route.Validate() == nil {
			t.Fatal("bad routed scale")
		}
	}
}
func TestScheduleIndexListsMatchSteps(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	s, _ := NewLayerSchedule(meta.Config)
	s.ConvIndices[0] = s.FullAttentionIndices[0]
	if s.Validate(meta.Config.NumHiddenLayers) == nil {
		t.Fatal("wrong convolution indices accepted")
	}
	if _, err := NewConvStateLayout(meta.Config, s); err == nil {
		t.Fatal("constructor ignored malformed schedule")
	}
	p, _ := NewExecutionPlan(meta.Config)
	p.MoEIndices[0] = -1
	if p.Validate(meta.Config.NumHiddenLayers) == nil {
		t.Fatal("wrong expert indices accepted")
	}
	if _, err := NewFFNLayout(meta.Config, p); err == nil {
		t.Fatal("constructor ignored malformed execution")
	}
}
