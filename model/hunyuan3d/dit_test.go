package hunyuan3d

import "testing"

func TestDiTConfigShapes(t *testing.T) {
	cfg := DiTConfig{InChannels: 64, ContextInDim: 1536, HiddenSize: 1024, NumHeads: 16, HeadDim: 64, Depth: 16, DepthSingleBlocks: 32}
	latent, err := cfg.LatentTokenShape(2, 512)
	if err != nil {
		t.Fatal(err)
	}
	context, err := cfg.ContextTokenShape(2, 1370)
	if err != nil {
		t.Fatal(err)
	}
	hidden, err := cfg.HiddenTokenShape(2, 512)
	if err != nil {
		t.Fatal(err)
	}
	qkv, err := cfg.QKVShape(2, 512)
	if err != nil {
		t.Fatal(err)
	}
	checks := []struct {
		name string
		got  []int
		want []int
	}{
		{"latent", latent, []int{2, 512, 64}},
		{"context", context, []int{2, 1370, 1536}},
		{"hidden", hidden, []int{2, 512, 1024}},
		{"qkv", qkv, []int{2, 512, 3, 16, 64}},
	}
	for _, check := range checks {
		if !sameInts(check.got, check.want) {
			t.Fatalf("%s shape=%v want %v", check.name, check.got, check.want)
		}
	}
}

func TestDiTFromShapeConfig(t *testing.T) {
	cfg, err := DiTFromShapeConfig(ShapeConfig{InChannels: 64, ContextInDim: 1536, HiddenSize: 1024, NumHeads: 16, HeadDim: 64, Depth: 16, DepthSingleBlocks: 32, GuidanceEmbed: true})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ContextInDim != 1536 || cfg.GuidanceEmbed != true {
		t.Fatalf("cfg=%+v", cfg)
	}
}

func TestDiTConfigValidation(t *testing.T) {
	if err := (DiTConfig{InChannels: 64, ContextInDim: 1536, HiddenSize: 1024, NumHeads: 15, HeadDim: 64}).Validate(); err == nil {
		t.Fatal("bad head dims accepted")
	}
	cfg := DiTConfig{InChannels: 64, ContextInDim: 1536, HiddenSize: 1024, NumHeads: 16, HeadDim: 64}
	if _, err := cfg.LatentTokenShape(0, 512); err == nil {
		t.Fatal("bad latent batch accepted")
	}
	if _, err := cfg.ContextTokenShape(1, 0); err == nil {
		t.Fatal("bad context token count accepted")
	}
}

func TestCFGExpandedBatch(t *testing.T) {
	if got, err := CFGExpandedBatch(3, false); err != nil || got != 3 {
		t.Fatalf("CFGExpandedBatch disabled=%d err=%v", got, err)
	}
	if got, err := CFGExpandedBatch(3, true); err != nil || got != 6 {
		t.Fatalf("CFGExpandedBatch enabled=%d err=%v", got, err)
	}
	if _, err := CFGExpandedBatch(0, true); err == nil {
		t.Fatal("bad batch accepted")
	}
}
