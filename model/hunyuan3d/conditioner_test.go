package hunyuan3d

import "testing"

func TestConditionerConfigShapes(t *testing.T) {
	cfg := ConditionerConfig{Type: "DinoImageEncoder", ImageSize: 518, PatchSize: 14, Channels: 3}
	shape, err := cfg.ImageTensorShape(2)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{2, 3, 518, 518}
	for i := range want {
		if shape[i] != want[i] {
			t.Fatalf("shape=%v want %v", shape, want)
		}
	}
	gw, gh, err := cfg.PatchGrid()
	if err != nil {
		t.Fatal(err)
	}
	if gw != 37 || gh != 37 {
		t.Fatalf("grid=%dx%d", gw, gh)
	}
	n, err := cfg.PatchTokenCount(true)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1370 {
		t.Fatalf("tokens=%d want 1370", n)
	}
}

func TestConditionerConfigValidation(t *testing.T) {
	if err := (ConditionerConfig{}).Validate(); err == nil {
		t.Fatal("empty conditioner accepted")
	}
	if err := (ConditionerConfig{Type: "DinoImageEncoder", ImageSize: 512, PatchSize: 14, Channels: 3}).Validate(); err == nil {
		t.Fatal("non-divisible patch grid accepted")
	}
	if _, err := (ConditionerConfig{Type: "DinoImageEncoder", ImageSize: 518, PatchSize: 14, Channels: 3}).ImageTensorShape(0); err == nil {
		t.Fatal("bad batch accepted")
	}
}

func TestConditionerFromShapeConfig(t *testing.T) {
	cfg, err := ConditionerFromShapeConfig(ShapeConfig{ConditionerType: "DinoImageEncoder"})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Type != "DinoImageEncoder" || cfg.ImageSize != 518 || cfg.PatchSize != 14 || cfg.Channels != 3 {
		t.Fatalf("cfg=%+v", cfg)
	}
}
