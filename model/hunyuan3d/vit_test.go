package hunyuan3d

import "testing"

func TestAddClassAndPosition(t *testing.T) {
	patches := []float32{1, 2, 3, 4}
	cls := []float32{10, 20}
	pos := []float32{1, 1, 2, 2, 3, 3}
	dst := make([]float32, 6)
	if err := AddClassAndPosition(dst, patches, cls, pos, 2, 2); err != nil {
		t.Fatal(err)
	}
	assertCloseSlice(t, dst, []float32{11, 21, 3, 4, 6, 7}, 1e-6)
}

func TestViTBlockFloat32(t *testing.T) {
	cfg := ViTBlockConfig{Tokens: 2, Dim: 2, Heads: 1, MLPDim: 3, Eps: 1e-6}
	w := ViTBlockWeights{
		Norm1Weight: []float32{1, 1},
		QKVWeight: []float32{
			1, 0, 0, 1, // q identity
			1, 0, 0, 1, // k identity
			1, 0, 0, 1, // v identity
		},
		ProjWeight:  []float32{1, 0, 0, 1},
		Norm2Weight: []float32{1, 1},
		FC1Weight: []float32{
			1, 0,
			0, 1,
			1, 1,
		},
		FC2Weight: []float32{
			1, 0, 0,
			0, 1, 0,
		},
	}
	x := []float32{1, 0, 0, 1}
	dst := make([]float32, len(x))
	scratch := make([]float32, cfg.Tokens*cfg.Dim+cfg.Tokens*3*cfg.Dim+cfg.Tokens*cfg.Dim+cfg.Tokens*cfg.MLPDim+cfg.Tokens*cfg.Dim)
	if err := ViTBlockFloat32(dst, x, cfg, w, scratch); err != nil {
		t.Fatal(err)
	}
	if len(dst) != 4 || dst[0] <= x[0] || dst[3] <= x[3] {
		t.Fatalf("unexpected block output=%v", dst)
	}
}

func TestViTBlockFloat32RejectsBadConfig(t *testing.T) {
	if err := ViTBlockFloat32(make([]float32, 1), make([]float32, 1), ViTBlockConfig{Tokens: 1, Dim: 3, Heads: 2, MLPDim: 4}, ViTBlockWeights{}, make([]float32, 1)); err == nil {
		t.Fatal("bad head split accepted")
	}
}
