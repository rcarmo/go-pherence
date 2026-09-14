package omnivoice

import (
	"os"
	"testing"
)

func TestRealCheckpointMetadata(t *testing.T) {
	modelDir := os.Getenv("GO_PHERENCE_REAL_OMNIVOICE")
	if modelDir == "" {
		t.Skip("set GO_PHERENCE_REAL_OMNIVOICE to a local model directory")
	}
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	meta, err := ValidateCheckpoint(modelDir, cfg)
	if err != nil {
		t.Fatalf("ValidateCheckpoint: %v", err)
	}
	if !meta.Valid {
		t.Fatalf("ValidateCheckpoint valid=false: %+v", meta)
	}
	if meta.TensorCount != 313 || meta.ExpectedTensorCount != 313 {
		t.Fatalf("counts got tensor=%d expected=%d want 313", meta.TensorCount, meta.ExpectedTensorCount)
	}
	if meta.DTypes["F16"] != 312 || meta.DTypes["I64"] != 1 {
		t.Fatalf("DTypes=%v", meta.DTypes)
	}
	if meta.DataBytes <= 0 {
		t.Fatalf("DataBytes=%d want > 0", meta.DataBytes)
	}
}
