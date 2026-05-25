package hunyuan3d

import "testing"

func TestValidateTensorCoverage(t *testing.T) {
	coverage, err := ValidateTensorCoverage([]string{
		"conditioner.main_image_encoder.model.embeddings.patch_embeddings.projection.weight",
		"model.double_blocks.0.img_attn_qkv.weight",
		"model.single_blocks.0.linear1.weight",
		"vae.latents",
		"vae.geo_decoder.query_proj.weight",
		"metadata.version",
	})
	if err != nil {
		t.Fatal(err)
	}
	if coverage.Total != 6 || coverage.Model != 2 || coverage.VAE != 2 || coverage.Conditioner != 1 || coverage.Other != 1 {
		t.Fatalf("coverage=%+v", coverage)
	}
	if len(coverage.Examples) == 0 {
		t.Fatalf("missing examples: %+v", coverage)
	}
}

func TestValidateTensorCoverageRejectsMissingGroups(t *testing.T) {
	coverage, err := ValidateTensorCoverage([]string{"model.x", "vae.y"})
	if err == nil {
		t.Fatal("missing conditioner accepted")
	}
	if len(coverage.Missing) != 1 || coverage.Missing[0] != "conditioner.*" {
		t.Fatalf("missing=%v err=%v", coverage.Missing, err)
	}
}

func TestTensorNames(t *testing.T) {
	got := TensorNames(map[string]int{"b": 2, "a": 1})
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("TensorNames=%v", got)
	}
}
