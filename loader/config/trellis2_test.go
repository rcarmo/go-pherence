package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTrellis2PipelineSummary(t *testing.T) {
	cfg := mustTrellis2JSON(t, "pipeline.json", `{
		"name":"Trellis2ImageTo3DPipeline",
		"args":{
			"default_pipeline_type":"1024_cascade",
			"models":{
				"sparse_structure_decoder":"microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
				"sparse_structure_flow_model":"microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16",
				"shape_slat_flow_model_512":"microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
			}
		}
	}`)
	got, family, err := ReadTrellis2Config(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if family != Trellis2FamilyPipeline {
		t.Fatalf("family=%s", family)
	}
	summary := SummarizeTrellis2Config(got, family)
	if summary.Name != "Trellis2ImageTo3DPipeline" || summary.DefaultPipelineType != "1024_cascade" {
		t.Fatalf("summary=%+v", summary)
	}
	if len(summary.ModelKeys) != 3 || summary.ModelKeys[0] != "shape_slat_flow_model_512" {
		t.Fatalf("model keys=%v", summary.ModelKeys)
	}
}

func TestTrellis2CheckpointSummaries(t *testing.T) {
	cases := []struct {
		path   string
		json   string
		family Trellis2Family
		check  func(*testing.T, Trellis2Summary)
	}{
		{
			path:   "ckpts/shape_enc_next_dc_f16c32_fp16.json",
			json:   `{"name":"FlexiDualGridVaeEncoder","args":{"model_channels":[64,128,256,512,1024],"latent_channels":32,"num_blocks":[0,4,8,16,4]}}`,
			family: Trellis2FamilyShapeEncoder,
			check: func(t *testing.T, s Trellis2Summary) {
				if s.LatentChannels != 32 || len(s.ModelChannels) != 5 || s.ModelChannels[4] != 1024 || len(s.NumBlocks) != 5 {
					t.Fatalf("shape encoder summary=%+v", s)
				}
			},
		},
		{
			path:   "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
			json:   `{"name":"SparseStructureFlowModel","args":{"resolution":64,"in_channels":8,"out_channels":8,"hidden_size":1536,"num_heads":24,"depth":24,"dtype":"bfloat16"}}`,
			family: Trellis2FamilySparseStructureFlow,
			check: func(t *testing.T, s Trellis2Summary) {
				if s.Resolution != 64 || s.InChannels != 8 || s.HiddenSize != 1536 || s.NumHeads != 24 || s.Depth != 24 || s.DType != "bfloat16" {
					t.Fatalf("sparse flow summary=%+v", s)
				}
			},
		},
		{
			path:   "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
			json:   `{"name":"SLatFlowModel","args":{"resolution":32,"in_channels":32,"out_channels":32,"hidden_size":1536,"num_heads":24,"depth":24,"dtype":"bfloat16"}}`,
			family: Trellis2FamilyStructuredLatentShapeFlow,
			check: func(t *testing.T, s Trellis2Summary) {
				if s.Resolution != 32 || s.InChannels != 32 || s.OutChannels != 32 || s.HiddenSize != 1536 {
					t.Fatalf("slat flow summary=%+v", s)
				}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.path, func(t *testing.T) {
			path := mustTrellis2JSON(t, tc.path, tc.json)
			cfg, family, err := ReadTrellis2Config(path)
			if err != nil {
				t.Fatal(err)
			}
			if family != tc.family {
				t.Fatalf("family=%s want %s", family, tc.family)
			}
			tc.check(t, SummarizeTrellis2Config(cfg, family))
		})
	}
}

func TestTrellis2ValidationRejectsBadPipeline(t *testing.T) {
	path := mustTrellis2JSON(t, "pipeline.json", `{"name":"Trellis2ImageTo3DPipeline","args":{}}`)
	if _, _, err := ReadTrellis2Config(path); err == nil {
		t.Fatal("bad pipeline accepted")
	}
}

func TestValidateTrellis2PipelineModelKeys(t *testing.T) {
	path := mustTrellis2JSON(t, "pipeline.json", `{
		"name":"Trellis2ImageTo3DPipeline",
		"args":{"models":{
			"sparse_structure_decoder":"ss-dec",
			"sparse_structure_flow_model":"ss-flow",
			"shape_slat_decoder":"shape-dec",
			"shape_slat_flow_model_512":"shape-flow-512",
			"shape_slat_flow_model_1024":"shape-flow-1024",
			"tex_slat_decoder":"tex-dec",
			"tex_slat_flow_model_512":"tex-flow-512",
			"tex_slat_flow_model_1024":"tex-flow-1024"
		}}
	}`)
	cfg, _, err := ReadTrellis2Config(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateTrellis2ShapePipeline(cfg, false); err != nil {
		t.Fatalf("shape-only pipeline rejected: %v", err)
	}
	if err := ValidateTrellis2ShapePipeline(cfg, true); err != nil {
		t.Fatalf("shape+texture pipeline rejected: %v", err)
	}

	missingTexture := mustTrellis2JSON(t, "pipeline.json", `{
		"name":"Trellis2ImageTo3DPipeline",
		"args":{"models":{
			"sparse_structure_decoder":"ss-dec",
			"sparse_structure_flow_model":"ss-flow",
			"shape_slat_decoder":"shape-dec",
			"shape_slat_flow_model_512":"shape-flow-512",
			"shape_slat_flow_model_1024":"shape-flow-1024"
		}}
	}`)
	cfg, _, err = ReadTrellis2Config(missingTexture)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateTrellis2ShapePipeline(cfg, false); err != nil {
		t.Fatalf("shape-only pipeline with no texture rejected: %v", err)
	}
	if err := ValidateTrellis2ShapePipeline(cfg, true); err == nil {
		t.Fatal("shape+texture pipeline with missing texture keys accepted")
	}
}

func TestValidateTrellis2TexturingPipelineModelKeys(t *testing.T) {
	path := mustTrellis2JSON(t, "texturing_pipeline.json", `{
		"name":"Trellis2TexturingPipeline",
		"args":{"models":{
			"shape_slat_encoder":"shape-enc",
			"tex_slat_decoder":"tex-dec",
			"tex_slat_flow_model_512":"tex-flow-512",
			"tex_slat_flow_model_1024":"tex-flow-1024"
		}}
	}`)
	cfg, _, err := ReadTrellis2Config(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateTrellis2TexturingPipeline(cfg); err != nil {
		t.Fatalf("texturing pipeline rejected: %v", err)
	}

	bad := mustTrellis2JSON(t, "texturing_pipeline.json", `{"name":"Trellis2TexturingPipeline","args":{"models":{"tex_slat_decoder":"tex-dec"}}}`)
	cfg, _, err = ReadTrellis2Config(bad)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateTrellis2TexturingPipeline(cfg); err == nil {
		t.Fatal("bad texturing pipeline accepted")
	}
}

func TestTrellis2FamilyForPathAndName(t *testing.T) {
	if got := Trellis2FamilyForPath("ckpts/tex_dec_next_dc_f16c32_fp16.json"); got != Trellis2FamilyTextureDecoder {
		t.Fatalf("family for path=%s", got)
	}
	if got := Trellis2FamilyForName("SparseStructureFlowModel"); got != Trellis2FamilySparseStructureFlow {
		t.Fatalf("family for name=%s", got)
	}
}

func mustTrellis2JSON(t *testing.T, rel, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), rel)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
