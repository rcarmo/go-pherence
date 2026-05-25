package config

import "testing"

const hunyuan3DSampleConfig = `
model:
  target: hy3dgen.shapegen.models.Hunyuan3DDiT
  params:
    in_channels: 64
    context_in_dim: 1536
    hidden_size: 1024
    mlp_ratio: 4.0
    num_heads: 16
    depth: 16
    depth_single_blocks: 32
    axes_dim: [64]
    theta: 10000
    qkv_bias: true
vae:
  target: hy3dgen.shapegen.models.ShapeVAE
  params:
    num_latents: 3072
    embed_dim: 64
    width: 1024
    heads: 16
    num_decoder_layers: 16
    num_encoder_layers: 8
    qk_norm: true
    scale_factor: 0.9990943042622529
conditioner:
  target: hy3dgen.shapegen.models.SingleImageEncoder
  params:
    main_image_encoder:
      type: DinoImageEncoder
      kwargs:
        image_size: 512
scheduler:
  target: hy3dgen.shapegen.schedulers.FlowMatchEulerDiscreteScheduler
  params:
    num_train_timesteps: 1000
image_processor:
  target: hy3dgen.shapegen.preprocessors.ImageProcessorV2
  params:
    size: 512
`

func TestParseHunyuan3DConfig(t *testing.T) {
	cfg, err := ParseHunyuan3DConfig([]byte(hunyuan3DSampleConfig))
	if err != nil {
		t.Fatalf("ParseHunyuan3DConfig: %v", err)
	}
	s := cfg.Summary()
	if s.DenoiserTarget != "hy3dgen.shapegen.models.Hunyuan3DDiT" || s.InChannels != 64 || s.ContextInDim != 1536 || s.HiddenSize != 1024 || s.NumHeads != 16 || s.Depth != 16 || s.DepthSingleBlocks != 32 {
		t.Fatalf("denoiser summary=%+v", s)
	}
	if s.VAETarget != "hy3dgen.shapegen.models.ShapeVAE" || s.VAELatents != 3072 || s.VAEEmbedDim != 64 || s.VAEWidth != 1024 || s.VAEHeads != 16 {
		t.Fatalf("vae summary=%+v", s)
	}
	if s.ConditionerType != "DinoImageEncoder" || s.SchedulerSteps != 1000 {
		t.Fatalf("conditioner/scheduler summary=%+v", s)
	}
}

func TestParseHunyuan3DConfigRejectsBadAxes(t *testing.T) {
	bad := []byte(`
model:
  target: m
  params:
    in_channels: 64
    context_in_dim: 1536
    hidden_size: 1024
    num_heads: 16
    depth: 1
    depth_single_blocks: 1
    axes_dim: [32]
vae:
  target: v
  params:
    num_latents: 1
    embed_dim: 64
    width: 1024
    heads: 16
    num_decoder_layers: 1
conditioner:
  target: c
scheduler:
  target: s
  params:
    num_train_timesteps: 1000
`)
	if _, err := ParseHunyuan3DConfig(bad); err == nil {
		t.Fatal("bad axes_dim parsed without error")
	}
}

func TestSummarizeHunyuan3DTensors(t *testing.T) {
	inv := SummarizeHunyuan3DTensors([]string{
		"model.latent_in.weight",
		"model.double_blocks.0.img_attn.qkv.weight",
		"vae.post_kl.weight",
		"conditioner.main_image_encoder.model.embeddings.patch_embeddings.projection.weight",
		"__metadata__",
	})
	if inv.Total != 5 || inv.Model != 2 || inv.VAE != 1 || inv.Conditioner != 1 || inv.Other != 1 {
		t.Fatalf("inventory=%+v", inv)
	}
	if got := ClassifyHunyuan3DTensorName("conditioner.foo"); got != Hunyuan3DTensorConditioner {
		t.Fatalf("conditioner group=%q", got)
	}
}
