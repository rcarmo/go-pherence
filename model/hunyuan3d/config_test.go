package hunyuan3d

import (
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

const sampleConfig = `
model:
  target: hy3dgen.shapegen.models.Hunyuan3DDiT
  params:
    in_channels: 64
    context_in_dim: 1536
    hidden_size: 1024
    mlp_ratio: 4.0
    num_heads: 16
    depth: 8
    depth_single_blocks: 16
    axes_dim: [64]
    theta: 10000
    qkv_bias: true
    guidance_embed: false
vae:
  target: hy3dgen.shapegen.models.ShapeVAE
  params:
    num_latents: 512
    embed_dim: 64
    width: 1024
    heads: 16
    num_decoder_layers: 16
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

func TestFromLoaderConfig(t *testing.T) {
	cfg, err := loaderconfig.ParseHunyuan3DConfig([]byte(sampleConfig))
	if err != nil {
		t.Fatal(err)
	}
	shape, err := FromLoaderConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if shape.InChannels != 64 || shape.HiddenSize != 1024 || shape.NumHeads != 16 || shape.HeadDim != 64 || shape.Depth != 8 || shape.DepthSingleBlocks != 16 {
		t.Fatalf("denoiser shape=%+v", shape)
	}
	if shape.VAELatents != 512 || shape.VAEWidth != 1024 || shape.VAEHeads != 16 || shape.VAEHeadDim != 64 {
		t.Fatalf("vae shape=%+v", shape)
	}
	if shape.ConditionerType != "DinoImageEncoder" || shape.SchedulerSteps != 1000 || shape.GuidanceEmbed {
		t.Fatalf("metadata=%+v", shape)
	}
	latent, err := shape.LatentShape(2)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{2, 512, 64}
	for i := range want {
		if latent[i] != want[i] {
			t.Fatalf("latent shape=%v want %v", latent, want)
		}
	}
}

func TestShapeConfigValidation(t *testing.T) {
	bad := ShapeConfig{DenoiserTarget: "m", VAETarget: "v", ConditionerTarget: "c", SchedulerTarget: "s", InChannels: 64, ContextInDim: 1536, HiddenSize: 1024, NumHeads: 15, HeadDim: 68, Depth: 1, DepthSingleBlocks: 1, VAELatents: 512, VAEEmbedDim: 64, VAEWidth: 1024, VAEHeads: 16, VAEHeadDim: 64, ConditionerType: "DinoImageEncoder", SchedulerSteps: 1000}
	if err := bad.Validate(); err == nil {
		t.Fatal("bad denoiser head dims accepted")
	}
	bad = ShapeConfig{DenoiserTarget: "m", VAETarget: "v", ConditionerTarget: "c", SchedulerTarget: "s", InChannels: 64, ContextInDim: 1536, HiddenSize: 1024, NumHeads: 16, HeadDim: 64, Depth: 1, DepthSingleBlocks: 1, VAELatents: 512, VAEEmbedDim: 64, VAEWidth: 1024, VAEHeads: 15, VAEHeadDim: 68, ConditionerType: "DinoImageEncoder", SchedulerSteps: 1000}
	if err := bad.Validate(); err == nil {
		t.Fatal("bad VAE head dims accepted")
	}
}
