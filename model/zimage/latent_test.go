package zimage

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

func TestPinnedVAEBoundary(t *testing.T) {
	cfg, err := loaderconfig.ReadZImageConfig("../../testdata/zimage")
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/vae_boundary_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Schema               int       `json:"schema"`
		ModelRevision        string    `json:"model_revision"`
		PipelineRevision     string    `json:"pipeline_revision"`
		PipelineSourceSHA256 string    `json:"pipeline_source_sha256"`
		VAEConfigSHA256      string    `json:"vae_config_sha256"`
		ImageHeight          int       `json:"image_height"`
		ImageWidth           int       `json:"image_width"`
		Batch                int       `json:"batch"`
		LatentShapeNCHW      []int     `json:"latent_shape_nchw"`
		DecodeInput          []float32 `json:"decode_input"`
		DecodeOutput         []float32 `json:"decode_output"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.Schema != 1 || ref.ModelRevision != "f332072aa78be7aecdf3ee76d5c247082da564a6" || ref.PipelineRevision != "0377f0c1b34e3ff313d41edad1bd79c2ed8bb5ec" || ref.PipelineSourceSHA256 != "2e31dfe83498a963895952053dbe939840f67a30c747b11ed5ab6f71751d6a92" || ref.VAEConfigSHA256 != "e80af1e64a71883a9d10c3159d2e493e5934508da57852f6a180ae6ae63b14bd" {
		t.Fatal("unexpected VAE boundary provenance")
	}
	shape, err := DefaultLatentShape(cfg.VAE, cfg.Transformer, ref.Batch, ref.ImageHeight, ref.ImageWidth)
	if err != nil {
		t.Fatal(err)
	}
	if got := []int{shape.Batch, shape.Channels, shape.Height, shape.Width}; !reflect.DeepEqual(got, ref.LatentShapeNCHW) || shape.Numel() != 384 {
		t.Fatalf("latent shape=%v numel=%d want=%v", got, shape.Numel(), ref.LatentShapeNCHW)
	}
	// Small independent F32 arithmetic vector; the runtime requires the
	// complete NCHW tensor and does not silently broadcast it.
	input := make([]float32, shape.Numel())
	copy(input, ref.DecodeInput)
	output := make([]float32, shape.Numel())
	if err := PrepareVAEDecodeInputInto(output, input, shape, cfg.VAE); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(output[:len(ref.DecodeOutput)], ref.DecodeOutput) {
		t.Fatalf("VAE boundary=%v want=%v", output[:len(ref.DecodeOutput)], ref.DecodeOutput)
	}
	inPlace := append([]float32(nil), input...)
	if err := PrepareVAEDecodeInputInto(inPlace, inPlace, shape, cfg.VAE); err != nil || !reflect.DeepEqual(inPlace, output) {
		t.Fatalf("in-place boundary err=%v", err)
	}
	if got := testing.AllocsPerRun(100, func() {
		if err := PrepareVAEDecodeInputInto(output, input, shape, cfg.VAE); err != nil {
			t.Fatal(err)
		}
	}); got != 0 {
		t.Fatalf("VAE boundary allocs=%v want 0", got)
	}
}

func TestVAEBoundaryRejectsBeforeWriting(t *testing.T) {
	cfg, err := loaderconfig.ReadZImageConfig("../../testdata/zimage")
	if err != nil {
		t.Fatal(err)
	}
	for _, dims := range [][2]int{{0, 32}, {15, 32}, {32, 31}} {
		if _, err := DefaultLatentShape(cfg.VAE, cfg.Transformer, 1, dims[0], dims[1]); err == nil {
			t.Fatalf("accepted invalid image dimensions %v", dims)
		}
	}
	if _, err := DefaultLatentShape(cfg.VAE, cfg.Transformer, 0, 32, 32); err == nil {
		t.Fatal("accepted zero batch")
	}
	if _, err := DefaultLatentShape(cfg.VAE, cfg.Transformer, int(^uint(0)>>1), 32, 32); err == nil {
		t.Fatal("accepted overflowing batch")
	}
	badTransformer := cfg.Transformer
	badTransformer.InChannels++
	if _, err := DefaultLatentShape(cfg.VAE, badTransformer, 1, 32, 32); err == nil {
		t.Fatal("accepted mismatched DiT channels")
	}
	if (LatentShape{Batch: 1, Channels: 16, Height: int(^uint(0) >> 1), Width: 2}).Numel() != 0 {
		t.Fatal("accepted overflowing tensor")
	}
	shape, _ := DefaultLatentShape(cfg.VAE, cfg.Transformer, 1, 16, 16)
	in := make([]float32, shape.Numel())
	dst := make([]float32, len(in))
	for i := range dst {
		dst[i] = 9
	}
	cases := []struct {
		name   string
		change func() ([]float32, []float32, LatentShape, loaderconfig.ZImageVAEConfig)
	}{
		{"short input", func() ([]float32, []float32, LatentShape, loaderconfig.ZImageVAEConfig) {
			return dst, in[:len(in)-1], shape, cfg.VAE
		}},
		{"wrong channels", func() ([]float32, []float32, LatentShape, loaderconfig.ZImageVAEConfig) {
			s := shape
			s.Channels = 8
			return dst, in, s, cfg.VAE
		}},
		{"bad scale", func() ([]float32, []float32, LatentShape, loaderconfig.ZImageVAEConfig) {
			c := cfg.VAE
			c.ScalingFactor = 0
			return dst, in, shape, c
		}},
		{"bad shift", func() ([]float32, []float32, LatentShape, loaderconfig.ZImageVAEConfig) {
			c := cfg.VAE
			c.ShiftFactor = math.NaN()
			return dst, in, shape, c
		}},
		{"wrong blocks", func() ([]float32, []float32, LatentShape, loaderconfig.ZImageVAEConfig) {
			c := cfg.VAE
			c.BlockOutChannels = []int{128, 256, 512}
			return dst, in, shape, c
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			to, from, s, v := tc.change()
			if err := PrepareVAEDecodeInputInto(to, from, s, v); err == nil || dst[0] != 9 {
				t.Fatalf("accepted malformed decode boundary or wrote result: %v", err)
			}
		})
	}
	in[len(in)-1] = float32(math.Inf(1))
	if err := PrepareVAEDecodeInputInto(dst, in, shape, cfg.VAE); err == nil || dst[0] != 9 {
		t.Fatalf("accepted non-finite input or wrote result: %v", err)
	}
	in[len(in)-1] = -math.MaxFloat32
	if err := PrepareVAEDecodeInputInto(dst, in, shape, cfg.VAE); err == nil || dst[0] != 9 {
		t.Fatalf("accepted overflowing result or wrote result: %v", err)
	}
	storage := make([]float32, len(in)+1)
	if err := PrepareVAEDecodeInputInto(storage[1:], storage[:len(in)], shape, cfg.VAE); err == nil {
		t.Fatal("accepted partial overlap")
	}
}
