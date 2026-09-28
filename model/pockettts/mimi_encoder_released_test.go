package pockettts

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type mimiEncoderOracle struct {
	Schema          int    `json:"schema"`
	Upstream        string `json:"upstream_revision"`
	WeightsRevision string `json:"weights_revision"`
	WeightsSHA256   string `json:"weights_sha256"`
	Samples         int    `json:"samples"`
	LatentShape     []int  `json:"latent_shape"`
	Audio, Latent   []float32
}

func TestMimiEncoderReleasedParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_POCKETTTS_VOICE_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_VOICE_MODEL to the pinned gated bundle")
	}
	data, err := os.ReadFile("testdata/mimi_encoder_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle mimiEncoderOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.Upstream != UpstreamCommit || oracle.WeightsRevision != "983151f13aaeab1b13c1e5e3c2c383d49a9edf3f" || oracle.WeightsSHA256 != "fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997" {
		t.Fatal("unexpected Mimi encoder oracle")
	}
	cfg := releasedConfig(t)
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	encoder, err := LoadMimiEncoderCPU(file, cfg)
	if err != nil {
		t.Fatal(err)
	}
	got, frames, err := encoder.Encode(oracle.Audio)
	if err != nil {
		t.Fatal(err)
	}
	if frames != oracle.LatentShape[1] || len(got) != len(oracle.Latent) {
		t.Fatalf("latent frames=%d len=%d shape=%v", frames, len(got), oracle.LatentShape)
	}
	assertSliceClose(t, "released Mimi encoder", got, oracle.Latent, 3e-5)
}

func TestMimiEncoderReleasedWarmZeroAlloc(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_POCKETTTS_VOICE_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_VOICE_MODEL")
	}
	data, err := os.ReadFile("testdata/mimi_encoder_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle mimiEncoderOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	cfg := releasedConfig(t)
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	encoder, err := LoadMimiEncoderCPU(file, cfg)
	file.Close()
	if err != nil {
		t.Fatal(err)
	}
	workspace, err := encoder.NewWorkspace(oracle.LatentShape[1])
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(oracle.Latent))
	if err = encoder.EncodeInto(out, oracle.Audio, workspace); err != nil {
		t.Fatal(err)
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if e := encoder.EncodeInto(out, oracle.Audio, workspace); e != nil {
			panic(e)
		}
	}); allocs != 0 {
		t.Fatalf("warm encoder allocations=%g", allocs)
	}
	assertSliceClose(t, "warm released Mimi encoder", out, oracle.Latent, 3e-5)
}

func TestMimiEncoderReleasedRejectsOverlapTransactionally(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_POCKETTTS_VOICE_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_VOICE_MODEL")
	}
	data, err := os.ReadFile("testdata/mimi_encoder_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle mimiEncoderOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	encoder, err := LoadMimiEncoderCPU(file, releasedConfig(t))
	file.Close()
	if err != nil {
		t.Fatal(err)
	}
	frames, latent := oracle.LatentShape[1], len(oracle.Latent)
	workspace, err := encoder.NewWorkspace(frames)
	if err != nil {
		t.Fatal(err)
	}
	baseline := make([]float32, latent)
	if err = encoder.EncodeInto(baseline, oracle.Audio, workspace); err != nil {
		t.Fatal(err)
	}
	position := workspace.Transformer.Position
	assertRejected := func(name string, out, audio []float32) {
		t.Helper()
		before := append([]float32(nil), out...)
		if err := encoder.EncodeInto(out, audio, workspace); err == nil {
			t.Fatalf("%s: accepted overlapping buffers", name)
		}
		assertSliceClose(t, name+" output", out, before, 0)
		if workspace.Transformer.Position != position {
			t.Fatalf("%s: state position=%d want %d", name, workspace.Transformer.Position, position)
		}
	}
	shared := append([]float32(nil), oracle.Audio...)
	assertRejected("input/output", shared[1:1+latent], shared)
	out := append([]float32(nil), baseline...)
	assertRejected("input/workspace", out, workspace.A[:len(oracle.Audio)])
	assertRejected("output/workspace", workspace.B[:latent], oracle.Audio)
}

func TestMimiEncoderReleasedLatentCacheInterchange(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_POCKETTTS_VOICE_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_VOICE_MODEL")
	}
	data, err := os.ReadFile("testdata/mimi_encoder_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle mimiEncoderOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	cfg := releasedConfig(t)
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	encoder, err := LoadMimiEncoderCPU(file, cfg)
	file.Close()
	if err != nil {
		t.Fatal(err)
	}
	latents, frames, err := encoder.Encode(oracle.Audio)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	manifest := filepath.Join(root, "train_latents.jsonl")
	hash := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	relative := "latents/01234567/train_00000000.safetensors"
	line := fmt.Sprintf(`{"path":"audio.wav","duration":4,"transcript":"one","words":[{"word":"one","start":0.1,"end":3.2}],"latents_file":%q}`, relative)
	if err = os.WriteFile(manifest, []byte(line+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	meta := fmt.Sprintf(`{"stitch_frames":1,"noise_floor":0,"frame_rate":12.5,"weights_path":"model.safetensors","mimi_hash":%q}`+"\n", hash)
	if err = os.WriteFile(filepath.Join(root, "train_latents.meta.json"), []byte(meta), 0o600); err != nil {
		t.Fatal(err)
	}
	shardPath := filepath.Join(root, filepath.FromSlash(relative))
	if err = os.MkdirAll(filepath.Dir(shardPath), 0o755); err != nil {
		t.Fatal(err)
	}
	upstream, err := os.ReadFile("testdata/mimi_encoder_upstream_latents.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(shardPath, upstream, 0o600); err != nil {
		t.Fatal(err)
	}
	cache, err := OpenTrainingLatentCache(manifest, hash, cfg.Mimi.InnerDim, TrainingLatentCacheLimits{MaxEntries: 1, MaxFramesPerRow: frames, MaxTotalElements: len(latents)})
	if err != nil {
		t.Fatal(err)
	}
	loaded, gotFrames, channels, err := cache.LoadRow(0)
	if err != nil {
		t.Fatal(err)
	}
	if gotFrames != frames || channels != cfg.Mimi.InnerDim {
		t.Fatalf("shape [%d,%d]", gotFrames, channels)
	}
	assertSliceClose(t, "upstream Mimi cache oracle", loaded, oracle.Latent, 0)
	assertSliceClose(t, "native Mimi cache interchange", loaded, latents, 3e-5)
}

func TestMimiEncoderRejectsMalformed(t *testing.T) {
	if _, _, err := (*MimiEncoderCPU)(nil).Encode(nil); err == nil {
		t.Fatal("accepted nil encoder")
	}
	m := &MimiEncoderCPU{FrameSize: SamplesPerFrame}
	if _, _, err := m.Encode([]float32{1}); err == nil {
		t.Fatal("accepted malformed encoder")
	}
	if _, err := LoadMimiEncoderCPU(nil, Config{}); err == nil {
		t.Fatal("accepted nil source")
	}
}
