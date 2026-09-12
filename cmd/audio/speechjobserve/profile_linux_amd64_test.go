//go:build linux && amd64

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

func putAsset(t *testing.T, dir, name string, b []byte, mode os.FileMode) Asset {
	t.Helper()
	p := filepath.Join(dir, name)
	if e := os.WriteFile(p, b, mode); e != nil {
		t.Fatal(e)
	}
	return Asset{p, hashBytes(b)}
}
func toyAssets(t *testing.T) ServerConfig {
	t.Helper()
	c := baseConfig(t)
	c.Threads = runtime.GOMAXPROCS(0)
	t.Setenv("GOMAXPROCS", fmt.Sprint(c.Threads))
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	dir := t.TempDir()
	model := map[string]any{"model_type": "whisper", "activation_function": "gelu", "is_encoder_decoder": true, "scale_embedding": false, "architectures": []string{"WhisperForConditionalGeneration"}, "num_mel_bins": 80, "d_model": 2, "encoder_layers": 0, "decoder_layers": 0, "encoder_attention_heads": 1, "decoder_attention_heads": 1, "encoder_ffn_dim": 2, "decoder_ffn_dim": 2, "vocab_size": 51865, "max_source_positions": 1, "max_target_positions": 8, "bos_token_id": 50257, "eos_token_id": 50257, "pad_token_id": 50257, "decoder_start_token_id": 50258}
	generation := map[string]any{"bos_token_id": 50257, "eos_token_id": 50257, "pad_token_id": 50257, "decoder_start_token_id": 50258, "no_timestamps_token_id": 50363, "is_multilingual": true, "lang_to_id": map[string]int{"<|en|>": 50259, "<|pt|>": 50267}, "task_to_id": map[string]int{"translate": 50358, "transcribe": 50359}, "max_length": 8, "max_initial_timestamp_index": 0, "suppress_tokens": []int{}, "begin_suppress_tokens": []int{}}
	langs := map[string]int{}
	for id := 50259; id < 50358; id++ {
		name := fmt.Sprintf("<|lang%d|>", id)
		if id == 50259 {
			name = "<|en|>"
		}
		if id == 50267 {
			name = "<|pt|>"
		}
		langs[name] = id
	}
	generation["lang_to_id"] = langs
	b, _ := json.Marshal(model)
	c.ModelConfig = putAsset(t, dir, "config.json", b, 0600)
	b, _ = json.Marshal(generation)
	c.Generation = putAsset(t, dir, "generation.json", b, 0600)
	vocab := map[string]int{}
	for i := 0; i < 51865; i++ {
		vocab[fmt.Sprintf("text%d", i)] = i
	}
	for id, text := range map[int]string{50257: "<|endoftext|>", 50258: "<|startoftranscript|>", 50259: "<|en|>", 50267: "<|pt|>", 50358: "<|translate|>", 50359: "<|transcribe|>", 50363: "<|notimestamps|>"} {
		delete(vocab, fmt.Sprintf("text%d", id))
		vocab[text] = id
	}
	for name, id := range langs {
		delete(vocab, fmt.Sprintf("text%d", id))
		vocab[name] = id
	}
	for i := 0; i <= 1500; i++ {
		id := 50364 + i
		delete(vocab, fmt.Sprintf("text%d", id))
		vocab[fmt.Sprintf("<|%.2f|>", float64(i)/50)] = id
	}
	b, _ = json.Marshal(map[string]any{"model": map[string]any{"vocab": vocab}, "added_tokens": []any{}, "normalizer": nil})
	c.Tokenizer = putAsset(t, dir, "tokenizer.json", b, 0600)
	type tensor struct {
		shape []int
		v     []float32
	}
	tensors := map[string]tensor{}
	add := func(n string, shape ...int) {
		count := 1
		for _, d := range shape {
			count *= d
		}
		tensors[n] = tensor{shape, make([]float32, count)}
	}
	add("model.encoder.conv1.weight", 2, 80, 3)
	add("model.encoder.conv1.bias", 2)
	add("model.encoder.conv2.weight", 2, 2, 3)
	add("model.encoder.conv2.bias", 2)
	add("model.encoder.embed_positions.weight", 1, 2)
	add("model.encoder.layer_norm.weight", 2)
	add("model.encoder.layer_norm.bias", 2)
	add("model.decoder.embed_tokens.weight", 51865, 2)
	add("model.decoder.embed_positions.weight", 8, 2)
	add("model.decoder.layer_norm.weight", 2)
	add("model.decoder.layer_norm.bias", 2)
	for _, n := range []string{"model.encoder.layer_norm.weight", "model.decoder.layer_norm.weight"} {
		for i := range tensors[n].v {
			tensors[n].v[i] = 1
		}
	}
	tensors["model.decoder.embed_tokens.weight"].v[50257*2] = 10
	tensors["model.decoder.embed_tokens.weight"].v[50257*2+1] = -10
	for i := 0; i < 8; i++ {
		tensors["model.decoder.embed_positions.weight"].v[i*2] = 1
		tensors["model.decoder.embed_positions.weight"].v[i*2+1] = -1
	}
	names := []string{}
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	header := map[string]safetensors.TensorInfo{}
	var payload bytes.Buffer
	for _, name := range names {
		v := tensors[name]
		start := payload.Len()
		for _, f := range v.v {
			binary.Write(&payload, binary.LittleEndian, math.Float32bits(f))
		}
		header[name] = safetensors.TensorInfo{DType: "F32", Shape: v.shape, DataOffsets: [2]int{start, payload.Len()}}
	}
	h, _ := json.Marshal(header)
	var file bytes.Buffer
	binary.Write(&file, binary.LittleEndian, uint64(len(h)))
	file.Write(h)
	file.Write(payload.Bytes())
	c.Weights = putAsset(t, dir, "model.safetensors", file.Bytes(), 0600)
	c.FFmpeg = putAsset(t, dir, "ffmpeg-fixture", []byte("not executed by metadata tests"), 0700)
	c.FFprobe = c.FFmpeg
	return c
}
func TestProfileCheckDoesNotCreateStoreOrLoadInference(t *testing.T) {
	c := toyAssets(t)
	raw, _ := json.Marshal(c)
	file := putAsset(t, t.TempDir(), "server.json", raw, 0600)
	var out bytes.Buffer
	if e := start(context.Background(), file.Path, true, &out); e != nil {
		t.Fatal(e)
	}
	if !strings.Contains(out.String(), `"model_loaded":false`) {
		t.Fatal(out.String())
	}
	if _, e := os.Stat(c.Store); !os.IsNotExist(e) {
		t.Fatal("check created store", e)
	}
	if e := start(context.Background(), file.Path, false, &out); e == nil {
		t.Fatal("execution implicit")
	}
	profiles, e := buildProfile(context.Background(), c, true)
	if e != nil || len(profiles) != 1 || len(profiles[0].Stages) != 4 {
		t.Fatal(profiles, e)
	}
	if profiles[0].Stages[0].Name != "decode" || profiles[0].Stages[1].Name != "asr-windows" || profiles[0].Stages[2].Name != "transcript" || profiles[0].Stages[3].Name != "vtt" {
		t.Fatal("profile order")
	}
}
func TestProfileMetadataAdmission(t *testing.T) {
	c := toyAssets(t)
	for _, kind := range []string{"hash", "weightcap", "ownedcap", "overlap", "language", "header", "generation-timestamp", "generation-tokens"} {
		bad := c
		switch kind {
		case "hash":
			bad.Tokenizer.SHA256 = hashBytes([]byte("bad"))
		case "weightcap":
			bad.Limits.WeightBytes = 8
		case "ownedcap":
			bad.Limits.OwnedWeightBytes = 100
		case "overlap":
			bad.Profile.OverlapSamples = 161
		case "language":
			bad.Profile.Language = "zz"
		case "generation-timestamp":
			bad.Profile.MaxInitialTimestampIndex = 1
		case "generation-tokens":
			bad.Profile.MaxNewTokens = 6
		case "header":
			b := make([]byte, 8)
			binary.LittleEndian.PutUint64(b, 9<<20)
			bad.Weights = putAsset(t, t.TempDir(), "bad.safetensors", b, 0600)
		}
		if _, e := buildProfile(context.Background(), bad, false); e == nil {
			t.Fatal(kind)
		}
	}
	t.Setenv("WHISPER_INT8", "1")
	if e := validateRuntime(c); e == nil {
		t.Fatal("runtime mode changed")
	}
}
func TestServingProfileSyntheticFFmpegToVTT(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_FFMPEG") != "1" {
		t.Skip("explicit FFmpeg integration with generated toy model")
	}
	c := toyAssets(t)
	for _, target := range []*Asset{&c.FFmpeg, &c.FFprobe} {
		name := "ffmpeg"
		if target == &c.FFprobe {
			name = "ffprobe"
		}
		p, e := exec.LookPath(name)
		if e != nil {
			t.Fatal(e)
		}
		p, _ = filepath.Abs(p)
		b, e := os.ReadFile(p)
		if e != nil {
			t.Fatal(e)
		}
		*target = Asset{p, hashBytes(b)}
	}
	profiles, e := buildProfile(context.Background(), c, true)
	if e != nil {
		t.Fatal(e)
	}
	s, e := speechjob.Open(c.Store, speechjob.Limits{MaxJobs: c.Limits.Jobs, MaxUploadBytes: c.Limits.UploadBytes, MaxArtifactBytes: c.Limits.ArtifactBytes, MaxBytes: c.Limits.StoreBytes})
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	const n = 320
	wav := make([]byte, 44+2*n)
	copy(wav, "RIFF")
	binary.LittleEndian.PutUint32(wav[4:], uint32(len(wav)-8))
	copy(wav[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(wav[16:], 16)
	binary.LittleEndian.PutUint16(wav[20:], 1)
	binary.LittleEndian.PutUint16(wav[22:], 1)
	binary.LittleEndian.PutUint32(wav[24:], 16000)
	binary.LittleEndian.PutUint32(wav[28:], 32000)
	binary.LittleEndian.PutUint16(wav[32:], 2)
	binary.LittleEndian.PutUint16(wav[34:], 16)
	copy(wav[36:], "data")
	binary.LittleEndian.PutUint32(wav[40:], 2*n)
	p := profiles[0]
	j, e := s.Create(context.Background(), "toy.wav", p.Configuration, bytes.NewReader(wav))
	if e != nil {
		t.Fatal(e)
	}
	j, e = s.Run(context.Background(), j.ID, p.Configuration, p.Stages, nil)
	if e != nil || j.Status != speechjob.Complete {
		t.Fatal(j, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), j.ID, "vtt")
	if e != nil {
		t.Fatal(e)
	}
	var b bytes.Buffer
	b.ReadFrom(r)
	r.Close()
	if b.String() != "WEBVTT\n\n" {
		t.Fatal(b.String())
	}
}
