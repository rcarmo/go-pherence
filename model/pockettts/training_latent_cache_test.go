package pockettts

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const testMimiHash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func writeLatentCacheFixture(t *testing.T, values []float32, frames, channels int) string {
	t.Helper()
	root := t.TempDir()
	manifest := filepath.Join(root, "train_latents.jsonl")
	shardRelative := "latents/01234567/train_00000000.safetensors"
	line := fmt.Sprintf(`{"path":"audio.wav","duration":4,"transcript":"one two","words":[{"word":"one","start":0.1,"end":0.8},{"word":"two","start":1.3,"end":3.2}],"latents_file":%q}`, shardRelative)
	if err := os.WriteFile(manifest, []byte(line+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	metadata := fmt.Sprintf("{\n  \"stitch_frames\": 2,\n  \"noise_floor\": 0.0001,\n  \"frame_rate\": 12.5,\n  \"weights_path\": \"model.safetensors\",\n  \"mimi_hash\": %q\n}\n", testMimiHash)
	if err := os.WriteFile(filepath.Join(root, "train_latents.meta.json"), []byte(metadata), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := SaveTrainingLatentShard(filepath.Join(root, filepath.FromSlash(shardRelative)), values, frames, channels); err != nil {
		t.Fatal(err)
	}
	return manifest
}

func TestTrainingLatentCacheRoundTripAndStitch(t *testing.T) {
	values := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	manifest := writeLatentCacheFixture(t, values, 4, 2)
	limits := TrainingLatentCacheLimits{MaxEntries: 1, MaxFramesPerRow: 4, MaxTotalElements: 8}
	cache, err := OpenTrainingLatentCache(manifest, testMimiHash, 2, limits)
	if err != nil {
		t.Fatal(err)
	}
	metadata := cache.Metadata()
	shard, ok := cache.Shard(0)
	entry, entryOK := cache.Entry(0)
	if metadata.StitchFrames != 2 || metadata.FrameRate != 12.5 || cache.Len() != 1 || !ok || shard.Frames != 4 || !entryOK || entry.LatentsFile == "" {
		t.Fatalf("cache=%+v", cache)
	}
	entry.Words[0].Word = "mutated"
	again, _ := cache.Entry(0)
	if again.Words[0].Word == "mutated" {
		t.Fatal("entry accessor exposed mutable cache state")
	}
	got, frames, channels, err := cache.LoadRow(0)
	if err != nil {
		t.Fatal(err)
	}
	if frames != 4 || channels != 2 || !reflect.DeepEqual(got, values) {
		t.Fatalf("row=%v [%d,%d]", got, frames, channels)
	}
	stitched, err := StitchTrainingLatentTarget(got, frames, channels, 1, 3, metadata.StitchFrames, []float32{20, 21, 22, 23})
	if err != nil {
		t.Fatal(err)
	}
	if want := []float32{20, 21, 22, 23, 6, 7}; !reflect.DeepEqual(stitched, want) {
		t.Fatalf("stitched=%v want=%v", stitched, want)
	}
	shortPrefix := make([]float32, 18)
	for i := range shortPrefix {
		shortPrefix[i] = float32(30 + i)
	}
	short, err := StitchTrainingLatentTarget(got, frames, channels, 2, 1, 9, shortPrefix)
	if err != nil || !reflect.DeepEqual(short, shortPrefix) {
		t.Fatalf("short=%v err=%v", short, err)
	}
}

func TestTrainingLatentCacheAdmitsUpstreamNoCutFallback(t *testing.T) {
	for _, words := range []string{`[{"word":"one","start":0.1,"end":3.2}]`, `[{"word":"one","start":null,"end":0.8},{"word":"two","start":1.3,"end":null}]`} {
		manifest := writeLatentCacheFixture(t, []float32{0, 1, 2, 3}, 2, 2)
		body, _ := os.ReadFile(manifest)
		line := string(body)
		start := strings.Index(line, `"words":`)
		end := strings.Index(line[start:], `,"latents_file"`) + start
		line = line[:start] + `"words":` + words + line[end:]
		if err := os.WriteFile(manifest, []byte(line), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := OpenTrainingLatentCache(manifest, testMimiHash, 2, TrainingLatentCacheLimits{MaxEntries: 1, MaxFramesPerRow: 2, MaxTotalElements: 4}); err != nil {
			t.Fatal(err)
		}
	}
	manifest := writeLatentCacheFixture(t, []float32{0, 1, 2, 3}, 2, 2)
	body, _ := os.ReadFile(manifest)
	line := string(body)
	start := strings.Index(line, `"words":`)
	end := strings.Index(line[start:], `,"latents_file"`) + start
	line = line[:start] + `"words":[]` + line[end:]
	os.WriteFile(manifest, []byte(line), 0o600)
	if _, err := OpenTrainingLatentCache(manifest, testMimiHash, 2, TrainingLatentCacheLimits{MaxEntries: 1, MaxFramesPerRow: 2, MaxTotalElements: 4}); err == nil {
		t.Fatal("accepted latent manifest without alignments")
	}
}

func TestTrainingLatentCacheRejectsManifestMetadataAndLimits(t *testing.T) {
	values := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	limits := TrainingLatentCacheLimits{MaxEntries: 1, MaxFramesPerRow: 4, MaxTotalElements: 8}
	t.Run("hash", func(t *testing.T) {
		manifest := writeLatentCacheFixture(t, values, 4, 2)
		if _, err := OpenTrainingLatentCache(manifest, strings.Repeat("f", 64), 2, limits); err == nil {
			t.Fatal("accepted wrong Mimi hash")
		}
	})
	t.Run("metadata unknown field", func(t *testing.T) {
		manifest := writeLatentCacheFixture(t, values, 4, 2)
		path := strings.TrimSuffix(manifest, ".jsonl") + ".meta.json"
		body, _ := os.ReadFile(path)
		body = []byte(strings.Replace(string(body), "\n}", ",\n  \"extra\": 1\n}", 1))
		os.WriteFile(path, body, 0o600)
		if _, err := OpenTrainingLatentCache(manifest, testMimiHash, 2, limits); err == nil {
			t.Fatal("accepted unknown metadata")
		}
	})
	t.Run("path contract", func(t *testing.T) {
		manifest := writeLatentCacheFixture(t, values, 4, 2)
		body, _ := os.ReadFile(manifest)
		body = []byte(strings.Replace(string(body), "train_00000000", "wrong_00000000", 1))
		os.WriteFile(manifest, body, 0o600)
		if _, err := OpenTrainingLatentCache(manifest, testMimiHash, 2, limits); err == nil {
			t.Fatal("accepted non-upstream shard name")
		}
	})
	for name, changed := range map[string]TrainingLatentCacheLimits{
		"frames":   {MaxEntries: 1, MaxFramesPerRow: 3, MaxTotalElements: 8},
		"elements": {MaxEntries: 1, MaxFramesPerRow: 4, MaxTotalElements: 7},
	} {
		t.Run(name, func(t *testing.T) {
			manifest := writeLatentCacheFixture(t, values, 4, 2)
			if _, err := OpenTrainingLatentCache(manifest, testMimiHash, 2, changed); err == nil {
				t.Fatal("accepted over limit cache")
			}
		})
	}
	manifest := writeLatentCacheFixture(t, values, 4, 2)
	for _, bad := range []TrainingLatentCacheLimits{{}, {MaxEntries: 1, MaxFramesPerRow: 4}} {
		if _, err := OpenTrainingLatentCache(manifest, testMimiHash, 2, bad); err == nil {
			t.Fatal("accepted invalid limits")
		}
	}
}

func TestTrainingLatentCacheRejectsPostAdmissionReplacement(t *testing.T) {
	values := []float32{0, 1, 2, 3}
	manifest := writeLatentCacheFixture(t, values, 2, 2)
	limits := TrainingLatentCacheLimits{MaxEntries: 1, MaxFramesPerRow: 2, MaxTotalElements: 4}
	cache, err := OpenTrainingLatentCache(manifest, testMimiHash, 2, limits)
	if err != nil {
		t.Fatal(err)
	}
	shard := filepath.Join(filepath.Dir(manifest), "latents/01234567/train_00000000.safetensors")
	if err = SaveTrainingLatentShard(shard, []float32{4, 5, 6, 7}, 2, 2); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err = cache.LoadRow(0); err == nil {
		t.Fatal("accepted post-admission shard replacement")
	}
}

func TestTrainingLatentCacheRejectsDTypeMutationAndNonFinite(t *testing.T) {
	values := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	limits := TrainingLatentCacheLimits{MaxEntries: 1, MaxFramesPerRow: 4, MaxTotalElements: 8}
	manifest := writeLatentCacheFixture(t, values, 4, 2)
	shard := filepath.Join(filepath.Dir(manifest), "latents/01234567/train_00000000.safetensors")
	body, err := os.ReadFile(shard)
	if err != nil {
		t.Fatal(err)
	}
	headerLength := int(binary.LittleEndian.Uint64(body[:8]))
	header := string(body[8 : 8+headerLength])
	header = strings.Replace(header, `"dtype":"F32"`, `"dtype":"U32"`, 1)
	copy(body[8:8+headerLength], header)
	if err = os.WriteFile(shard, body, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err = OpenTrainingLatentCache(manifest, testMimiHash, 2, limits); err == nil {
		t.Fatal("accepted non-F32 latent shard")
	}

	manifest = writeLatentCacheFixture(t, values, 4, 2)
	cache, err := OpenTrainingLatentCache(manifest, testMimiHash, 2, limits)
	if err != nil {
		t.Fatal(err)
	}
	shard = filepath.Join(filepath.Dir(manifest), "latents/01234567/train_00000000.safetensors")
	body, _ = os.ReadFile(shard)
	headerLength = int(binary.LittleEndian.Uint64(body[:8]))
	binary.LittleEndian.PutUint32(body[8+headerLength:], math.Float32bits(float32(math.NaN())))
	if err = os.WriteFile(shard, body, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err = cache.LoadRow(0); err == nil {
		t.Fatal("accepted non-finite latent row")
	}
}

func TestTrainingLatentStitchRejectsMalformedWithoutMutation(t *testing.T) {
	stored := []float32{0, 1, 2, 3}
	original := append([]float32(nil), stored...)
	cases := []struct {
		storedFrames, channels, cut, target, stitch int
		fresh                                       []float32
	}{
		{2, 2, -1, 1, 1, []float32{1, 2}}, {2, 2, 2, 1, 1, []float32{1, 2}},
		{2, 2, 0, 3, 1, []float32{1, 2}}, {2, 2, 0, 1, 2, []float32{1, 2}},
		{2, 3, 0, 1, 1, []float32{1, 2, 3}}, {2, 2, 0, 1, 0, nil},
	}
	for i, test := range cases {
		if _, err := StitchTrainingLatentTarget(stored, test.storedFrames, test.channels, test.cut, test.target, test.stitch, test.fresh); err == nil {
			t.Fatalf("case %d accepted", i)
		}
		if !reflect.DeepEqual(stored, original) {
			t.Fatalf("case %d mutated stored latents", i)
		}
	}
	path := filepath.Join(t.TempDir(), "bad.safetensors")
	if err := SaveTrainingLatentShard(path, []float32{1, float32(math.Inf(1))}, 1, 2); err == nil {
		t.Fatal("saved non-finite shard")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("failed save published file: %v", err)
	}
}
