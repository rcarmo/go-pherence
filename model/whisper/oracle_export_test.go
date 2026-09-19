package whisper

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio/media"
)

// External-reference diagnostics only. Explicit opt-in writes bounded tensors
// for public fixtures into a NEW caller-named directory. No oracle in runtime.
func TestSpeechOracleExport(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_ORACLE_EXPORT") != "1" {
		t.Skip("explicit public oracle export required")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 120*time.Second {
		t.Fatal("timeout<=120s")
	}
	dest := os.Getenv("GO_PHERENCE_ORACLE_EXPORT_DIR")
	if dest == "" {
		t.Fatal("set fresh GO_PHERENCE_ORACLE_EXPORT_DIR")
	}
	if err := os.Mkdir(dest, 0700); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 110*time.Second)
	defer cancel()
	model, tok, policy := pinnedTinySpeechModel(t, ctx)
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Fatal(err)
	}
	ffprobe, err := exec.LookPath("ffprobe")
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := media.NewFFmpeg(media.Config{FFmpegPath: ffmpeg, FFprobePath: ffprobe})
	if err != nil {
		t.Fatal(err)
	}
	writeF32 := func(path string, a []float32) {
		b := make([]byte, 4*len(a))
		for i, v := range a {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
		}
		if err := os.WriteFile(path, b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	fixtures := append([]publicSpeechFixture(nil), mindsSpeechFixtures...)
	for i := range fixtures {
		fixtures[i].File = filepath.Join(os.Getenv("GO_PHERENCE_MINDS_FIXTURE_DIR"), fixtures[i].File)
	}
	fixtures = append(fixtures, publicSpeechFixture{Name: "jfk", Language: "en", File: os.Getenv("GO_PHERENCE_WHISPER_JFK_PATH"), SHA256: "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e", Samples: 176000, Reference: "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"})
	silencePath := filepath.Join(t.TempDir(), "silence.wav")
	writeSpeechFixturePCM(t, silencePath, make([]float32, 80000))
	silenceBytes, err := os.ReadFile(silencePath)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(silenceBytes)
	fixtures = append(fixtures, publicSpeechFixture{Name: "silence-5s", Language: "en", File: silencePath, SHA256: hex.EncodeToString(sum[:]), Samples: 80000})
	for _, fixture := range fixtures {
		f := fixture
		dir := filepath.Join(dest, f.Name)
		if err := os.Mkdir(dir, 0700); err != nil {
			t.Fatal(err)
		}
		pinnedSpeechFile(t, f.File, f.SHA256)
		decoded, err := adapter.DecodeToFile(ctx, f.File, filepath.Join(dir, "canonical.wav"))
		if err != nil {
			t.Fatal(err)
		}
		reader, err := media.OpenCanonicalPCM(ctx, decoded.Path)
		if err != nil {
			t.Fatal(err)
		}
		pcm := make([]float32, int(reader.Timeline().Samples))
		n, err := reader.ReadSamplesAt(ctx, pcm, 0)
		closeErr := reader.Close()
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		if err != nil || n != len(pcm) || int64(n) != f.Samples {
			t.Fatal("PCM", n, err)
		}
		padded := make([]float32, 480000)
		copy(padded, pcm)
		mel, frames, err := MelFlatFromSamplesCheckedContext(ctx, padded, model.Config)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := model.Encoder.ForwardContext(ctx, mel, frames)
		if err != nil {
			t.Fatal(err)
		}
		writeF32(filepath.Join(dir, "mel.f32"), mel)
		writeF32(filepath.Join(dir, "encoder.f32"), encoded)
		state, err := NewDecoderStateContext(ctx, model.Config, encoded, 1500, model.Decoder)
		if err != nil {
			t.Fatal(err)
		}
		v, err := checkedTimestampVocabulary(model.Config, tok, f.Language)
		if err != nil {
			t.Fatal(err)
		}
		opts, suppress, begin, err := resolvePCMGeneration(model.Config, v, PCMTranscribeOptions{Language: f.Language, Generation: policy, MaxNewTokens: 96}, model.Decoder.SuppressTokens, model.Decoder.BeginSuppressTokens)
		if err != nil {
			t.Fatal(err)
		}
		inputs := []int{}
		logits := []float32{}
		segments, decodeErr := decodeCheckedTimestamps(ctx, model.Config, tok, v, opts, suppress, begin, func(token int) ([]float32, error) {
			values := model.Decoder.ForwardToken(token, state)
			inputs = append(inputs, token)
			logits = append(logits, values...)
			return values, nil
		})
		writeF32(filepath.Join(dir, "logits.f32"), logits)
		text := ""
		for _, s := range segments {
			text += " " + s.Text
		}
		edits, words := speechFixtureWER(f.Reference, text)
		meta := map[string]any{"fixture": f.Name, "source_sha256": f.SHA256, "language": f.Language, "reference": f.Reference, "pcm_samples": len(pcm), "frames": frames, "encoder_shape": []int{1500, 384}, "mel_shape": []int{80, 3000}, "decoder_inputs": inputs, "logits_shape": []int{len(inputs), 51865}, "go_segments": segments, "word_edits": edits, "reference_words": words, "suppress": suppress, "begin_suppress": begin, "max_initial_timestamp_index": opts.MaxInitialTimestampIndex, "no_foreign_runtime": true, "eot": v.eot, "go_generated": append(append([]int(nil), inputs[3:]...), v.eot), "prompt_length": 3}
		if decodeErr != nil {
			meta["error"] = decodeErr.Error()
		}
		b, _ := json.MarshalIndent(meta, "", "  ")
		if err := os.WriteFile(filepath.Join(dir, "metadata.json"), append(b, '\n'), 0600); err != nil {
			t.Fatal(err)
		}
		t.Logf("ORACLE_EXPORT fixture=%s calls=%d word_edits=%d/%d error=%v", f.Name, len(inputs), edits, words, decodeErr)
		if decodeErr != nil {
			t.Fatal(fmt.Errorf("export decode: %w", decodeErr))
		}
	}
}
