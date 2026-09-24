package qwen3tts

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// This offline fixture records one natural EOS observation in the independent
// Rust/Candle CPU oracle. Go generation remains capped at 32 frames.
func TestPinnedHi64FrameEOSObservationFixture(t *testing.T) {
	root := filepath.Join("testdata", "customvoice_0b6_ryan_hello")
	data, err := os.ReadFile(filepath.Join(root, "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		OracleRevision string `json:"oracle_revision"`
		ScriptSHA      string `json:"hi_64_eos_probe_script_sha256"`
		CodesSHA       string `json:"hi_64_eos_probe_codes_sha256"`
		ObservationSHA string `json:"hi_64_eos_probe_observation_sha256"`
		WaveSHA        string `json:"hi_64_eos_probe_waveform_sha256"`
		Cap            int    `json:"hi_64_eos_probe_cap"`
		Frames         int    `json:"hi_64_eos_probe_frames"`
		EOSStep        int    `json:"hi_64_eos_probe_eos_step"`
		Seed           uint64 `json:"hi_64_eos_probe_seed"`
		PrefixSHA      string `json:"hi_32_eos_probe_codes_sha256"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.OracleRevision != "711ceee07cad92673f86de8997bdf54c30caa49f" ||
		ref.ScriptSHA != "adcfca5bc31f171391683e7ce46dc21a3cbe59bd0dc32c79d7620fdb836bf5ec" ||
		ref.CodesSHA != "076cd7e828d71c652aae38471928e7f111c6b5334e746e9bc9a09610b23bd449" ||
		ref.ObservationSHA != "e70f88327ec25195db3e3b6db765c9138029c860d255f4ae90ae0f201da2f6a1" ||
		ref.WaveSHA != "8011f727acf104e24e606b5aea4f3881bf55e1d5d575dd5063586a72dce33da2" ||
		ref.PrefixSHA != "88aafb544a81f0f09d4b83125c9b4f01370ade24fe24f47e9fcde86818a58391" ||
		ref.Cap != 64 || ref.Frames != 46 || ref.EOSStep != 46 || ref.Seed != 42 {
		t.Fatal("unexpected pinned Rust EOS provenance")
	}
	for _, item := range []struct {
		path, hash string
		size       int64
	}{
		{filepath.Join("..", "..", "scripts", "qwen3tts_probe_hi_64_eos.rs"), ref.ScriptSHA, 0},
		{filepath.Join(root, "probe_hi_64_codes.u32le"), ref.CodesSHA, int64(ref.Frames * 16 * 4)},
		{filepath.Join(root, "probe_hi_64_observation.json"), ref.ObservationSHA, 0},
		{filepath.Join(root, "probe_hi_32_codes.u32le"), ref.PrefixSHA, 32 * 16 * 4},
	} {
		if err := verifyReleasedFile(item.path, item.hash, item.size); err != nil {
			t.Fatal(err)
		}
	}
	var observation struct {
		Seed     uint64 `json:"seed"`
		Cap      int    `json:"cap"`
		Frames   int    `json:"frames"`
		EOSStep  *int   `json:"eos_step"`
		EOSToken int    `json:"eos_token_id"`
		Samples  int    `json:"samples"`
	}
	obs, err := os.ReadFile(filepath.Join(root, "probe_hi_64_observation.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(obs, &observation); err != nil {
		t.Fatal(err)
	}
	if observation.Seed != ref.Seed || observation.Cap != ref.Cap || observation.Frames != ref.Frames || observation.EOSStep == nil || *observation.EOSStep != ref.EOSStep || observation.EOSToken != int(CodecEOS) || observation.Samples != ref.Frames*1920 {
		t.Fatalf("unexpected EOS observation: %+v", observation)
	}
	codes, err := os.ReadFile(filepath.Join(root, "probe_hi_64_codes.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	prefix, err := os.ReadFile(filepath.Join(root, "probe_hi_32_codes.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(codes[:len(prefix)], prefix) {
		t.Fatal("Rust EOS observation diverged from the pinned 32-frame prefix")
	}
	for frame := 0; frame < ref.Frames; frame++ {
		if token := binary.LittleEndian.Uint32(codes[frame*64:]); token == CodecEOS {
			t.Fatalf("EOS stored as acoustic frame %d", frame)
		}
	}
	// Waveform bytes are retained outside Git. A missing waveform does not
	// count as a released-model parity pass; its hash remains pinned above.
}
