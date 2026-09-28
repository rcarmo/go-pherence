package qwen3tts

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// This default offline gate records a negative Rust-only EOS observation. It
// does not run the Go generator beyond its admitted sixteen-frame cap.
func TestPinnedHi32FrameEOSProbeFixture(t *testing.T) {
	root := filepath.Join("testdata", "customvoice_0b6_ryan_hello")
	data, err := os.ReadFile(filepath.Join(root, "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		OracleRevision string   `json:"oracle_revision"`
		ScriptSHA      string   `json:"hi_32_eos_probe_script_sha256"`
		CodesSHA       string   `json:"hi_32_eos_probe_codes_sha256"`
		WaveformSHA    string   `json:"hi_32_eos_probe_waveform_sha256"`
		Frames         int      `json:"hi_32_eos_probe_frames"`
		Seed           uint64   `json:"hi_32_eos_probe_seed"`
		EOSObserved    bool     `json:"hi_32_eos_probe_eos_observed"`
		HiSemantic     []uint32 `json:"hi_seeded_sixteen_semantic"`
		HiCodesSHA     string   `json:"hi_seeded_sixteen_codes_sha256"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.OracleRevision != "711ceee07cad92673f86de8997bdf54c30caa49f" || ref.ScriptSHA != "8acce5b9e8fb6cae755f23ae019416329a26b9fad1529f773bd0043e5e1b91b6" || ref.CodesSHA != "88aafb544a81f0f09d4b83125c9b4f01370ade24fe24f47e9fcde86818a58391" || ref.WaveformSHA != "831831b02d90b880311275807008585f82a9425f32db7401a4c6996397039a82" || ref.Frames != 32 || ref.Seed != 42 || ref.EOSObserved || len(ref.HiSemantic) != 16 || ref.HiCodesSHA != "17b16ec184d5a075805c057aa5eaa91b1960f97710f0f06206c6676f8b861a52" {
		t.Fatal("unexpected pinned Rust-only EOS probe provenance")
	}
	if err := verifyReleasedFile(filepath.Join("..", "..", "scripts", "qwen3tts_probe_hi_32_eos.rs"), ref.ScriptSHA, 0); err != nil {
		t.Fatal(err)
	}
	if err := verifyReleasedFile(filepath.Join(root, "probe_hi_32_codes.u32le"), ref.CodesSHA, 32*16*4); err != nil {
		t.Fatal(err)
	}
	if err := verifyReleasedFile(filepath.Join(root, "hi_seeded_sixteen_codes.u32le"), ref.HiCodesSHA, 16*16*4); err != nil {
		t.Fatal(err)
	}
	probe, err := os.ReadFile(filepath.Join(root, "probe_hi_32_codes.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	prefix, err := os.ReadFile(filepath.Join(root, "hi_seeded_sixteen_codes.u32le"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(probe[:len(prefix)], prefix) {
		t.Fatal("32-frame Rust probe diverged from pinned 16-frame prefix")
	}
	for frame := 0; frame < ref.Frames; frame++ {
		semantic := binary.LittleEndian.Uint32(probe[frame*64:])
		if semantic == CodecEOS {
			t.Fatalf("EOS stored as acoustic frame %d", frame)
		}
		if frame < 16 && semantic != ref.HiSemantic[frame] {
			t.Fatalf("prefix semantic frame %d", frame)
		}
	}
}
