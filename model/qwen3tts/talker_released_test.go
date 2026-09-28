package qwen3tts

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// TestTalkerReleasedPrefill compares the independently generated Candle F32
// boundary with the native Go Talker. It is deliberately opt-in: the released
// checkpoint must be supplied and hashed before its tensor payload is loaded.
func TestTalkerReleasedPrefill(t *testing.T) {
	const envName = "GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR"
	dir := os.Getenv(envName)
	if dir == "" {
		t.Skipf("set %s to the pinned Qwen3-TTS 0.6B CustomVoice directory", envName)
	}
	root := filepath.Join("testdata", "customvoice_0b6_ryan_hello")
	var ref struct {
		Schema                 int       `json:"schema"`
		PrefillPositions       int       `json:"prefill_positions"`
		FirstSemanticToken     int       `json:"first_semantic_token"`
		ModelRepository        string    `json:"model_repository"`
		ModelRevision          string    `json:"model_revision"`
		ModelSafetensorsSize   int64     `json:"model_safetensors_size"`
		ModelSafetensorsSHA256 string    `json:"model_safetensors_sha256"`
		ModelConfigSHA256      string    `json:"model_config_sha256"`
		TokenizerVocabSHA256   string    `json:"tokenizer_vocab_sha256"`
		TokenizerMergesSHA256  string    `json:"tokenizer_merges_sha256"`
		OracleRepository       string    `json:"oracle_repository"`
		OracleRevision         string    `json:"oracle_revision"`
		OracleScriptSHA256     string    `json:"oracle_script_sha256"`
		HiddenShape            []int     `json:"hidden_shape"`
		LogitsShape            []int     `json:"logits_shape"`
		HiddenSHA256           string    `json:"hidden_sha256"`
		LogitsSHA256           string    `json:"logits_sha256"`
		HiddenMaxAbsThreshold  float64   `json:"hidden_max_abs_threshold"`
		LogitsMaxAbsThreshold  float64   `json:"logits_max_abs_threshold"`
		Prompt                 PromptIDs `json:"prompt"`
	}
	data, err := os.ReadFile(filepath.Join(root, "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.Schema != 1 || ref.ModelRepository != "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" || ref.ModelRevision != "85e237c12c027371202489a0ec509ded67b5e4b5" || ref.OracleRepository != "https://github.com/TrevorS/qwen3-tts-rs" || ref.OracleRevision != "711ceee07cad92673f86de8997bdf54c30caa49f" || ref.OracleScriptSHA256 != "d774a67b956f28c7f5cd3beb4e3906ad929208d40fc08c3685721f7498239a0c" || ref.FirstSemanticToken != 1995 || ref.ModelSafetensorsSize != 1811626576 || ref.ModelSafetensorsSHA256 != "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb" || ref.ModelConfigSHA256 != "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455" || ref.TokenizerVocabSHA256 != "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910" || ref.TokenizerMergesSHA256 != "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3" || ref.HiddenSHA256 != "a3d0a5ece4565507bd72e990605b7d080840fd751a123ff81eba70a2b69bc4c3" || ref.LogitsSHA256 != "a21ec2ea111681b69195b8bedbdd089a5ac18a64eb7eec32b76fa3cf34c90149" || ref.HiddenMaxAbsThreshold != 5e-5 || ref.LogitsMaxAbsThreshold != 3e-5 {
		t.Fatal("unexpected released Qwen3-TTS fixture provenance")
	}
	for _, asset := range []struct {
		name, sha string
		bytes     int64
	}{
		{"model.safetensors", ref.ModelSafetensorsSHA256, ref.ModelSafetensorsSize},
		{"config.json", ref.ModelConfigSHA256, 0},
		{"vocab.json", ref.TokenizerVocabSHA256, 0},
		{"merges.txt", ref.TokenizerMergesSHA256, 0},
	} {
		if err := verifyReleasedFile(filepath.Join(dir, asset.name), asset.sha, asset.bytes); err != nil {
			t.Fatal(err)
		}
	}
	cfg, err := ReadModelDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := LoadTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := BuildCustomVoicePrompt(tok, "Hello world", Ryan, English)
	if err != nil || !reflect.DeepEqual(prompt, ref.Prompt) {
		t.Fatalf("released tokenizer prompt=%+v want=%+v err=%v", prompt, ref.Prompt, err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: Ryan, Language: English}, Prompt: prompt, MaxFrames: 12})
	if err != nil {
		t.Fatal(err)
	}
	model, err := LoadTalkerCPUFromDir(dir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Prefill(plan)
	if err != nil {
		t.Fatal(err)
	}
	if result.SemanticToken != uint32(ref.FirstSemanticToken) || result.PrefillTokens != ref.PrefillPositions {
		t.Fatalf("Talker token=%d positions=%d want=%d/%d", result.SemanticToken, result.PrefillTokens, ref.FirstSemanticToken, ref.PrefillPositions)
	}
	for _, row := range []struct {
		name   string
		got    []float32
		shape  []int
		sha    string
		maxAbs float64
	}{
		{"hidden", result.Hidden, ref.HiddenShape, ref.HiddenSHA256, ref.HiddenMaxAbsThreshold},
		{"logits", result.Logits, ref.LogitsShape, ref.LogitsSHA256, ref.LogitsMaxAbsThreshold},
	} {
		if len(row.shape) != 1 || len(row.got) != row.shape[0] {
			t.Fatalf("%s shape=%v got=%d", row.name, row.shape, len(row.got))
		}
		blob, err := os.ReadFile(filepath.Join(root, row.name+".f32le"))
		if err != nil {
			t.Fatal(err)
		}
		if len(blob) != len(row.got)*4 || fmt.Sprintf("%x", sha256.Sum256(blob)) != row.sha {
			t.Fatalf("%s independent reference hash or length mismatch", row.name)
		}
		var maxAbs float64
		for i, actual := range row.got {
			want := math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
			if !finiteTalkerValue(actual) || !finiteTalkerValue(want) {
				t.Fatalf("%s[%d] non-finite value", row.name, i)
			}
			delta := math.Abs(float64(actual) - float64(want))
			if delta > maxAbs {
				maxAbs = delta
			}
		}
		if maxAbs > row.maxAbs {
			t.Fatalf("%s max absolute difference=%g exceeds pinned threshold=%g", row.name, maxAbs, row.maxAbs)
		}
		t.Logf("%s max absolute difference=%g threshold=%g", row.name, maxAbs, row.maxAbs)
	}
}

func verifyReleasedFile(path, want string, size int64) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || size > 0 && info.Size() != size {
		return fmt.Errorf("%s: unexpected artifact type or size=%d want=%d", path, info.Size(), size)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if fmt.Sprintf("%x", h.Sum(nil)) != want {
		return fmt.Errorf("%s: SHA-256 mismatch", path)
	}
	return nil
}

func finiteTalkerValue(v float32) bool {
	return !math.IsInf(float64(v), 0) && !math.IsNaN(float64(v))
}
