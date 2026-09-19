package whisper

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
	"unicode"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/audio/media"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Fixture scoring only: case/punctuation-insensitive word edits. Not a corpus
// normaliser or a proposed multilingual WER standard. Apostrophes are removed;
// hyphens become boundaries, and Unicode letters/digits are retained.
func speechFixtureWords(s string) []string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else if r != '\'' && r != '’' {
			b.WriteByte(' ')
		}
	}
	return strings.Fields(b.String())
}
func speechFixtureWER(reference, hypothesis string) (int, int) {
	a, b := speechFixtureWords(reference), speechFixtureWords(hypothesis)
	row := make([]int, len(b)+1)
	for j := range row {
		row[j] = j
	}
	for i, x := range a {
		prev := row[0]
		row[0] = i + 1
		for j, y := range b {
			old := row[j+1]
			cost := 0
			if x != y {
				cost = 1
			}
			row[j+1] = min(min(row[j]+1, row[j+1]+1), prev+cost)
			prev = old
		}
	}
	return row[len(b)], len(a)
}
func TestSpeechFixtureWER(t *testing.T) {
	for _, c := range []struct {
		ref, hyp string
		edits, n int
	}{{"Ask not!", "ask NOT", 0, 2}, {"a b c", "a c", 1, 3}, {"a b", "a new b", 1, 2}, {"a b", "a c", 1, 2}, {"", "extra", 1, 0}, {"", "", 0, 0}, {"DON'T stop", "don't stop.", 0, 2}, {"Olá, mundo!", "olá mundo", 0, 2}} {
		e, n := speechFixtureWER(c.ref, c.hyp)
		if e != c.edits || n != c.n {
			t.Fatal(c, e, n)
		}
	}
}
func pinnedSpeechFile(t *testing.T, path, want string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(b)
	if hex.EncodeToString(sum[:]) != want {
		t.Fatal("unpinned file", filepath.Base(path))
	}
	return b
}
func pinnedTinySpeechModel(t *testing.T, ctx context.Context) (*Whisper, *Tokenizer, *CheckedGenerationConfig) {
	t.Helper()
	dir := os.Getenv("GO_PHERENCE_WHISPER_TINY_DIR")
	if dir == "" {
		t.Fatal("set GO_PHERENCE_WHISPER_TINY_DIR")
	}
	f, err := os.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.New()
	_, readErr := io.Copy(h, f)
	closeErr := f.Close()
	if readErr != nil || closeErr != nil {
		t.Fatal(readErr, closeErr)
	}
	if hex.EncodeToString(h.Sum(nil)) != "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395" {
		t.Fatal("unpinned weights")
	}
	config := pinnedSpeechFile(t, filepath.Join(dir, "config.json"), "ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98")
	generation := pinnedSpeechFile(t, filepath.Join(dir, "generation_config.json"), "a5d5325911f16e74001a72fa13d6e208eee51548f994646de1f4b4cc8b35b512")
	tokenPath := filepath.Join(dir, "tokenizer.json")
	pinnedSpeechFile(t, tokenPath, "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566")
	tok, err := LoadTokenizer(tokenPath)
	if err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	model, policy, err := LoadConfiguredModelSourceChecked(ctx, source, config, generation, tok)
	closeErr = source.Close()
	if err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	return model, tok, policy
}

type speechQualityResult struct {
	Path            string             `json:"path"`
	Phase           string             `json:"phase"`
	Block           int                `json:"block"`
	DurationSeconds float64            `json:"duration_seconds"`
	InferenceNS     int64              `json:"inference_ns"`
	Text            string             `json:"text"`
	Reference       string             `json:"reference"`
	WordEdits       int                `json:"word_edits"`
	ReferenceWords  int                `json:"reference_words"`
	WER             float64            `json:"wer"`
	Windows         []WindowTranscript `json:"windows"`
}

func TestVulkanPCMSpeechTiny(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_SPEECH_QUALITY") != "1" {
		t.Skip("explicit public speech compute opt-in required")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 120*time.Second {
		t.Fatal("use go test timeout<=120s")
	}
	input := os.Getenv("GO_PHERENCE_WHISPER_JFK_PATH")
	if input == "" {
		t.Fatal("set GO_PHERENCE_WHISPER_JFK_PATH")
	}
	pinnedSpeechFile(t, input, "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e")
	ctx, cancel := context.WithTimeout(context.Background(), 110*time.Second)
	defer cancel()
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Fatal(err)
	}
	ffprobe, err := exec.LookPath("ffprobe")
	if err != nil {
		t.Fatal(err)
	}
	version, err := exec.CommandContext(ctx, ffmpeg, "-version").Output()
	if err != nil {
		t.Fatal(err)
	}
	t.Log("SPEECH_MEDIA " + strings.SplitN(string(version), "\n", 2)[0])
	settings, _ := json.Marshal(map[string]any{"gomaxprocs": runtime.GOMAXPROCS(0), "linear_workers": linearWorkers, "block_m": blockM, "int8": useInt8, "attention_int8": attnInt8, "attention_f16": attnF16, "nvidia_disabled": os.Getenv("GO_PHERENCE_DISABLE_NVIDIA")})
	t.Log("SPEECH_SETTINGS " + string(settings))
	adapter, err := media.NewFFmpeg(media.Config{FFmpegPath: ffmpeg, FFprobePath: ffprobe})
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	decoded, err := adapter.DecodeToFile(ctx, input, filepath.Join(t.TempDir(), "canonical.wav"))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("SPEECH_DECODE ns=%d samples=%d sample_rate=%d", time.Since(start).Nanoseconds(), decoded.Timeline.Samples, decoded.Timeline.SampleRate)
	if decoded.Timeline.Samples != 176000 || decoded.Timeline.SampleRate != 16000 {
		t.Fatal("changed JFK canonical timeline", decoded.Timeline)
	}
	reader, err := media.OpenCanonicalPCM(ctx, decoded.Path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	model, tok, policy := pinnedTinySpeechModel(t, ctx)
	if !vk.VulkanInit() {
		t.Fatal("Vulkan unavailable")
	}
	name := vk.VulkanDeviceName()
	expected := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if expected == "" || !strings.Contains(name, expected) || strings.Contains(strings.ToLower(name), "llvmpipe") {
		t.Fatal("unexpected device", name)
	}
	before := vk.VulkanMemoryStats()
	start = time.Now()
	gpu, err := NewVulkanEncoder(ctx, model.Encoder, model.Config.MaxLength)
	if gpu != nil {
		t.Cleanup(func() {
			if err := gpu.Close(); err != nil {
				t.Error(err)
			}
			nativeEncoderMemory(t, before)
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("SPEECH_RESIDENT construct_ns=%d device=%s", time.Since(start).Nanoseconds(), name)
	const reference = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"
	if len(speechFixtureWords(reference)) != 22 {
		t.Fatal("changed reference word count")
	}
	var previous []WindowTranscript
	run := func(path, phase string, block int) int64 {
		opts := PCMTranscribeOptions{Language: "en", Generation: policy, MaxNewTokens: 96}
		if path == "vulkan" {
			opts.VulkanEncoder = gpu
		}
		result := speechQualityResult{Path: path, Phase: phase, Block: block, DurationSeconds: float64(reader.Timeline().Samples) / 16000, Reference: reference}
		start := time.Now()
		err := model.TranscribePCMWindows(ctx, reader, int64(reader.Timeline().Samples), tok, opts, func(window WindowTranscript) error { result.Windows = append(result.Windows, window); return nil })
		result.InferenceNS = time.Since(start).Nanoseconds()
		for _, w := range result.Windows {
			for _, s := range w.Segments {
				result.Text += " " + s.Text
			}
		}
		result.Text = strings.TrimSpace(result.Text)
		result.WordEdits, result.ReferenceWords = speechFixtureWER(reference, result.Text)
		result.WER = float64(result.WordEdits) / float64(result.ReferenceWords)
		raw, _ := json.Marshal(result)
		t.Log("SPEECH_RESULT " + string(raw))
		if err != nil {
			t.Fatal("speech failed (earlier callbacks retained)", err)
		}
		if len(result.Windows) != 1 || len(result.Windows[0].Segments) == 0 {
			t.Fatal("missing speech segments")
		}
		if result.WER > .15 {
			t.Fatal("JFK fixture WER above fixed15% smoke gate", result.WER)
		}
		end := 0.
		for _, s := range result.Windows[0].Segments {
			if s.Start < end || s.End <= s.Start || s.End > result.DurationSeconds {
				t.Fatal("invalid speech timestamps", s)
			}
			end = s.End
		}
		if previous != nil && !reflect.DeepEqual(previous, result.Windows) {
			t.Fatal("CPU/GPU/repeated token/timestamp/window mismatch")
		}
		previous = result.Windows
		return result.InferenceNS
	}
	run("vulkan", "initial", -1)
	compare := os.Getenv("GO_PHERENCE_TEST_SPEECH_CPU_COMPARE") == "1"
	if compare {
		run("cpu", "initial", -1)
	}
	if os.Getenv("GO_PHERENCE_TEST_SPEECH_TIMING") == "1" {
		if !compare {
			t.Fatal("timing requires explicit CPU comparison")
		}
		// Initial validated call warms each path. Alternating ABBA/BAAB blocks
		// use identical decodedPCM/generation and validate every emitted token.
		byPath := map[string][]int64{"vulkan": {}, "cpu": {}}
		for block := 0; block < 4; block++ {
			order := []string{"vulkan", "cpu", "cpu", "vulkan"}
			if block%2 == 1 {
				order = []string{"cpu", "vulkan", "vulkan", "cpu"}
			}
			for _, path := range order {
				ns := run(path, "warm", block)
				byPath[path] = append(byPath[path], ns)
			}
		}
		medians := map[string]float64{}
		for path, values := range byPath {
			sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
			medians[path] = float64(values[3]+values[4]) / 2
		}
		raw, _ := json.Marshal(map[string]any{"samples_per_path": 8, "inference_median_ns": medians, "cpu_over_vulkan": medians["cpu"] / medians["vulkan"], "vulkan_realtime_multiple": 11e9 / medians["vulkan"], "duration_seconds": 11, "output_identical": true, "includes_frontend_crossKV_decode": true, "includes_media_model_load": false})
		t.Log("SPEECH_TIMING " + string(raw))
	}
	// Separate sequential diagnostic follows the same single-window API steps,
	// but exposes stage wall times and token-call counts; excluded from ABBA.
	samples := make([]float32, model.Config.MaxLength*160)
	n, err := reader.ReadSamplesAt(ctx, samples, 0)
	if err != io.EOF || n != 176000 {
		t.Fatal("diagnosticPCMread", n, err)
	}
	start = time.Now()
	mel, frames, err := MelFlatFromSamplesCheckedContext(ctx, samples, model.Config)
	if err != nil {
		t.Fatal(err)
	}
	frontNS := time.Since(start).Nanoseconds()
	start = time.Now()
	encoded, err := gpu.Forward(ctx, mel)
	if err != nil {
		t.Fatal(err)
	}
	encoderNS := time.Since(start).Nanoseconds()
	start = time.Now()
	state, err := NewDecoderStateContext(ctx, model.Config, encoded, (frames+1)/2, model.Decoder)
	if err != nil {
		t.Fatal(err)
	}
	crossNS := time.Since(start).Nanoseconds()
	vocab, err := checkedTimestampVocabulary(model.Config, tok, "en")
	if err != nil {
		t.Fatal(err)
	}
	opts, suppress, begin, err := resolvePCMGeneration(model.Config, vocab, PCMTranscribeOptions{Language: "en", Generation: policy, MaxNewTokens: 96}, model.Decoder.SuppressTokens, model.Decoder.BeginSuppressTokens)
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	start = time.Now()
	segments, err := decodeCheckedTimestamps(ctx, model.Config, tok, vocab, opts, suppress, begin, func(token int) ([]float32, error) { calls++; return model.Decoder.ForwardToken(token, state), nil })
	if err != nil {
		t.Fatal(err)
	}
	decodeNS := time.Since(start).Nanoseconds()
	if !reflect.DeepEqual(segments, previous[0].Segments) {
		t.Fatal("stage diagnostic differs fromPCMAPI")
	}
	textTokens := 0
	for _, s := range segments {
		textTokens += len(s.Tokens)
	}
	raw, _ := json.Marshal(map[string]any{"frontend_ns": frontNS, "encoder_ns": encoderNS, "cross_kv_ns": crossNS, "decoder_ns": decodeNS, "decoder_calls": calls, "text_tokens": textTokens, "separate_diagnostic": true, "matches_PCM_API": true})
	t.Log("SPEECH_STAGES " + string(raw))
}
