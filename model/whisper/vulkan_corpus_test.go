package whisper

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/audio/media"
)

type publicSpeechFixture struct {
	Name, Language, File, SHA256, Reference string
	Samples                                 int64
}

var mindsSpeechFixtures = []publicSpeechFixture{
	{"minds-pt-0", "pt", "pt-real-0-source.wav", "fc084982ad50c6ea6cf066f08374b9b3aaa628d9a9accb167be5ae9376dbd275", "Bom dia estou a ligar porque precisava de informações sobre como é que eu posso depositar dinheiro na minha conta", 144726},
	{"minds-pt-1", "pt", "pt-real-1-source.wav", "aacee91914f902b0949425ee29ac984c48a34df55e5c195e65e0c8ff66977484", "Como faço para transferir dinheiro para a minha conta", 75094},
	{"minds-fr-0", "fr", "fr-real-0-source.wav", "84defdc828ef59cec10364354fbc284bc2cc683fdd4a5edd5863b7bb2c6123a8", "je souhaite changer mon adresse", 60074},
}

// Synthetic composition only uses verified public speech and digital silence.
// This is not natural long-form audio or an annotated multi-speaker corpus.
func writeSpeechFixturePCM(t *testing.T, path string, samples []float32) {
	t.Helper()
	b := make([]byte, 44+2*len(samples))
	copy(b, "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1)
	binary.LittleEndian.PutUint16(b[22:], 1)
	binary.LittleEndian.PutUint32(b[24:], 16000)
	binary.LittleEndian.PutUint32(b[28:], 32000)
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], uint32(2*len(samples)))
	for i, v := range samples {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v < -1 || v >= 1 {
			t.Fatal("fixturePCM out of range")
		}
		binary.LittleEndian.PutUint16(b[44+2*i:], uint16(int16(v*32768)))
	}
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
}

type speechCorpusResult struct {
	Fixture            string             `json:"fixture"`
	Language           string             `json:"language"`
	Path               string             `json:"backend"`
	Samples            int64              `json:"samples"`
	InferenceNS        int64              `json:"inference_ns"`
	Text               string             `json:"text"`
	Reference          string             `json:"reference"`
	Edits              int                `json:"word_edits"`
	Words              int                `json:"reference_words"`
	WER                *float64           `json:"wer"`
	Silent             bool               `json:"silence"`
	EmptyOutput        bool               `json:"empty_output"`
	Error              string             `json:"error,omitempty"`
	Windows            []WindowTranscript `json:"-"` // bounded per-window records below
	SkipDigitalSilence bool               `json:"skip_digital_silence"`
}

func TestVulkanPublicCorpus(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_VULKAN_CORPUS") != "1" {
		t.Skip("explicit public corpus diagnostic opt-in required")
	}
	deadline, ok := t.Deadline()
	if !ok || time.Until(deadline) > 120*time.Second {
		t.Fatal("timeout<=120s required")
	}
	root := os.Getenv("GO_PHERENCE_MINDS_FIXTURE_DIR")
	jfk := os.Getenv("GO_PHERENCE_WHISPER_JFK_PATH")
	if root == "" || jfk == "" {
		t.Fatal("set MINDS fixture directory and JFK path")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 110*time.Second)
	defer cancel()
	model, tok, policy := pinnedTinySpeechModel(t, ctx)
	if !vk.VulkanInit() {
		t.Fatal("Vulkan unavailable")
	}
	device := vk.VulkanDeviceName()
	want := os.Getenv("GO_PHERENCE_VULKAN_DEVICE")
	if want == "" || !strings.Contains(device, want) || strings.Contains(strings.ToLower(device), "llvmpipe") {
		t.Fatal("unexpected device", device)
	}
	before := vk.VulkanMemoryStats()
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
	version, err := exec.CommandContext(ctx, ffmpeg, "-version").Output()
	if err != nil {
		t.Fatal(err)
	}
	t.Log("CORPUS_MEDIA " + strings.SplitN(string(version), "\n", 2)[0])
	fixtures := append([]publicSpeechFixture(nil), mindsSpeechFixtures...)
	dir := t.TempDir()
	for i, f := range fixtures {
		p := filepath.Join(root, f.File)
		pinnedSpeechFile(t, p, f.SHA256)
		fixtures[i].File = p
	}
	// Read the exact pinned JFK source, preserving its PCM quantisation.
	pinnedSpeechFile(t, jfk, "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e")
	jfkReader, err := media.OpenCanonicalPCM(ctx, jfk)
	if err != nil {
		t.Fatal(err)
	}
	speech := make([]float32, 176000)
	n, err := jfkReader.ReadSamplesAt(ctx, speech, 0)
	closeErr := jfkReader.Close()
	if err != nil || closeErr != nil || n != len(speech) {
		t.Fatal(n, err, closeErr)
	}
	const jfkText = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"
	for _, s := range []struct {
		name      string
		seconds   int
		starts    []int
		reference string
	}{{"silence-5s", 5, nil, ""}, {"jfk-padded-21s", 21, []int{5}, jfkText}, {"jfk-three-windows-63s", 63, []int{2, 42}, jfkText + " " + jfkText}} {
		pcm := make([]float32, s.seconds*16000)
		for _, start := range s.starts {
			copy(pcm[start*16000:], speech)
		}
		path := filepath.Join(dir, s.name+".wav")
		writeSpeechFixturePCM(t, path, pcm)
		bytes, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256(bytes)
		fixtures = append(fixtures, publicSpeechFixture{s.name, "en", path, hex.EncodeToString(hash[:]), s.reference, int64(len(pcm))})
	}
	for index, f := range fixtures {
		decoded, err := adapter.DecodeToFile(ctx, f.File, filepath.Join(dir, fmt.Sprintf("decoded-%d.wav", index)))
		if err != nil {
			t.Fatal(f.Name, err)
		}
		if int64(decoded.Timeline.Samples) != f.Samples || decoded.Timeline.SampleRate != 16000 {
			t.Fatal("fixture PCM extent", f.Name, decoded.Timeline, f.Samples)
		}
		reader, err := media.OpenCanonicalPCM(ctx, decoded.Path)
		if err != nil {
			t.Fatal(err)
		}
		// Diagnose quality independently from backend correctness: don't convert a
		// non-English WER/silence failure into passing quality merely due to parity.
		modes := []bool{false}
		if f.Name == "silence-5s" || f.Name == "jfk-three-windows-63s" {
			modes = append(modes, true)
		}
		for _, skipSilence := range modes {
			var outcomes []speechCorpusResult
			for _, backend := range []string{"vulkan", "cpu"} {
				opts := PCMTranscribeOptions{Language: f.Language, Generation: policy, MaxNewTokens: 96, SkipDigitalSilence: skipSilence}
				if backend == "vulkan" {
					opts.VulkanEncoder = gpu
				}
				out := speechCorpusResult{Fixture: f.Name, Language: f.Language, Path: backend, Samples: f.Samples, Reference: f.Reference, Silent: f.Reference == "", SkipDigitalSilence: skipSilence}
				start := time.Now()
				err := model.TranscribePCMWindows(ctx, reader, f.Samples, tok, opts, func(w WindowTranscript) error { out.Windows = append(out.Windows, w); return nil })
				out.InferenceNS = time.Since(start).Nanoseconds()
				if err != nil {
					out.Error = err.Error()
				}
				for _, w := range out.Windows {
					for _, s := range w.Segments {
						out.Text += " " + s.Text
					}
				}
				out.Text = strings.TrimSpace(out.Text)
				out.Edits, out.Words = speechFixtureWER(f.Reference, out.Text)
				if out.Words > 0 {
					wer := float64(out.Edits) / float64(out.Words)
					out.WER = &wer
				}
				out.EmptyOutput = out.Text == ""
				data, _ := json.Marshal(out)
				t.Log("CORPUS_RESULT " + string(data))
				for _, w := range out.Windows {
					data, _ := json.Marshal(map[string]any{"fixture": f.Name, "backend": backend, "skip_digital_silence": skipSilence, "window": w})
					t.Log("CORPUS_WINDOW " + string(data))
				}
				outcomes = append(outcomes, out)
				if err != nil {
					if !vk.VulkanReady() {
						reader.Close()
						t.Fatal("native state failure; stopping study", err)
					}
				}
			}
			for _, outcome := range outcomes {
				if outcome.Error != "" || len(outcome.Windows) != int((f.Samples+479999)/480000) {
					t.Fatal("incomplete corpus inference", f.Name, outcome.Path, outcome.Error, len(outcome.Windows))
				}
			}
			if outcomes[0].Error != outcomes[1].Error || !reflect.DeepEqual(outcomes[0].Windows, outcomes[1].Windows) {
				t.Fatal("CPU/GPU corpus mismatch", f.Name)
			}
			t.Logf("CORPUS_PARITY fixture=%s skip_digital_silence=%t identical=true quality_diagnostic_only=true source_sha256=%s", f.Name, skipSilence, f.SHA256)
			if skipSilence {
				for _, o := range outcomes {
					if o.Error != "" {
						t.Fatal("silenceskip error", o.Error)
					}
					if f.Name == "silence-5s" && (!o.EmptyOutput || len(o.Windows) != 1) {
						t.Fatal("silenceskip emittedtext")
					}
					if f.Name == "jfk-three-windows-63s" && (len(o.Windows) != 3 || len(o.Windows[2].Segments) != 0 || o.Edits != 0) {
						t.Fatal("silenttail notempty orspeechchanged")
					}
				}
			}
		}
		if err := reader.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
func TestSpeechFixturePCM(t *testing.T) {
	p := filepath.Join(t.TempDir(), "fixture.wav")
	want := []float32{-1, -.5, 0, .5, 32767. / 32768}
	writeSpeechFixturePCM(t, p, want)
	r, err := media.OpenCanonicalPCM(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	got := make([]float32, 6)
	n, err := r.ReadSamplesAt(context.Background(), got, 0)
	if n != 5 || err != io.EOF || !reflect.DeepEqual(got[:5], want) {
		t.Fatal(n, err, got)
	}
	names := map[string]bool{}
	for _, f := range mindsSpeechFixtures {
		_, hashErr := hex.DecodeString(f.SHA256)
		if f.Name == "" || names[f.Name] || (f.Language != "pt" && f.Language != "fr") || filepath.Base(f.File) != f.File || !strings.HasSuffix(f.File, "-source.wav") || hashErr != nil || len(f.SHA256) != 64 || f.Samples < 1 || len(speechFixtureWords(f.Reference)) < 1 {
			t.Fatal("invalidfixture", f)
		}
		names[f.Name] = true
	}
}
