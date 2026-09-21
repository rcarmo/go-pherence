package whisper

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio/media"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// TestWhisperTinyJFKWordAlignment is an opt-in trained-model check. The pinned
// model and fixture are intentionally not repository test dependencies.
type multilingualWordOracle struct {
	name, language, file, hash string
	tokens                     []int
	words                      []WordTiming
}

// TestWhisperTinyMultilingualWordAlignment is an opt-in cross-check against
// token/word times generated independently by Transformers 4.57.1 and PyTorch
// 2.14.0+cpu with eager attention. These three immutable MINDS clips establish
// Portuguese/French alignment parity only; Tiny's recognition errors mean this
// is not broad multilingual WER qualification.
func TestWhisperTinyMultilingualWordAlignment(t *testing.T) {
	modelDir := os.Getenv("WHISPER_TINY_MODEL_DIR")
	root := os.Getenv("WHISPER_MINDS_FIXTURE_DIR")
	if modelDir == "" || root == "" {
		t.Skip("set WHISPER_TINY_MODEL_DIR and WHISPER_MINDS_FIXTURE_DIR")
	}
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	tok, err := LoadTokenizer(filepath.Join(modelDir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	modelJSON, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	generationJSON, err := os.ReadFile(filepath.Join(modelDir, "generation_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.Open(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	model, generation, err := LoadConfiguredModelSourceChecked(ctx, source, modelJSON, generationJSON, tok)
	if err != nil {
		t.Fatal(err)
	}
	oracles := []multilingualWordOracle{
		{"minds-pt-0", "pt", "pt-real-0-source.wav", "fc084982ad50c6ea6cf066f08374b9b3aaa628d9a9accb167be5ae9376dbd275", []int{19812, 6801, 11, 17660, 11742, 1573, 1515, 631, 3003, 39915, 368, 42542, 5473, 277, 631, 277, 9230, 23257, 2431, 385, 594, 265, 303, 13}, []WordTiming{
			{Word: "Bom", Start: 0, End: 1.62, TokenStart: 0, TokenEnd: 1}, {Word: "dia,", Start: 1.62, End: 1.78, TokenStart: 1, TokenEnd: 3}, {Word: "estou", Start: 1.78, End: 2.10, TokenStart: 3, TokenEnd: 4}, {Word: "ligado", Start: 2.10, End: 2.50, TokenStart: 4, TokenEnd: 6}, {Word: "por", Start: 2.50, End: 2.68, TokenStart: 6, TokenEnd: 7}, {Word: "que", Start: 2.68, End: 2.78, TokenStart: 7, TokenEnd: 8}, {Word: "os", Start: 2.78, End: 2.88, TokenStart: 8, TokenEnd: 9}, {Word: "dados", Start: 2.88, End: 3.16, TokenStart: 9, TokenEnd: 10}, {Word: "de", Start: 3.16, End: 3.50, TokenStart: 10, TokenEnd: 11}, {Word: "informações", Start: 3.50, End: 4.08, TokenStart: 11, TokenEnd: 12}, {Word: "sobre", Start: 4.08, End: 4.30, TokenStart: 12, TokenEnd: 13}, {Word: "o", Start: 4.30, End: 4.30, TokenStart: 13, TokenEnd: 14}, {Word: "que", Start: 4.30, End: 4.38, TokenStart: 14, TokenEnd: 15}, {Word: "o", Start: 4.38, End: 4.46, TokenStart: 15, TokenEnd: 16}, {Word: "meu", Start: 4.46, End: 4.96, TokenStart: 16, TokenEnd: 17}, {Word: "corpo", Start: 4.96, End: 5.70, TokenStart: 17, TokenEnd: 18}, {Word: "não", Start: 5.70, End: 5.78, TokenStart: 18, TokenEnd: 19}, {Word: "me", Start: 5.78, End: 6.06, TokenStart: 19, TokenEnd: 20}, {Word: "arreve.", Start: 6.06, End: 6.64, TokenStart: 20, TokenEnd: 24},
		}},
		{"minds-pt-1", "pt", "pt-real-1-source.wav", "aacee91914f902b0949425ee29ac984c48a34df55e5c195e65e0c8ff66977484", []int{2432, 283, 296, 79, 418, 1145, 5779, 327, 716, 347, 3206, 11720, 24001, 13}, []WordTiming{
			{Word: "Com", Start: 0, End: 1.44, TokenStart: 0, TokenEnd: 1}, {Word: "faspore", Start: 1.44, End: 2.02, TokenStart: 1, TokenEnd: 5}, {Word: "transfridneir", Start: 2.02, End: 2.88, TokenStart: 5, TokenEnd: 10}, {Word: "pra", Start: 2.88, End: 3.12, TokenStart: 10, TokenEnd: 11}, {Word: "minha", Start: 3.12, End: 3.64, TokenStart: 11, TokenEnd: 12}, {Word: "conta.", Start: 3.64, End: 4.68, TokenStart: 12, TokenEnd: 14},
		}},
		{"minds-fr-0", "fr", "fr-real-0-source.wav", "84defdc828ef59cec10364354fbc284bc2cc683fdd4a5edd5863b7bb2c6123a8", []int{2588, 9369, 22822, 1108, 614, 22603, 13}, []WordTiming{
			{Word: "Je", Start: 0, End: 1.38, TokenStart: 0, TokenEnd: 1}, {Word: "vais", Start: 1.38, End: 1.74, TokenStart: 1, TokenEnd: 2}, {Word: "changer", Start: 1.74, End: 2.04, TokenStart: 2, TokenEnd: 3}, {Word: "mon", Start: 2.04, End: 2.26, TokenStart: 3, TokenEnd: 4}, {Word: "adresse.", Start: 2.26, End: 3.16, TokenStart: 4, TokenEnd: 7},
		}},
	}
	for _, oracle := range oracles {
		t.Run(oracle.name, func(t *testing.T) {
			src := filepath.Join(root, oracle.file)
			pinnedSpeechFile(t, src, oracle.hash)
			wav := filepath.Join(t.TempDir(), oracle.name+".wav")
			if output, err := exec.CommandContext(ctx, ffmpeg, "-nostdin", "-v", "error", "-i", src, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav).CombinedOutput(); err != nil {
				t.Fatal(err, string(output))
			}
			reader, err := media.OpenCanonicalPCM(ctx, wav)
			if err != nil {
				t.Fatal(err)
			}
			defer reader.Close()
			pcm := make([]float32, reader.Timeline().Samples)
			if n, err := reader.ReadSamplesAt(ctx, pcm, 0); err != nil || n != len(pcm) {
				t.Fatal(n, err)
			}
			padded := make([]float32, model.Config.MaxLength*160)
			copy(padded, pcm)
			mel, frames, err := MelFlatFromSamplesCheckedContext(ctx, padded, model.Config)
			if err != nil {
				t.Fatal(err)
			}
			encoderOutput := model.Encoder.Forward(mel, frames)
			state := NewDecoderState(model.Config, encoderOutput, (frames+1)/2, model.Decoder)
			got, err := AlignWordsChecked(ctx, model.Decoder, state, tok, generation, oracle.language, oracle.tokens, (len(pcm)+159)/160)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(oracle.words) {
				t.Fatalf("words=%+v want=%+v", got, oracle.words)
			}
			for i := range got {
				want := oracle.words[i]
				if got[i].Word != want.Word || got[i].TokenStart != want.TokenStart || got[i].TokenEnd != want.TokenEnd || got[i].Start-want.Start < -0.02 || got[i].Start-want.Start > 0.02 || got[i].End-want.End < -0.02 || got[i].End-want.End > 0.02 {
					t.Fatalf("word %d=%+v want %+v (+/-0.02s)", i, got[i], want)
				}
			}
			t.Logf("WHISPER_MULTILINGUAL_WORD_ALIGNMENT fixture=%s words=%d first=%.2f last=%.2f", oracle.name, len(got), got[0].Start, got[len(got)-1].End)
		})
	}
}

func TestWhisperTinyJFKWordAlignment(t *testing.T) {
	modelDir := os.Getenv("WHISPER_TINY_MODEL_DIR")
	wav := os.Getenv("WHISPER_JFK_WAV")
	if modelDir == "" || wav == "" {
		t.Skip("set WHISPER_TINY_MODEL_DIR and WHISPER_JFK_WAV")
	}
	for _, name := range []string{"config.json", "generation_config.json", "tokenizer.json", "model.safetensors"} {
		if _, err := os.Stat(filepath.Join(modelDir, name)); err != nil {
			t.Skip(err)
		}
	}
	if _, err := os.Stat(wav); err != nil {
		t.Skip(err)
	}

	tok, err := LoadTokenizer(filepath.Join(modelDir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	modelJSON, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	generationJSON, err := os.ReadFile(filepath.Join(modelDir, "generation_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.Open(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	model, generation, err := LoadConfiguredModelSourceChecked(context.Background(), source, modelJSON, generationJSON, tok)
	if err != nil {
		t.Fatal(err)
	}
	reader, err := media.OpenCanonicalPCM(context.Background(), wav)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	pcm := make([]float32, reader.Timeline().Samples)
	if n, err := reader.ReadSamplesAt(context.Background(), pcm, 0); err != nil || n != len(pcm) {
		t.Fatal(n, err)
	}
	padded := make([]float32, model.Config.MaxLength*160)
	copy(padded, pcm)
	mel, frames, err := MelFlatFromSamplesCheckedContext(context.Background(), padded, model.Config)
	if err != nil {
		t.Fatal(err)
	}
	encoderOutput := model.Encoder.Forward(mel, frames)
	state := NewDecoderState(model.Config, encoderOutput, (frames+1)/2, model.Decoder)

	// Pinned tokenizer IDs for the checked JFK transcript. The aligner accepts
	// generated text tokens only; EOT and prompt/control tokens are excluded.
	tokens := []int{400, 370, 452, 7177, 6280, 1029, 406, 437, 428, 1941, 393, 360, 337, 291, 1029, 437, 291, 393, 360, 337, 428, 1941, 13}
	words, err := AlignWordsChecked(context.Background(), model.Decoder, state, tok, generation, "en", tokens, (len(pcm)+159)/160)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"And", "so", "my", "fellow", "Americans", "ask", "not", "what", "your", "country", "can", "do", "for", "you", "ask", "what", "you", "can", "do", "for", "your", "country."}
	got := make([]string, len(words))
	for i, word := range words {
		got[i] = word.Word
		if word.Start < 0 || word.End < word.Start || word.End > float64(len(pcm))/16000 || i > 0 && word.Start < words[i-1].End {
			t.Fatalf("word %d has invalid timing: %+v", i, word)
		}
	}
	wantStarts := []float64{0, 1.08, 1.34, 1.70, 2.28, 3.80, 4.70, 5.70, 5.96, 6.42, 6.72, 6.96, 7.24, 8.14, 8.62, 8.98, 9.22, 9.44, 9.70, 9.88, 10.10, 10.60}
	if !reflect.DeepEqual(got, want) || len(words) != len(wantStarts) {
		t.Fatalf("words=%+v", words)
	}
	for i := range words {
		if delta := words[i].Start - wantStarts[i]; delta < -0.02 || delta > 0.02 {
			t.Fatalf("word %d start=%.2f want %.2f (+/-0.02): %+v", i, words[i].Start, wantStarts[i], words)
		}
	}
	if delta := words[len(words)-1].End - 10.98; delta < -0.02 || delta > 0.02 {
		t.Fatalf("last word end=%.2f want 10.98 (+/-0.02): %+v", words[len(words)-1].End, words[len(words)-1])
	}
	var windows []WindowTranscript
	err = model.TranscribePCMWindows(context.Background(), reader, int64(len(pcm)), tok, PCMTranscribeOptions{Language: "en", Generation: generation, MaxNewTokens: 96, WordTimestamps: true}, func(window WindowTranscript) error {
		windows = append(windows, window)
		return nil
	})
	if err != nil || len(windows) != 1 {
		t.Fatal("integrated word alignment", len(windows), err)
	}
	var integrated []WordTiming
	integrated = append(integrated, windows[0].Words...)
	if len(integrated) != len(want) || integrated[0].Word != "And" || integrated[len(integrated)-1].Word != "country." {
		t.Fatalf("integrated windows=%+v words=%+v", windows, integrated)
	}
	integratedStarts := []float64{0, 1.08, 1.34, 1.70, 2.28, 3.80, 4.70, 5.70, 5.96, 6.42, 6.72, 6.96, 7.24, 8.24, 8.60, 8.98, 9.22, 9.42, 9.70, 9.88, 10.10, 10.56}
	for i := range integrated {
		if delta := integrated[i].Start - integratedStarts[i]; delta < -0.02 || delta > 0.02 {
			t.Fatalf("integrated word %d start=%.2f want %.2f (+/-0.02)", i, integrated[i].Start, integratedStarts[i])
		}
	}
	t.Logf("WHISPER_WORD_ALIGNMENT words=%d first=%.2f last=%.2f", len(words), words[0].Start, words[len(words)-1].End)
}
