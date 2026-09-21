package whisper

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/rcarmo/go-pherence/loader/audio/media"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

// This explicit CPU-only media A/B test consumes the immutable paired outputs
// created by TestGo264PublicMedia. It does not change backend defaults.
func TestGo264PairedSpeech(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_GO264_SPEECH") != "1" {
		t.Skip("explicit paired media ASR qualification")
	}
	if deadline, ok := t.Deadline(); !ok || time.Until(deadline) > 120*time.Second {
		t.Fatal("timeout<=120s required")
	}
	root := os.Getenv("GO_PHERENCE_GO264_PUBLIC_OUTPUT")
	if root == "" {
		t.Fatal("paired PCM directory required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 110*time.Second)
	defer cancel()
	m, tok, policy := pinnedTinySpeechModel(t, ctx)
	type result struct {
		Fixture, Variant, Backend, Text, Reference string
		Frames                                     int64
		Edits, Words                               int
		Windows                                    []WindowTranscript
	}
	var results []result
	for _, f := range mindsSpeechFixtures {
		for _, variant := range []string{"wav", "aac48k"} {
			var pair []result
			for _, backend := range []string{"ff", "go"} {
				path := filepath.Join(root, fmt.Sprintf("%s-%s-%s.wav", f.Name, variant, backend))
				r, e := media.OpenCanonicalPCM(ctx, path)
				if e != nil {
					t.Fatal(e)
				}
				if int64(r.Timeline().Samples) < f.Samples {
					t.Fatal("short PCM", path)
				}
				out := result{Fixture: f.Name, Variant: variant, Backend: backend, Reference: f.Reference, Frames: f.Samples}
				e = m.TranscribePCMWindows(ctx, r, f.Samples, tok, PCMTranscribeOptions{Language: f.Language, Generation: policy, MaxNewTokens: 96}, func(w WindowTranscript) error {
					out.Windows = append(out.Windows, w)
					for _, s := range w.Segments {
						if out.Text != "" {
							out.Text += " "
						}
						out.Text += s.Text
					}
					return nil
				})
				r.Close()
				if e != nil {
					t.Fatal(f.Name, variant, backend, e)
				}
				out.Edits, out.Words = speechFixtureWER(f.Reference, out.Text)
				pair = append(pair, out)
				results = append(results, out)
				b, _ := json.Marshal(out)
				t.Log("GO264_SPEECH " + string(b))
			}
			a, b := pair[0], pair[1]
			t.Logf("GO264_PAIR fixture=%s variant=%s equalWindows=%t deltaEdits=%d", f.Name, variant, reflect.DeepEqual(a.Windows, b.Windows), b.Edits-a.Edits)
			// Tiny baseline quality is poor on PT/FR; this gate measures frontend
			// regression only, not production ASR quality or DER.
			if b.Edits > a.Edits {
				t.Errorf("media WER regression: %s %s %d->%d edits", f.Name, variant, a.Edits, b.Edits)
			}
		}
	}
	if out := os.Getenv("GO_PHERENCE_GO264_SPEECH_REPORT"); out != "" {
		b, _ := json.MarshalIndent(results, "", "  ")
		if e := os.WriteFile(out, b, 0600); e != nil {
			t.Fatal(e)
		}
	}
}
