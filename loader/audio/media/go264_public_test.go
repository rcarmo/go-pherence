package media

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// Public input licence: PolyAI MINDS-14 CC-BY-4.0, revision
// 40ce77cb32a384e4d50a568e1ec39ac804019d33. No private recordings.
var go264SpeechFixtures = []struct {
	Name, File, Hash, Language, Reference string
	Frames                                int64
}{
	{"minds-pt-0", "pt-real-0-source.wav", "fc084982ad50c6ea6cf066f08374b9b3aaa628d9a9accb167be5ae9376dbd275", "pt", "Bom dia estou a ligar porque precisava de informações sobre como é que eu posso depositar dinheiro na minha conta", 144726},
	{"minds-pt-1", "pt-real-1-source.wav", "aacee91914f902b0949425ee29ac984c48a34df55e5c195e65e0c8ff66977484", "pt", "Como faço para transferir dinheiro para a minha conta", 75094},
	{"minds-fr-0", "fr-real-0-source.wav", "84defdc828ef59cec10364354fbc284bc2cc683fdd4a5edd5863b7bb2c6123a8", "fr", "je souhaite changer mon adresse", 60074},
}

func TestGo264PublicMedia(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_GO264_PUBLIC") != "1" {
		t.Skip("explicit public fixture comparison")
	}
	root := os.Getenv("GO_PHERENCE_MINDS_FIXTURE_DIR")
	if root == "" {
		t.Fatal("MINDS source directory required")
	}
	ctx := context.Background()
	g, e := NewGo264(Go264Config{})
	if e != nil {
		t.Fatal(e)
	}
	ff, e := NewFFmpeg(Config{FFmpegPath: "ffmpeg", FFprobePath: "ffprobe"})
	if e != nil {
		t.Fatal(e)
	}
	for _, f := range go264SpeechFixtures {
		t.Run(f.Name, func(t *testing.T) {
			src := filepath.Join(root, f.File)
			b, e := os.ReadFile(src)
			if e != nil {
				t.Fatal(e)
			}
			sum := sha256.Sum256(b)
			if hex.EncodeToString(sum[:]) != f.Hash {
				t.Fatal("fixture checksum")
			}
			if _, e := g.Probe(ctx, src); !errors.Is(e, ErrUnsupportedInput) {
				t.Fatal("original mu-law must reject", e)
			}
			dir := t.TempDir()
			pcmSource := filepath.Join(dir, "source-pcm8k.wav")
			if log, e := exec.Command("ffmpeg", "-v", "error", "-i", src, "-c:a", "pcm_s16le", pcmSource).CombinedOutput(); e != nil {
				t.Fatal(e, string(log))
			}
			for _, variant := range []string{"wav", "aac48k"} {
				t.Run(variant, func(t *testing.T) {
					input := pcmSource
					if variant == "aac48k" {
						input = filepath.Join(dir, "input.m4a")
						if log, e := exec.Command("ffmpeg", "-v", "error", "-i", src, "-ar", "48000", "-c:a", "aac", "-movie_timescale", "48000", input).CombinedOutput(); e != nil {
							t.Fatal(e, string(log))
						}
					}
					goOut, e := g.DecodeToFile(ctx, input, filepath.Join(dir, variant+"-go.wav"))
					if e != nil {
						t.Fatal(e)
					}
					ffOut, e := ff.DecodeToFile(ctx, input, filepath.Join(dir, variant+"-ff.wav"))
					if e != nil {
						t.Fatal(e)
					}
					if int64(goOut.Timeline.Samples) != f.Frames {
						t.Fatal("go frame extent", goOut.Timeline, f.Frames)
					}
					if int64(ffOut.Timeline.Samples) < f.Frames {
						t.Fatal("reference shorter than media")
					}
					read := func(path string, n int64) []float32 {
						r, e := OpenCanonicalPCM(ctx, path)
						if e != nil {
							t.Fatal(e)
						}
						defer r.Close()
						buf := make([]float32, n)
						got, e := r.ReadSamplesAt(ctx, buf, 0)
						if e != nil || got != len(buf) {
							t.Fatal(got, e)
						}
						return buf
					}
					got, want := read(goOut.Path, f.Frames), read(ffOut.Path, f.Frames)
					maxAbs, errEnergy, energy := 0.0, 0.0, 0.0
					different := 0
					for i, v := range got {
						delta := float64(v - want[i])
						maxAbs = math.Max(maxAbs, math.Abs(delta))
						errEnergy += delta * delta
						energy += float64(want[i]) * float64(want[i])
						if v != want[i] {
							different++
						}
					}
					snr := 999.0
					if errEnergy != 0 {
						snr = 10 * math.Log10(energy/errEnergy)
					}
					metrics := map[string]any{"fixture": f.Name, "variant": variant, "source_sha256": f.Hash, "language": f.Language, "reference": f.Reference, "frames": f.Frames, "ffmpeg_frames": ffOut.Timeline.Samples, "different_samples": different, "snr_db": snr, "max_abs": maxAbs, "pcm_diagnostic_not_wer": true}
					j, _ := json.Marshal(metrics)
					t.Log("GO264_MEDIA " + string(j))
					if math.IsNaN(snr) || math.IsInf(maxAbs, 0) || snr < 20 || maxAbs > 0.15 {
						t.Fatalf("gross PCM regression: SNR%g maxabs%g", snr, maxAbs)
					}
					if out := os.Getenv("GO_PHERENCE_GO264_PUBLIC_OUTPUT"); out != "" {
						if e = os.MkdirAll(out, 0700); e != nil {
							t.Fatal(e)
						}
						for name, path := range map[string]string{"go": goOut.Path, "ff": ffOut.Path} {
							data, e := os.ReadFile(path)
							if e != nil {
								t.Fatal(e)
							}
							if e = os.WriteFile(filepath.Join(out, fmt.Sprintf("%s-%s-%s.wav", f.Name, variant, name)), data, 0600); e != nil {
								t.Fatal(e)
							}
						}
					}
				})
			}
		})
	}
}
