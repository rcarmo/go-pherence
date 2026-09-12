package media

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// Public pyannote-audio tutorial assets at b749285c5cdd4636b2edc7f766f1352c8dde9369.
// The model/DER runner is offline qualification only and not a runtime dependency.
func TestGo264DiarizationMedia(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_GO264_DIAR_MEDIA") != "1" {
		t.Skip("public annotated fixture opt-in")
	}
	root, out := os.Getenv("GO_PHERENCE_GO264_DIAR_FIXTURE"), os.Getenv("GO_PHERENCE_GO264_DIAR_OUTPUT")
	if root == "" || out == "" {
		t.Fatal("fixture and output paths required")
	}
	data, e := os.ReadFile(root)
	if e != nil {
		t.Fatal(e)
	}
	sum := sha256.Sum256(data)
	if hex.EncodeToString(sum[:]) != "c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e" {
		t.Fatal("fixture hash")
	}
	if e = os.MkdirAll(out, 0700); e != nil {
		t.Fatal(e)
	}
	g, e := NewGo264(Go264Config{})
	if e != nil {
		t.Fatal(e)
	}
	ff, e := NewFFmpeg(Config{FFmpegPath: "ffmpeg", FFprobePath: "ffprobe"})
	if e != nil {
		t.Fatal(e)
	}
	for _, variant := range []string{"wav", "aac48k"} {
		src := root
		if variant == "aac48k" {
			src = filepath.Join(t.TempDir(), "source.m4a")
			if log, e := exec.Command("ffmpeg", "-v", "error", "-i", root, "-ar", "48000", "-c:a", "aac", "-movie_timescale", "48000", src).CombinedOutput(); e != nil {
				t.Fatal(e, string(log))
			}
		}
		for _, backend := range []string{"ff", "go"} {
			var adapter Adapter = ff
			if backend == "go" {
				adapter = g
			}
			dst := filepath.Join(out, variant+"-"+backend+".wav")
			res, e := adapter.DecodeToFile(context.Background(), src, dst)
			if e != nil {
				t.Fatal(e)
			}
			if backend == "go" && res.Timeline.Samples != 480000 {
				t.Fatal(res.Timeline)
			}
			t.Logf("DIAR_PCM %s %s frames%d", variant, backend, res.Timeline.Samples)
		}
	}
}
