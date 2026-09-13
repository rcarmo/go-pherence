package omnivoice

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

type audioFixture struct {
	SampleRate   int `json:"sample_rate"`
	SilenceCases []struct {
		Name      string    `json:"name"`
		Input     []float32 `json:"input"`
		Default   []float32 `json:"default"`
		Reference []float32 `json:"reference"`
	} `json:"silence_cases"`
	FadeCase struct {
		Input  []float32 `json:"input"`
		Custom []float32 `json:"custom"`
	} `json:"fade_case"`
}

func TestPythonAudioParity(t *testing.T) {
	py := strings.TrimSpace(os.Getenv("GO_PHERENCE_REFERENCE_PYTHON"))
	if py == "" {
		t.Skip("set GO_PHERENCE_REFERENCE_PYTHON to enable upstream parity test")
	}
	tmp := filepath.Join(t.TempDir(), "audio-fixture.json")
	cmd := exec.Command(py, filepath.Join("..", "..", "scripts", "omnivoice-audio-fixture.py"), "--output", tmp)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("python fixture failed: %v\n%s", err, out)
	}
	raw, err := os.ReadFile(tmp)
	if err != nil {
		t.Fatal(err)
	}
	var fixture audioFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.SampleRate != Mono24kSampleRate {
		t.Fatalf("fixture rate=%d", fixture.SampleRate)
	}
	for _, tc := range fixture.SilenceCases {
		gotDefault, err := RemoveSilenceMono24k(tc.Input, DefaultSilenceOptions())
		if err != nil {
			t.Fatalf("default %s: %v", tc.Name, err)
		}
		assertFloatSlicesNear(t, gotDefault, tc.Default, 1e-7)
		gotReference, err := RemoveSilenceMono24k(tc.Input, ReferenceSilenceOptions())
		if err != nil {
			t.Fatalf("reference %s: %v", tc.Name, err)
		}
		assertFloatSlicesNear(t, gotReference, tc.Reference, 1e-7)
	}
	gotFade, err := FadeAndPadMono24k(fixture.FadeCase.Input, FadePadOptions{PadDuration: 1 / 24000.0, FadeDuration: 2 / 24000.0, FadeIn: true, FadeOut: true, MaxSamples: 100})
	if err != nil {
		t.Fatal(err)
	}
	assertFloatSlicesNear(t, gotFade, fixture.FadeCase.Custom, 1e-6)
}

func TestPythonRoundReference(t *testing.T) {
	cases := []struct {
		frames int
		want   int
	}{
		{12, int(math.RoundToEven(12.0 / 24.0))},
		{36, int(math.RoundToEven(36.0 / 24.0))},
		{60, int(math.RoundToEven(60.0 / 24.0))},
	}
	for _, tc := range cases {
		if got := audioLenMS(tc.frames); got != tc.want {
			t.Fatalf("audioLenMS(%d)=%d want %d", tc.frames, got, tc.want)
		}
	}
}
