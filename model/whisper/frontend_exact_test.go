package whisper

import (
	"github.com/rcarmo/go-pherence/loader/audio"
	"testing"
)

func TestExactFrontendUses128BandContract(t *testing.T) {
	// The opt-in legacy GPU mel flag must not change the checked frontend's math.
	t.Setenv("GO_PHERENCE_WHISPER_GPU_MEL", "1")
	samples := make([]float32, 1001)
	samples[0] = 0.75
	samples[199] = -0.25
	want, n, err := audio.WhisperLogMel(samples, 128)
	if err != nil {
		t.Fatal(err)
	}
	got, frames, err := MelFlatFromSamplesChecked(samples, LargeV3Turbo())
	if err != nil {
		t.Fatal(err)
	}
	if frames != n || frames != 6 || len(got) != len(want) {
		t.Fatalf("shape frames=%d len=%d", frames, len(got))
	}
	for i, v := range got {
		if v != want[i] {
			t.Fatalf("different feature %d", i)
		}
	}
	if _, _, err := MelFlatFromSamplesChecked(samples, Config{NumMelBins: 64}); err == nil {
		t.Fatal("unsupported bands accepted")
	}
}
