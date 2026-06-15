package ptx

import (
	"strings"
	"testing"
)

func TestMelSpectrogramPTXContract(t *testing.T) {
	for _, want := range []string{
		".visible .entry mel_spectrogram",
		".param .u64 out_ptr",
		".param .u64 audio_ptr",
		".param .u64 window_ptr",
		".param .u64 mel_filters_ptr",
		".param .u32 num_frames",
		".param .u32 fft_size",
		".param .u32 hop_length",
		".param .u32 num_mels",
		".param .u32 num_bins",
		"sh_re[512]",
		"sh_im[512]",
	} {
		if !strings.Contains(FFTPTX, want) {
			t.Fatalf("FFTPTX missing %q", want)
		}
	}
}

func TestMelSpectrogramPTXTracksWhisperLogContract(t *testing.T) {
	if !strings.Contains(FFTPTX, "log(max(mel[m], 1e-10))") {
		t.Fatalf("FFTPTX should document mel log clamp contract")
	}
	if !strings.Contains(FFTPTX, "out[m * num_frames + frame]") {
		t.Fatalf("FFTPTX should document mel-major output layout")
	}
}
