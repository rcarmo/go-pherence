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
		"cos.approx.ftz.f32",
		"sin.approx.ftz.f32",
		"lg2.approx.ftz.f32",
		"st.global.f32",
	} {
		if !strings.Contains(FFTPTX, want) {
			t.Fatalf("FFTPTX missing %q", want)
		}
	}
	if strings.Contains(FFTPTX, "TODO") {
		t.Fatalf("FFTPTX still advertises TODO-only body")
	}
}

func TestMelSpectrogramPTXTracksWhisperLogContract(t *testing.T) {
	if !strings.Contains(FFTPTX, "log10(max(mel[m], 1e-10))") {
		t.Fatalf("FFTPTX should document mel log10 clamp contract")
	}
	if !strings.Contains(FFTPTX, "mad.lo.u32 %out_idx, %m, %num_frames_u, %frame") {
		t.Fatalf("FFTPTX should write mel-major output layout")
	}
}
