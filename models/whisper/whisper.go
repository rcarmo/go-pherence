package whisper

import (
	"github.com/rcarmo/go-pherence/loader/audio"
)

// Whisper is the full speech recognition pipeline.
type Whisper struct {
	Encoder *Encoder
	Decoder *Decoder
	Config  Config
}

// Transcribe takes a WAV file path and returns the transcribed text.
func (w *Whisper) Transcribe(wavPath string) (string, error) {
	// Load and preprocess audio
	samples, sampleRate, err := audio.WAV(wavPath)
	if err != nil {
		return "", err
	}

	// Resample to 16kHz if needed
	if sampleRate != 16000 {
		samples = audio.ResampleSinc(samples, sampleRate, 16000)
	}

	// Compute mel spectrogram
	melCfg := audio.MelConfig{
		SampleRate: 16000,
		FFTSize:    400,
		HopLength:  160,
		NumMels:    w.Config.NumMelBins,
		NFFTPadded: 512,
	}
	mel := audio.MelSpectrogram(samples, melCfg)
	if mel == nil {
		return "", nil
	}

	// Flatten mel to channel-first: [numMels, T]
	T := len(mel[0])
	melFlat := make([]float32, w.Config.NumMelBins*T)
	for m := 0; m < w.Config.NumMelBins; m++ {
		copy(melFlat[m*T:], mel[m])
	}

	// Encoder forward
	encoderOutput := w.Encoder.Forward(melFlat, T)
	encLen := len(encoderOutput) / w.Config.EncoderDModel

	// Initialize decoder state
	state := NewDecoderState(w.Config, encoderOutput, encLen, w.Decoder)

	// Greedy decode
	tokens := GreedyDecode(w.Decoder, state, w.Config)

	// Convert tokens to text (placeholder — needs tokenizer)
	return TokensToText(tokens), nil
}

// TranscribeFromSamples takes pre-loaded 16kHz mono float32 samples.
func (w *Whisper) TranscribeFromSamples(samples []float32) (string, error) {
	melCfg := audio.MelConfig{
		SampleRate: 16000,
		FFTSize:    400,
		HopLength:  160,
		NumMels:    w.Config.NumMelBins,
		NFFTPadded: 512,
	}
	mel := audio.MelSpectrogram(samples, melCfg)
	if mel == nil {
		return "", nil
	}

	T := len(mel[0])
	melFlat := make([]float32, w.Config.NumMelBins*T)
	for m := 0; m < w.Config.NumMelBins; m++ {
		copy(melFlat[m*T:], mel[m])
	}

	encoderOutput := w.Encoder.Forward(melFlat, T)
	encLen := len(encoderOutput) / w.Config.EncoderDModel

	state := NewDecoderState(w.Config, encoderOutput, encLen, w.Decoder)
	tokens := GreedyDecode(w.Decoder, state, w.Config)

	return TokensToText(tokens), nil
}
