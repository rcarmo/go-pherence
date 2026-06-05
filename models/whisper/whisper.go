package whisper

import (
	"fmt"
	"os"
	"time"

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

	encStart := time.Now()
	encoderOutput := w.Encoder.Forward(melFlat, T)
	encLen := len(encoderOutput) / w.Config.EncoderDModel

	state := NewDecoderState(w.Config, encoderOutput, encLen, w.Decoder)
	stateDone := time.Now()
	tokens := GreedyDecode(w.Decoder, state, w.Config)
	if os.Getenv("WHISPER_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "[timing] encoder+xkv=%.1fs decode=%.1fs tokens=%d\n",
			stateDone.Sub(encStart).Seconds(), time.Since(stateDone).Seconds(), len(tokens))
		if useInt8 {
			fmt.Fprintf(os.Stderr, "[int8] quant=%.1fs pack=%.1fs gemm=%.1fs dequant=%.1fs\n",
				float64(i8QuantNs)/1e9, float64(i8PackNs)/1e9, float64(i8GemmNs)/1e9, float64(i8DeqNs)/1e9)
			fmt.Fprintf(os.Stderr, "[dec] self=%.1fs cross=%.1fs mlp=%.1fs lmhead=%.1fs\n",
				float64(decSelfNs)/1e9, float64(decCrossNs)/1e9, float64(decMlpNs)/1e9, float64(decLmNs)/1e9)
		}
	}

	return TokensToText(tokens), nil
}
