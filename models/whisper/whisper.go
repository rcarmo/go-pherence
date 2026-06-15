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
	return w.TranscribePrompt(wavPath, TokenEnglish, TokenTranscribe)
}

// TranscribePrompt takes a WAV file path and decodes with an explicit Whisper
// language/task prompt.
func (w *Whisper) TranscribePrompt(wavPath string, languageToken, taskToken int) (string, error) {
	samples, sampleRate, err := audio.WAV(wavPath)
	if err != nil {
		return "", err
	}
	if sampleRate != 16000 {
		samples = audio.ResampleSinc(samples, sampleRate, 16000)
	}
	return w.TranscribeFromSamplesPrompt(samples, languageToken, taskToken)
}

// TranscribeFromSamples takes pre-loaded 16kHz mono float32 samples.
func (w *Whisper) TranscribeFromSamples(samples []float32) (string, error) {
	return w.TranscribeFromSamplesPrompt(samples, TokenEnglish, TokenTranscribe)
}

// TranscribeFromSamplesPrompt takes pre-loaded 16kHz mono float32 samples and
// decodes with an explicit Whisper language/task prompt.
func (w *Whisper) TranscribeFromSamplesPrompt(samples []float32, languageToken, taskToken int) (string, error) {
	melFlat, T := MelFlatFromSamples(samples, w.Config)
	if len(melFlat) == 0 || T == 0 {
		return "", nil
	}

	encStart := time.Now()
	encoderOutput := w.Encoder.Forward(melFlat, T)
	encLen := len(encoderOutput) / w.Config.EncoderDModel

	state := NewDecoderState(w.Config, encoderOutput, encLen, w.Decoder)
	stateDone := time.Now()
	tokens := GreedyDecodePrompt(w.Decoder, state, w.Config, languageToken, taskToken)
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
