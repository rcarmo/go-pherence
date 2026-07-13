package mosstranscribe

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/models/whisper"
)

const (
	AudioSampleRate   = 16000
	AudioChunkSamples = 30 * AudioSampleRate
	WhisperHopSamples = 160
	WhisperConvStride = 2
	AudioTokenStride  = WhisperHopSamples * WhisperConvStride * AudioMergeFrames
)

// AudioChunk is one independently padded 30-second Whisper input. TokenLength
// records only real source audio; Samples always contains exactly 480,000 rows.
type AudioChunk struct {
	Samples     []float32
	TokenLength int
}

// ReadAudioWAV decodes mono WAV input and resamples it to the checkpoint's
// required 16 kHz using the native windowed-sinc resampler.
func ReadAudioWAV(path string) ([]float32, error) {
	samples, sampleRate, err := audio.WAV(path)
	if err != nil {
		return nil, err
	}
	if sampleRate <= 0 {
		return nil, fmt.Errorf("MOSS audio: invalid sample rate %d", sampleRate)
	}
	if sampleRate != AudioSampleRate {
		samples = audio.ResampleSinc(samples, sampleRate, AudioSampleRate)
	}
	return samples, nil
}

// ChunkAudio applies the upstream processor's non-overlapping 30-second split,
// right zero-padding, and exact ceil(numSamples/1280) token accounting.
func ChunkAudio(samples []float32) ([]AudioChunk, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("MOSS audio: empty waveform")
	}
	chunks := make([]AudioChunk, 0, (len(samples)+AudioChunkSamples-1)/AudioChunkSamples)
	for start := 0; start < len(samples); start += AudioChunkSamples {
		end := start + AudioChunkSamples
		if end > len(samples) {
			end = len(samples)
		}
		length := end - start
		chunk := AudioChunk{
			Samples:     make([]float32, AudioChunkSamples),
			TokenLength: (length-1)/AudioTokenStride + 1,
		}
		copy(chunk.Samples, samples[start:end])
		chunks = append(chunks, chunk)
	}
	return chunks, nil
}

// InputFeatures computes the 80-bin Whisper log-mel matrix in channel-major
// [80,3000] form. It delegates FFT/filter arithmetic to the shared native SIMD
// frontend; Transformers numerical parity is enforced separately by fixtures.
func (chunk AudioChunk) InputFeatures(cfg whisper.Config) ([]float32, error) {
	if len(chunk.Samples) != AudioChunkSamples || chunk.TokenLength <= 0 || chunk.TokenLength > AudioChunkSamples/AudioTokenStride {
		return nil, fmt.Errorf("MOSS audio: malformed chunk samples=%d tokens=%d", len(chunk.Samples), chunk.TokenLength)
	}
	// Hugging Face centers the STFT with n_fft/2 reflect padding and drops the
	// final frame. The shared frontend's frame loop omits that final frame, so
	// explicit reflection produces the required 3,000 columns.
	centered := reflectPad(chunk.Samples, 200)
	features, frames := whisper.MelFlatFromSamples(centered, cfg)
	if frames != cfg.MaxLength || len(features) != cfg.NumMelBins*cfg.MaxLength {
		return nil, fmt.Errorf("MOSS audio: frontend shape [%d,%d], want [%d,%d]", cfg.NumMelBins, frames, cfg.NumMelBins, cfg.MaxLength)
	}
	return features, nil
}

// AudioTokenLength is the upstream processor formula exposed for prompt planning.
func reflectPad(samples []float32, padding int) []float32 {
	out := make([]float32, len(samples)+2*padding)
	copy(out[padding:], samples)
	for i := 0; i < padding; i++ {
		out[padding-1-i] = samples[i+1]
		out[padding+len(samples)+i] = samples[len(samples)-2-i]
	}
	return out
}

func AudioTokenLength(numSamples int) int {
	if numSamples <= 0 {
		return 0
	}
	return int(math.Ceil(float64(numSamples) / AudioTokenStride))
}
