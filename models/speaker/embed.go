package speaker

import (
	"github.com/rcarmo/go-pherence/loader/audio"
)

// ExtractEmbeddings computes speaker embeddings for each VAD segment.
// samples: 16kHz mono float32 audio.
// segments: voice-active segments.
// ecapa: speaker embedding model.
// Returns one embedding per segment.
func ExtractEmbeddings(samples []float32, sampleRate int, segments []VADSegment, ecapa *ECAPA) [][]float32 {
	if ecapa == nil || len(segments) == 0 {
		return nil
	}

	cfg := ecapa.cfg
	embeddings := make([][]float32, len(segments))

	for i, seg := range segments {
		// Extract segment audio
		startSample := int(seg.Start * float64(sampleRate))
		endSample := int(seg.End * float64(sampleRate))
		if startSample < 0 {
			startSample = 0
		}
		if endSample > len(samples) {
			endSample = len(samples)
		}
		if startSample >= endSample {
			embeddings[i] = make([]float32, cfg.EmbedDim)
			continue
		}

		segAudio := samples[startSample:endSample]

		// Compute mel spectrogram for segment
		melCfg := audio.MelConfig{
			SampleRate: sampleRate,
			FFTSize:    400,
			HopLength:  160,
			NumMels:    cfg.NumMels,
			NFFTPadded: 512,
		}
		mel := audio.MelSpectrogram(segAudio, melCfg)
		if mel == nil || len(mel[0]) == 0 {
			embeddings[i] = make([]float32, cfg.EmbedDim)
			continue
		}

		// Flatten mel to channel-first
		T := len(mel[0])
		melFlat := make([]float32, cfg.NumMels*T)
		for m := 0; m < cfg.NumMels; m++ {
			copy(melFlat[m*T:], mel[m])
		}

		// Compute embedding
		embeddings[i] = ecapa.Embed(melFlat, T)
	}

	return embeddings
}

// Diarize performs full speaker diarization on audio samples.
// Returns diarization labels for each detected speech segment.
func Diarize(samples []float32, sampleRate int, ecapa *ECAPA, threshold float32) ([]VADSegment, []int) {
	// Step 1: VAD
	segments := EnergyVAD(samples, sampleRate, 25, 10, 0)
	if len(segments) == 0 {
		return nil, nil
	}

	// Step 2: Extract speaker embeddings
	embeddings := ExtractEmbeddings(samples, sampleRate, segments, ecapa)
	if len(embeddings) == 0 {
		return segments, make([]int, len(segments))
	}

	// Step 3: Cluster
	if threshold <= 0 {
		threshold = 0.7
	}
	labels := AgglomerativeCluster(embeddings, threshold)

	return segments, labels
}
