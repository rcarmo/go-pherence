package speaker

// ExtractSpeechBrainEmbeddings computes SpeechBrain ECAPA embeddings for VAD
// segments using SpeechBrain-compatible Fbank preprocessing.
func ExtractSpeechBrainEmbeddings(samples []float32, sampleRate int, segments []VADSegment, ecapa *SpeechBrainECAPA) [][]float32 {
	return ExtractSpeechBrainEmbeddingsWithContext(samples, sampleRate, segments, ecapa, 0.5)
}

// ExtractSpeechBrainEmbeddingsWithContext computes SpeechBrain ECAPA embeddings
// with optional context padding around each VAD segment. Short segments produce
// unstable speaker embeddings; a small amount of surrounding audio improves
// clustering while keeping transcript cue boundaries unchanged.
func ExtractSpeechBrainEmbeddingsWithContext(samples []float32, sampleRate int, segments []VADSegment, ecapa *SpeechBrainECAPA, contextSec float64) [][]float32 {
	if ecapa == nil || len(segments) == 0 {
		return nil
	}
	embeddings := make([][]float32, len(segments))
	pad := int(contextSec * float64(sampleRate))
	if pad < 0 {
		pad = 0
	}
	for i, seg := range segments {
		startSample := int(seg.Start*float64(sampleRate)) - pad
		endSample := int(seg.End*float64(sampleRate)) + pad
		if startSample < 0 {
			startSample = 0
		}
		if endSample > len(samples) {
			endSample = len(samples)
		}
		if startSample >= endSample {
			embeddings[i] = make([]float32, 192)
			continue
		}
		features := SpeechBrainFbank(samples[startSample:endSample], sampleRate)
		if len(features) != 80 || len(features[0]) == 0 {
			embeddings[i] = make([]float32, 192)
			continue
		}
		frames := len(features[0])
		flat := make([]float32, 80*frames)
		for m := 0; m < 80; m++ {
			copy(flat[m*frames:], features[m])
		}
		embeddings[i] = ecapa.Embed(flat, frames)
	}
	return embeddings
}
