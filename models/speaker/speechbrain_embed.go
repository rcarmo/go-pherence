package speaker

// ExtractSpeechBrainEmbeddings computes SpeechBrain ECAPA embeddings for VAD
// segments using SpeechBrain-compatible Fbank preprocessing.
func ExtractSpeechBrainEmbeddings(samples []float32, sampleRate int, segments []VADSegment, ecapa *SpeechBrainECAPA) [][]float32 {
	if ecapa == nil || len(segments) == 0 {
		return nil
	}
	embeddings := make([][]float32, len(segments))
	for i, seg := range segments {
		startSample := int(seg.Start * float64(sampleRate))
		endSample := int(seg.End * float64(sampleRate))
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
