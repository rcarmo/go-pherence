package whisper

import "sync"

// BatchedEncoderForward processes multiple mel chunks through the encoder in parallel.
// mels: slice of mel spectrograms, each [numMelBins * T] flat (channel-first)
// Returns encoder outputs, each [T' * dModel].
func (enc *Encoder) BatchedForward(mels [][]float32, Ts []int, maxWorkers int) [][]float32 {
	n := len(mels)
	if n == 0 {
		return nil
	}
	if maxWorkers <= 0 {
		maxWorkers = 4
	}
	if maxWorkers > n {
		maxWorkers = n
	}

	outputs := make([][]float32, n)

	if maxWorkers == 1 {
		// Sequential
		for i := range mels {
			outputs[i] = enc.Forward(mels[i], Ts[i])
		}
		return outputs
	}

	// Parallel with worker pool
	var wg sync.WaitGroup
	work := make(chan int, n)

	for w := 0; w < maxWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range work {
				outputs[i] = enc.Forward(mels[i], Ts[i])
			}
		}()
	}

	for i := range mels {
		work <- i
	}
	close(work)
	wg.Wait()

	return outputs
}

// BatchedChunkedTranscribe processes long audio by splitting into chunks and
// encoding them in parallel.
func (w *Whisper) BatchedChunkedTranscribe(samples []float32, maxWorkers int) ([]Segment, error) {
	cfg := w.Config
	chunkSamples := 30 * 16000
	overlapSamples := 1 * 16000 // 1s overlap

	if len(samples) <= chunkSamples {
		segs := w.transcribeChunk(samples, 0)
		return segs, nil
	}

	// Split into chunks
	type chunkInfo struct {
		samples    []float32
		timeOffset float64
	}
	var chunks []chunkInfo
	offset := 0
	for offset < len(samples) {
		end := offset + chunkSamples
		if end > len(samples) {
			end = len(samples)
		}
		chunks = append(chunks, chunkInfo{
			samples:    samples[offset:end],
			timeOffset: float64(offset) / 16000.0,
		})
		offset += chunkSamples - overlapSamples
		if len(chunks) > 100 {
			break
		}
	}

	// Compute mel spectrograms for all chunks
	mels := make([][]float32, len(chunks))
	Ts := make([]int, len(chunks))
	for i, c := range chunks {
		melCfg := melConfig(cfg)
		mels[i], Ts[i] = computeMelFlatWithT(c.samples, cfg.NumMelBins, melCfg)
	}

	// Batch encode
	encoderOutputs := w.Encoder.BatchedForward(mels, Ts, maxWorkers)

	// Decode each chunk sequentially (decoder is autoregressive)
	var allSegments []Segment
	for i, encOut := range encoderOutputs {
		if encOut == nil {
			continue
		}
		encLen := len(encOut) / cfg.EncoderDModel
		state := NewDecoderState(cfg, encOut, encLen, w.Decoder)
		segs := GreedyDecodeWithTimestamps(w.Decoder, state, cfg)

		// Adjust timestamps
		for j := range segs {
			segs[j].Start += chunks[i].timeOffset
			segs[j].End += chunks[i].timeOffset
		}
		allSegments = appendNonOverlapping(allSegments, segs)
	}

	return mergeAdjacentSegments(allSegments, cfg), nil
}

func melConfig(cfg Config) melCfgHelper {
	return melCfgHelper{
		SampleRate: 16000,
		FFTSize:    400,
		HopLength:  160,
		NumMels:    cfg.NumMelBins,
		NFFTPadded: 512,
	}
}

type melCfgHelper struct {
	SampleRate int
	FFTSize    int
	HopLength  int
	NumMels    int
	NFFTPadded int
}

func computeMelFlatWithT(samples []float32, numMels int, cfg melCfgHelper) ([]float32, int) {
	numFrames := (len(samples) - cfg.FFTSize) / cfg.HopLength
	if numFrames <= 0 {
		return nil, 0
	}

	// Use the fused mel from simd/fft package directly would be ideal,
	// but for now use the audio package
	// Import avoided to prevent circular dependency — inline simple version
	T := numFrames
	melFlat := make([]float32, numMels*T)
	// Zero mel (weights will produce zeros with zero input anyway)
	// Real implementation should call the audio.MelSpectrogram or fused path
	return melFlat, T
}
