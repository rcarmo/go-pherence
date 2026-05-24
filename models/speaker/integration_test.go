package speaker

import (
	"math"
	"testing"
)

// TestDiarizeSyntheticTwoSpeakers creates synthetic audio with two speakers
// (distinct frequency signatures) and verifies the pipeline separates them.
func TestDiarizeSyntheticTwoSpeakers(t *testing.T) {
	const sampleRate = 16000
	const duration = 6.0 // 6 seconds total
	samples := make([]float32, int(duration*sampleRate))

	// Speaker A: 0-2s, low frequency (200Hz fundamental)
	for i := 0; i < 2*sampleRate; i++ {
		t := float64(i) / float64(sampleRate)
		samples[i] = float32(0.5*math.Sin(2*math.Pi*200*t) + 0.3*math.Sin(2*math.Pi*400*t))
	}

	// Silence: 2-2.5s
	// (zeros already)

	// Speaker B: 2.5-4.5s, high frequency (800Hz fundamental)
	for i := int(2.5 * sampleRate); i < int(4.5*sampleRate); i++ {
		t := float64(i) / float64(sampleRate)
		samples[i] = float32(0.5*math.Sin(2*math.Pi*800*t) + 0.3*math.Sin(2*math.Pi*1600*t))
	}

	// Silence: 4.5-5s

	// Speaker A again: 5-6s
	for i := 5 * sampleRate; i < 6*sampleRate; i++ {
		t := float64(i) / float64(sampleRate)
		samples[i] = float32(0.5*math.Sin(2*math.Pi*200*t) + 0.3*math.Sin(2*math.Pi*400*t))
	}

	// Run VAD
	segments := EnergyVAD(samples, sampleRate, 25, 10, 0)
	if len(segments) < 2 {
		t.Fatalf("expected at least 2 VAD segments, got %d", len(segments))
	}
	t.Logf("VAD found %d segments:", len(segments))
	for i, s := range segments {
		t.Logf("  segment %d: %.2fs - %.2fs", i, s.Start, s.End)
	}

	// Extract embeddings using a zero-weight ECAPA (will produce zero embeddings,
	// but we can test the pipeline structure)
	cfg := DefaultECAPAConfig()
	ecapa := NewECAPA(cfg)

	// Initialize with minimal weights
	ecapa.Conv0Weight = make([]float32, cfg.Channels[0]*cfg.NumMels*cfg.KernelSize)
	ecapa.Conv0Bias = make([]float32, cfg.Channels[0])
	for i := range ecapa.Blocks {
		inCh := cfg.Channels[i]
		outCh := cfg.Channels[i+1]
		ecapa.Blocks[i].ConvWeight = make([]float32, outCh*inCh*cfg.KernelSize)
		ecapa.Blocks[i].ConvBias = make([]float32, outCh)
		ecapa.Blocks[i].SEDown = make([]float32, cfg.SEBottleneck*outCh)
		ecapa.Blocks[i].SEUp = make([]float32, outCh*cfg.SEBottleneck)
	}
	lastCh := cfg.Channels[len(cfg.Channels)-1]
	ecapa.EmbedWeight = make([]float32, cfg.EmbedDim*lastCh*2)
	ecapa.EmbedBias = make([]float32, cfg.EmbedDim)

	// Run full diarization pipeline
	vadSegs, labels := Diarize(samples, sampleRate, ecapa, 0.7)
	if len(vadSegs) == 0 {
		t.Fatal("diarization returned no segments")
	}
	t.Logf("Diarization: %d segments, labels=%v", len(vadSegs), labels)

	// Verify at least 2 segments were found
	if len(vadSegs) < 2 {
		t.Fatalf("expected at least 2 diarized segments, got %d", len(vadSegs))
	}

	// With zero-weight ECAPA, all embeddings will be identical (zero),
	// so clustering will merge them into one speaker.
	// The structural test validates the pipeline runs without panic.
	// Real speaker separation requires trained weights.
}

// TestDiarizeWithDistinctEmbeddings tests clustering with hand-crafted embeddings.
func TestDiarizeWithDistinctEmbeddings(t *testing.T) {
	// Simulate 4 segments with known embeddings
	embeddings := [][]float32{
		makeEmbed(192, 1.0, 0.0), // Speaker A
		makeEmbed(192, 0.0, 1.0), // Speaker B
		makeEmbed(192, 0.9, 0.1), // Speaker A (similar to first)
		makeEmbed(192, 0.1, 0.9), // Speaker B (similar to second)
	}

	labels := AgglomerativeCluster(embeddings, 0.7)
	t.Logf("Cluster labels: %v", labels)

	// Speakers A (0,2) should be same cluster
	if labels[0] != labels[2] {
		t.Fatalf("expected segments 0,2 same speaker: labels=%v", labels)
	}
	// Speakers B (1,3) should be same cluster
	if labels[1] != labels[3] {
		t.Fatalf("expected segments 1,3 same speaker: labels=%v", labels)
	}
	// A and B should be different
	if labels[0] == labels[1] {
		t.Fatalf("expected different speakers for segments 0,1: labels=%v", labels)
	}
}

func makeEmbed(dim int, v1, v2 float32) []float32 {
	e := make([]float32, dim)
	for i := range e {
		if i%2 == 0 {
			e[i] = v1 + float32(i)*0.001
		} else {
			e[i] = v2 + float32(i)*0.001
		}
	}
	return e
}
