package speaker

import (
	"math"
	"testing"
)

func TestECAPAEmbedShape(t *testing.T) {
	cfg := DefaultECAPAConfig()
	ecapa := NewECAPA(cfg)

	// Initialize conv0 weights
	ecapa.Conv0Weight = make([]float32, cfg.Channels[0]*cfg.NumMels*cfg.KernelSize)
	ecapa.Conv0Bias = make([]float32, cfg.Channels[0])

	// Initialize blocks
	for i := range ecapa.Blocks {
		inCh := cfg.Channels[i]
		outCh := cfg.Channels[i+1]
		b := &ecapa.Blocks[i]
		b.ConvWeight = make([]float32, outCh*inCh*cfg.KernelSize)
		b.ConvBias = make([]float32, outCh)
		b.BNWeight = make([]float32, outCh)
		b.BNBias = make([]float32, outCh)
		b.SEDown = make([]float32, cfg.SEBottleneck*outCh)
		b.SEUp = make([]float32, outCh*cfg.SEBottleneck)
	}

	// Attentive pooling
	lastCh := cfg.Channels[len(cfg.Channels)-1]
	ecapa.PoolAttnWeight = make([]float32, cfg.AttentionDim*lastCh)
	ecapa.PoolAttnBias = make([]float32, cfg.AttentionDim)
	ecapa.PoolOutWeight = make([]float32, cfg.AttentionDim)
	ecapa.PoolOutBias = make([]float32, 1)

	// Embedding projection: input is mean+std = 2*lastCh
	ecapa.EmbedWeight = make([]float32, cfg.EmbedDim*lastCh*2)
	ecapa.EmbedBias = make([]float32, cfg.EmbedDim)

	// Test with 3 seconds of mel (300 frames)
	T := 300
	mel := make([]float32, cfg.NumMels*T)

	embedding := ecapa.Embed(mel, T)
	if len(embedding) != cfg.EmbedDim {
		t.Fatalf("embedding dim=%d want %d", len(embedding), cfg.EmbedDim)
	}
}

func TestCosineSimilarity(t *testing.T) {
	// Same vectors should have cosine similarity = 1
	a := []float32{1, 2, 3, 4}
	b := []float32{1, 2, 3, 4}
	sim := cosineSim(a, b)
	if math.Abs(float64(sim)-1.0) > 0.001 {
		t.Fatalf("same vector cosine=%f want 1.0", sim)
	}

	// Orthogonal vectors should have cosine similarity = 0
	c := []float32{1, 0, 0, 0}
	d := []float32{0, 1, 0, 0}
	sim = cosineSim(c, d)
	if math.Abs(float64(sim)) > 0.001 {
		t.Fatalf("orthogonal cosine=%f want 0.0", sim)
	}
}

func cosineSim(a, b []float32) float32 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return float32(dot / denom)
}
