package speaker

import (
	"math"
	"testing"
)

func TestEnergyVAD(t *testing.T) {
	// Create 2 seconds of audio: 0.5s silence, 1s speech, 0.5s silence
	const sampleRate = 16000
	samples := make([]float32, sampleRate*2)

	// Add speech-level energy in the middle 1 second
	for i := sampleRate / 2; i < sampleRate*3/2; i++ {
		samples[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/sampleRate))
	}

	segments := EnergyVAD(samples, sampleRate, 25, 10, 0)
	if len(segments) == 0 {
		t.Fatal("expected at least one VAD segment")
	}

	// The segment should roughly cover 0.5s to 1.5s
	seg := segments[0]
	if seg.Start < 0.3 || seg.Start > 0.7 {
		t.Fatalf("segment start=%f want ~0.5", seg.Start)
	}
	if seg.End < 1.3 || seg.End > 1.7 {
		t.Fatalf("segment end=%f want ~1.5", seg.End)
	}
}

func TestAgglomerativeCluster(t *testing.T) {
	// 4 embeddings: two pairs of similar vectors
	embeddings := [][]float32{
		{1, 0, 0},     // speaker A
		{0.9, 0.1, 0}, // speaker A (similar)
		{0, 0, 1},     // speaker B
		{0, 0.1, 0.9}, // speaker B (similar)
	}

	labels := AgglomerativeCluster(embeddings, 0.8)
	if len(labels) != 4 {
		t.Fatalf("labels length=%d want 4", len(labels))
	}

	// Embeddings 0 and 1 should be in the same cluster
	if labels[0] != labels[1] {
		t.Fatalf("expected embeddings 0,1 same cluster: %v", labels)
	}
	// Embeddings 2 and 3 should be in the same cluster
	if labels[2] != labels[3] {
		t.Fatalf("expected embeddings 2,3 same cluster: %v", labels)
	}
	// The two clusters should be different
	if labels[0] == labels[2] {
		t.Fatalf("expected two different clusters: %v", labels)
	}
}

func TestAgglomerativeClusterSingle(t *testing.T) {
	labels := AgglomerativeCluster([][]float32{{1, 2, 3}}, 0.5)
	if len(labels) != 1 || labels[0] != 0 {
		t.Fatalf("single element labels=%v", labels)
	}
}
