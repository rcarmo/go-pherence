package mosstranscribe

import "testing"

func TestAudioTokenLengthMatchesUpstreamCeil(t *testing.T) {
	for _, tc := range []struct{ samples, tokens int }{
		{0, 0}, {1, 1}, {1279, 1}, {1280, 1}, {1281, 2},
		{AudioSampleRate, 13}, {AudioChunkSamples, 375},
	} {
		if got := AudioTokenLength(tc.samples); got != tc.tokens {
			t.Fatalf("AudioTokenLength(%d)=%d want %d", tc.samples, got, tc.tokens)
		}
	}
}

func TestChunkAudioNonOverlappingPaddingAndLengths(t *testing.T) {
	samples := make([]float32, AudioChunkSamples+1281)
	for i := range samples {
		samples[i] = float32(i + 1)
	}
	chunks, err := ChunkAudio(samples)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 2 || chunks[0].TokenLength != 375 || chunks[1].TokenLength != 2 {
		t.Fatalf("unexpected chunks: count=%d lengths=%v", len(chunks), []int{chunks[0].TokenLength, chunks[1].TokenLength})
	}
	if len(chunks[0].Samples) != AudioChunkSamples || len(chunks[1].Samples) != AudioChunkSamples {
		t.Fatal("chunks are not padded to 30 seconds")
	}
	if chunks[1].Samples[0] != samples[AudioChunkSamples] || chunks[1].Samples[1280] != samples[len(samples)-1] || chunks[1].Samples[1281] != 0 {
		t.Fatal("second chunk has overlap, truncation, or incorrect right padding")
	}
}

func TestChunkInputFeatureShape(t *testing.T) {
	chunks, err := ChunkAudio(make([]float32, 1600))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	features, err := chunks[0].InputFeatures(cfg.WhisperConfig())
	if err != nil {
		t.Fatal(err)
	}
	if len(features) != 80*3000 {
		t.Fatalf("features=%d want %d", len(features), 80*3000)
	}
}
