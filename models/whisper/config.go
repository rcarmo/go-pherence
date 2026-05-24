// Package whisper implements OpenAI Whisper speech recognition models.
package whisper

// Config holds Whisper model architecture parameters.
type Config struct {
	// Audio
	NumMelBins int // 80
	MaxLength  int // 3000 (30s at 10ms hop = 3000 frames)

	// Encoder
	EncoderLayers int // 4 (tiny) to 32 (large-v3)
	EncoderDModel int // 384 (tiny) to 1280 (large-v3)
	EncoderHeads  int // 6 (tiny) to 20 (large-v3)
	EncoderFFNDim int // 4 * d_model

	// Decoder
	DecoderLayers int
	DecoderDModel int
	DecoderHeads  int
	DecoderFFNDim int

	// Vocabulary
	VocabSize        int // 51865 for multilingual, 51864 for English-only
	MaxDecoderLength int // 448

	// Derived
	HeadDim int // d_model / num_heads
}

// Tiny returns config for whisper-tiny (39M parameters).
func Tiny() Config {
	return Config{
		NumMelBins:       80,
		MaxLength:        3000,
		EncoderLayers:    4,
		EncoderDModel:    384,
		EncoderHeads:     6,
		EncoderFFNDim:    1536,
		DecoderLayers:    4,
		DecoderDModel:    384,
		DecoderHeads:     6,
		DecoderFFNDim:    1536,
		VocabSize:        51865,
		MaxDecoderLength: 448,
		HeadDim:          64,
	}
}

// Base returns config for whisper-base (74M parameters).
func Base() Config {
	return Config{
		NumMelBins:       80,
		MaxLength:        3000,
		EncoderLayers:    6,
		EncoderDModel:    512,
		EncoderHeads:     8,
		EncoderFFNDim:    2048,
		DecoderLayers:    6,
		DecoderDModel:    512,
		DecoderHeads:     8,
		DecoderFFNDim:    2048,
		VocabSize:        51865,
		MaxDecoderLength: 448,
		HeadDim:          64,
	}
}

// Small returns config for whisper-small (244M parameters).
func Small() Config {
	return Config{
		NumMelBins:       80,
		MaxLength:        3000,
		EncoderLayers:    12,
		EncoderDModel:    768,
		EncoderHeads:     12,
		EncoderFFNDim:    3072,
		DecoderLayers:    12,
		DecoderDModel:    768,
		DecoderHeads:     12,
		DecoderFFNDim:    3072,
		VocabSize:        51865,
		MaxDecoderLength: 448,
		HeadDim:          64,
	}
}

// Medium returns config for whisper-medium (769M parameters).
func Medium() Config {
	return Config{
		NumMelBins:       80,
		MaxLength:        3000,
		EncoderLayers:    24,
		EncoderDModel:    1024,
		EncoderHeads:     16,
		EncoderFFNDim:    4096,
		DecoderLayers:    24,
		DecoderDModel:    1024,
		DecoderHeads:     16,
		DecoderFFNDim:    4096,
		VocabSize:        51865,
		MaxDecoderLength: 448,
		HeadDim:          64,
	}
}

// LargeV3 returns config for whisper-large-v3 (1550M parameters).
func LargeV3() Config {
	return Config{
		NumMelBins:       128, // large-v3 uses 128 mel bins
		MaxLength:        3000,
		EncoderLayers:    32,
		EncoderDModel:    1280,
		EncoderHeads:     20,
		EncoderFFNDim:    5120,
		DecoderLayers:    32,
		DecoderDModel:    1280,
		DecoderHeads:     20,
		DecoderFFNDim:    5120,
		VocabSize:        51866,
		MaxDecoderLength: 448,
		HeadDim:          64,
	}
}
