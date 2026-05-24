// Package speaker implements speaker embedding and diarization models.
package speaker

// Config holds speaker encoder architecture parameters.
type Config struct {
	// Input
	SampleRate int // 16000
	NumMels    int // 80

	// ECAPA-TDNN architecture
	Channels   []int // Channel sizes for each conv block, e.g. [512, 512, 512, 512, 1536]
	KernelSize int   // Typically 5
	EmbedDim   int   // Output embedding dimension, typically 192

	// SE-block
	SEBottleneck int // SE reduction ratio channel count

	// Attentive statistics pooling
	AttentionDim int // Attention hidden dim for pooling
}

// DefaultECAPAConfig returns a standard ECAPA-TDNN configuration.
func DefaultECAPAConfig() Config {
	return Config{
		SampleRate:   16000,
		NumMels:      80,
		Channels:     []int{512, 512, 512, 512, 1536},
		KernelSize:   5,
		EmbedDim:     192,
		SEBottleneck: 128,
		AttentionDim: 128,
	}
}
