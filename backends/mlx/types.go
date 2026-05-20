package mlx

// MLXQuantWeight holds MLX affine quantized weight data.
type QuantWeight struct {
	Weight    []uint32  // [outDim, inDim/8] packed 4-bit
	Scales    []float32 // [outDim, numGroups]
	Biases    []float32 // [outDim, numGroups]
	OutDim    int
	InDim     int
	Groups    int // numGroups = inDim / groupSize
	GroupSize int
	Bits      int
}
