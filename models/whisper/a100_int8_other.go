//go:build !riscv64

package whisper

func linearForwardA100FC1(x, weight, bias []float32, seqLen, inDim, outDim int) ([]float32, bool) {
	return nil, false
}
func prepackA100EncoderWeights(enc *Encoder) {}
func forwardA100FFNFC1NativeFC2Raw(mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	return nil, false
}
func forwardA100FFNFusedRaw(mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	return nil, false
}
func forwardA100FFNFused(mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	return nil, false
}
func resetA100Timers()       {}
func a100TimingLine() string { return "" }
