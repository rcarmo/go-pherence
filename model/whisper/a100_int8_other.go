//go:build !riscv64

package whisper

import "github.com/rcarmo/go-pherence/backends/spacemit/ime2"

var (
	useA100FC1      = false
	useA100FC2      = false
	useA100FFNFused = false
	useA100X100Pack = false
	useA100NativeQ8 = false
)

func runA100Gemm(x []float32, seqLen, inDim int, w any, out []float32) bool         { return false }
func runA100GemmX100Pack(x []float32, seqLen, inDim int, w any, out []float32) bool { return false }
func linearForwardA100FC1(x, weight, bias []float32, seqLen, inDim, outDim int) ([]float32, bool) {
	return nil, false
}
func prepackA100EncoderWeights(enc *Encoder)                              {}
func getA100Q80x32Weight(weight []float32, outDim, inDim int) ime2.Q80x32 { return ime2.Q80x32{} }
func forwardA100FFNTile(layerIdx int, mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	return nil, false
}
func forwardA100FFNFC1NativeFC2Raw(mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	return nil, false
}
func forwardA100FFNFusedRaw(mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	return nil, false
}
func forwardA100FFNFused(layerIdx int, mlpIn []float32, layer *EncoderLayer, residual []float32, seqLen, dModel, ffnDim int) ([]float32, bool) {
	return nil, false
}
func resetA100Timers()       {}
func a100TimingLine() string { return "" }
