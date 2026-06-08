//go:build !riscv64

package whisper

func linearForwardA100FC1(x, weight, bias []float32, seqLen, inDim, outDim int) ([]float32, bool) {
	return nil, false
}
func prepackA100EncoderWeights(enc *Encoder) {}
func resetA100Timers()                       {}
func a100TimingLine() string                 { return "" }
