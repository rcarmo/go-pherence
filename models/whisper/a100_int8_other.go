//go:build !riscv64

package whisper

var useA100FC1 = false

func linearForwardA100FC1(x, weight, bias []float32, seqLen, inDim, outDim int) ([]float32, bool) {
	return nil, false
}

func resetA100Timers()       {}
func a100TimingLine() string { return "[a100] fc1=0s" }
