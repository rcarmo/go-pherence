package training

import "fmt"

// TimeEmbeddingShape admits a sinusoidal cos-then-sin embedding (mode0), its
// selected-time JVP (mode1), or reverse-over-forward time adjoint (mode2).
// Inputs are frequency[half], [time,dTime], and two seed[2*half] vectors.
// Mode2 owns one scalar reduction, in frequency order, without float atomics.
// Raw callers must keep finite inputs and |time*frequency|<=8192. Tape admission
// checks this explicitly; native trigonometry still requires qualification.
func TimeEmbeddingShape(half int, mode uint32) (output int, err error) {
	if half < 1 || half > 8192 || mode > 2 {
		return 0, fmt.Errorf("invalid training time embedding shape/mode")
	}
	output = 2 * half
	if mode == 2 {
		output = 1
	}
	return output, nil
}
