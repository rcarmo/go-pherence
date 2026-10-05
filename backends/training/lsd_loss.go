package training

import "fmt"

// LSDLossShape admits one normalized LSD row. Mode0 writes [rawSquare, loss,
// dLogVariance]; mode1 writes prediction seeds. Statistics use a compensated
// F32 square reduction; logvar is bounded to [-80,80] by tape admission.
// Mode2 reports a scalar invalid flag using finite inputs/F32 differences and
// an exact host-derived cutoff in the third input, rather than logvar.
func LSDLossShape(channels int, mode uint32) (int, error) {
	if channels < 1 || channels > 16384 || mode > 2 {
		return 0, fmt.Errorf("invalid training LSD loss shape/mode")
	}
	if mode == 0 {
		return 3, nil
	}
	if mode == 2 {
		return 1, nil
	}
	return channels, nil
}
