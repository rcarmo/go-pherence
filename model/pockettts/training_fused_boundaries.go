package pockettts

import "fmt"

// Fused component decoders retain the public result contracts. Keeping the
// count checks here also protects callers independently of executor admission.
func decodeDiagonalBoundary(out [][]float32, condition, input, prediction, desired []float32) (DiagonalTrainingResult, error) {
	if len(out) != 3 || len(out[2]) != 3 {
		return DiagonalTrainingResult{}, fmt.Errorf("diagonal boundary count")
	}
	r := DiagonalTrainingResult{out[0], out[1], float64(out[2][0]), float64(out[2][1]), out[2][2]}
	if err := validateDiagonalResult(r, condition, input, prediction, desired); err != nil {
		return DiagonalTrainingResult{}, err
	}
	return r, nil
}
func decodePrimaryBoundary(out [][]float32, in PrimaryTrainingInputs) (PrimaryTrainingResult, error) {
	if len(out) != 4 {
		return PrimaryTrainingResult{}, fmt.Errorf("primary boundary count")
	}
	r := PrimaryTrainingResult{out[0], out[1], out[2], out[3]}
	if err := validatePrimaryResult(r, in); err != nil {
		return PrimaryTrainingResult{}, err
	}
	return r, nil
}
func decodeDistillBoundary(out [][]float32, in DistillTrainingInputs) (DistillTrainingResult, error) {
	if len(out) != 5 || len(out[4]) != 3 {
		return DistillTrainingResult{}, fmt.Errorf("distill boundary count")
	}
	r := DistillTrainingResult{PrimaryTrainingResult{out[0], out[1], out[2], out[3]}, float64(out[4][0]), float64(out[4][1]), out[4][2]}
	if err := validateDistillResult(r, in); err != nil {
		return DistillTrainingResult{}, err
	}
	return r, nil
}
func decodeDistillForwardBoundary(out [][]float32, in DistillForwardInputs) (DistillForwardResult, error) {
	if len(out) != 6 {
		return DistillForwardResult{}, fmt.Errorf("distill forward boundary count")
	}
	r := DistillForwardResult{out[0], out[1], out[2], out[3], out[4], out[5]}
	if err := validateDistillForwardResult(r, in); err != nil {
		return DistillForwardResult{}, err
	}
	return r, nil
}
