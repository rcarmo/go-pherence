package pockettts

import "testing"

func TestTrainingLinearRowsMatchesRowReference(t *testing.T) {
	const rows, in, out = 17, 7, 11
	linear := LinearF32{Weight: make([]float32, in*out), Bias: make([]float32, out), In: in, Out: out}
	input, dOutput := make([]float32, rows*in), make([]float32, rows*out)
	for i := range linear.Weight {
		linear.Weight[i] = float32((i%19)-9) / 23
	}
	for i := range linear.Bias {
		linear.Bias[i] = float32(i-5) / 17
	}
	for i := range input {
		input[i] = float32((i%13)-6) / 11
	}
	for i := range dOutput {
		dOutput[i] = float32((i%17)-8) / 19
	}

	got := linearForwardRowsTraining(linear, input, rows)
	want := make([]float32, len(got))
	for row := 0; row < rows; row++ {
		affineScalar(want[row*out:(row+1)*out], input[row*in:(row+1)*in], linear.Weight, linear.Bias, in, out)
	}
	assertSliceClose(t, "batched training forward", got, want, 2e-6)

	gotGradient := newLinearGradient(linear)
	gotInput := linearBackwardRowsTraining(linear, input, dOutput, rows, &gotGradient)
	wantGradient := newLinearGradient(linear)
	wantInput := make([]float32, rows*in)
	for row := 0; row < rows; row++ {
		for output, d := range dOutput[row*out : (row+1)*out] {
			wantGradient.Bias[output] += d
			for column, x := range input[row*in : (row+1)*in] {
				wantGradient.Weight[output*in+column] += d * x
				wantInput[row*in+column] += d * linear.Weight[output*in+column]
			}
		}
	}
	assertSliceClose(t, "batched training dInput", gotInput, wantInput, 2e-6)
	assertSliceClose(t, "batched training dWeight", gotGradient.Weight, wantGradient.Weight, 3e-6)
	assertSliceClose(t, "batched training dBias", gotGradient.Bias, wantGradient.Bias, 0)
}
