package jevlike

import (
	"math"
	"reflect"
	"testing"
)

func TestVisionEncoderShapeAndNamedParameters(t *testing.T) {
	if _, err := NewVisionEncoder(6); err == nil {
		t.Fatal("expected width divisibility error")
	}
	encoder, err := NewVisionEncoder(4)
	if err != nil {
		t.Fatal(err)
	}
	if err := encoder.Validate(); err != nil {
		t.Fatal(err)
	}
	params := encoder.NamedParameters()
	if got, want := len(params), 11; got != want {
		t.Fatalf("parameter count=%d want %d", got, want)
	}
	wantNames := []string{
		"positions",
		"stem.0.weight",
		"stem.0.bias",
		"stem.1.weight",
		"stem.1.bias",
		"stem.3.weight",
		"stem.3.bias",
		"stem.4.weight",
		"stem.4.bias",
		"patch.weight",
		"patch.bias",
	}
	gotNames := make([]string, len(params))
	for i, param := range params {
		gotNames[i] = param.Name
	}
	if !reflect.DeepEqual(gotNames, wantNames) {
		t.Fatalf("parameter names=%v want %v", gotNames, wantNames)
	}
	loaded, err := NewVisionEncoder(4)
	if err != nil {
		t.Fatal(err)
	}
	state := loaded.NamedParameterMap()
	state["stem.0.bias"] = []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	state["patch.bias"] = []float32{0.5, -0.5, 0.25, -0.25}
	if err := loaded.LoadNamedParameters(state); err != nil {
		t.Fatal(err)
	}
	assertClose1D(t, "stem.0.bias", loaded.StemConv1Bias, state["stem.0.bias"], 0)
	assertClose1D(t, "patch.bias", loaded.PatchBias, state["patch.bias"], 0)
	if err := loaded.LoadNamedParameters(map[string][]float32{"unknown": []float32{1}}); err == nil {
		t.Fatal("expected unknown parameter error")
	}
	badLen := loaded.NamedParameterMap()
	badLen["patch.bias"] = badLen["patch.bias"][:len(badLen["patch.bias"])-1]
	if err := loaded.LoadNamedParameters(badLen); err == nil {
		t.Fatal("expected parameter length error")
	}
	badPositions := loaded.NamedParameterMap()
	badPositions["positions"][0] += 1e-3
	if err := loaded.LoadNamedParameters(badPositions); err == nil {
		t.Fatal("expected fixed positions validation error")
	}
}

func TestVisionEncoderEncodeValidation(t *testing.T) {
	encoder, err := NewVisionEncoder(4)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := encoder.Encode(nil); err == nil {
		t.Fatal("expected empty observation batch error")
	}
	cases := []ObservationTensorResult{
		{Values: make([]float32, visionObservationChannels*visionObservationHeight*visionObservationWidth), Shape: [4]int{0, 4, 120, 160}},
		{Values: make([]float32, 3*visionObservationHeight*visionObservationWidth), Shape: [4]int{1, 3, 120, 160}},
		{Values: make([]float32, visionObservationChannels*119*visionObservationWidth), Shape: [4]int{1, 4, 119, 160}},
		{Values: make([]float32, visionObservationChannels*visionObservationHeight*visionObservationWidth-1), Shape: [4]int{1, 4, 120, 160}},
	}
	for i, tc := range cases {
		if _, _, err := encoder.EncodeTensor(tc); err == nil {
			t.Fatalf("case %d: expected validation error", i)
		}
	}
}

func TestVisionEncoderDeterministicScalarReference(t *testing.T) {
	encoder, err := NewVisionEncoder(4)
	if err != nil {
		t.Fatal(err)
	}
	fillVisionEncoderDeterministic(encoder)
	items := []PackedObservation{
		makePatternObservation(func(x, y, c int) uint8 {
			switch c {
			case 0:
				return uint8((3*x + 5*y) & 0xff)
			case 1:
				return uint8((7*x + 11*y + 13) & 0xff)
			case 2:
				return uint8((17*x + 19*y + 23) & 0xff)
			default:
				return uint8((x + 2*y) & 0xff)
			}
		}),
		makePatternObservation(func(x, y, c int) uint8 {
			switch c {
			case 0:
				return 96
			case 1:
				return 96
			case 2:
				return 96
			default:
				return uint8((9*x + 5*y + 31) & 0xff)
			}
		}),
	}
	gotFeatures, gotPositions, err := encoder.Encode(items)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(gotFeatures), len(items); got != want {
		t.Fatalf("batch rows=%d want %d", got, want)
	}
	for row := range gotFeatures {
		if got, want := len(gotFeatures[row]), visionPatchRows*visionPatchColumns; got != want {
			t.Fatalf("token rows[%d]=%d want %d", row, got, want)
		}
		for token := range gotFeatures[row] {
			if got, want := len(gotFeatures[row][token]), encoder.Width; got != want {
				t.Fatalf("feature width row=%d token=%d width=%d want %d", row, token, got, want)
			}
		}
	}
	tensor, err := ObservationTensor(items)
	if err != nil {
		t.Fatal(err)
	}
	wantFeatures, wantPositions := referenceVisionEncodeTensor(t, encoder, tensor)
	assertClose3D(t, "vision features", gotFeatures, wantFeatures, 1e-4)
	assertClose1D(t, "vision positions", gotPositions, wantPositions, 1e-7)
	fixedPositions, err := Position2D(visionPatchRows, visionPatchColumns, encoder.Width)
	if err != nil {
		t.Fatal(err)
	}
	assertClose1D(t, "fixed positions", gotPositions, fixedPositions, 1e-7)
}

func fillVisionEncoderDeterministic(encoder *VisionEncoder) {
	for i := range encoder.StemConv1Weight {
		encoder.StemConv1Weight[i] = float32((i%11)-5) / 40
	}
	for i := range encoder.StemConv1Bias {
		encoder.StemConv1Bias[i] = float32((i%7)-3) / 20
		encoder.StemNorm1Weight[i] = 0.75 + float32(i%5)/10
		encoder.StemNorm1Bias[i] = float32((i%9)-4) / 25
	}
	for i := range encoder.StemConv2Weight {
		encoder.StemConv2Weight[i] = float32((i%13)-6) / 50
	}
	for i := range encoder.StemConv2Bias {
		encoder.StemConv2Bias[i] = float32(i-1) / 10
		encoder.StemNorm2Weight[i] = 1.1 - float32(i)/20
		encoder.StemNorm2Bias[i] = float32(2*i-3) / 30
		encoder.PatchBias[i] = float32(3-i) / 40
	}
	for i := range encoder.PatchWeight {
		encoder.PatchWeight[i] = float32((i%17)-8) / 60
	}
}

func makePatternObservation(value func(x, y, c int) uint8) PackedObservation {
	pixels := make([]uint8, visionObservationHeight*visionObservationWidth*visionObservationChannels)
	for y := 0; y < visionObservationHeight; y++ {
		for x := 0; x < visionObservationWidth; x++ {
			base := (y*visionObservationWidth + x) * visionObservationChannels
			for c := 0; c < visionObservationChannels; c++ {
				pixels[base+c] = value(x, y, c)
			}
		}
	}
	return PackedObservation{Width: visionObservationWidth, Height: visionObservationHeight, Pixels: pixels}
}

func referenceVisionEncodeTensor(t *testing.T, encoder *VisionEncoder, tensor ObservationTensorResult) ([][][]float32, []float32) {
	t.Helper()
	prepared := referencePrepareVisionTensor(tensor)
	stem1, stem1H, stem1W := referenceVisionConv2D(t, prepared, tensor.Shape[0], visionObservationChannels, visionObservationHeight+2*visionPaddingTopBottom, visionObservationWidth, visionStem1Channels, visionStem1Kernel, visionStem1Stride, visionStem1Padding, encoder.StemConv1Weight, encoder.StemConv1Bias)
	referenceVisionGroupNormSiLU(stem1, tensor.Shape[0], visionStem1Channels, stem1H, stem1W, visionStemGroups, encoder.StemNorm1Weight, encoder.StemNorm1Bias, visionStemEpsilon)
	stem2, stem2H, stem2W := referenceVisionConv2D(t, stem1, tensor.Shape[0], visionStem1Channels, stem1H, stem1W, encoder.Width, visionStem2Kernel, visionStem2Stride, visionStem2Padding, encoder.StemConv2Weight, encoder.StemConv2Bias)
	referenceVisionGroupNormSiLU(stem2, tensor.Shape[0], encoder.Width, stem2H, stem2W, visionStemGroups, encoder.StemNorm2Weight, encoder.StemNorm2Bias, visionStemEpsilon)
	patch, patchH, patchW := referenceVisionConv2D(t, stem2, tensor.Shape[0], encoder.Width, stem2H, stem2W, encoder.Width, visionPatchKernel, visionPatchStride, 0, encoder.PatchWeight, encoder.PatchBias)
	if patchH != visionPatchRows || patchW != visionPatchColumns {
		t.Fatalf("reference patch grid=%dx%d want %dx%d", patchH, patchW, visionPatchRows, visionPatchColumns)
	}
	features := make([][][]float32, tensor.Shape[0])
	for b := 0; b < tensor.Shape[0]; b++ {
		features[b] = make([][]float32, patchH*patchW)
		for y := 0; y < patchH; y++ {
			for x := 0; x < patchW; x++ {
				token := y*patchW + x
				row := make([]float32, encoder.Width)
				for c := 0; c < encoder.Width; c++ {
					row[c] = patch[((b*encoder.Width+c)*patchH+y)*patchW+x]
				}
				features[b][token] = row
			}
		}
	}
	positions, err := Position2D(visionPatchRows, visionPatchColumns, encoder.Width)
	if err != nil {
		t.Fatal(err)
	}
	return features, positions
}

func referencePrepareVisionTensor(tensor ObservationTensorResult) []float32 {
	batch := tensor.Shape[0]
	pixels := visionObservationHeight * visionObservationWidth
	paddedHeight := visionObservationHeight + 2*visionPaddingTopBottom
	out := make([]float32, batch*visionObservationChannels*paddedHeight*visionObservationWidth)
	for b := 0; b < batch; b++ {
		base := b * visionObservationChannels * pixels
		for c := 0; c < 3; c++ {
			channel := tensor.Values[base+c*pixels : base+(c+1)*pixels]
			mean, std := referenceSampleMeanStdClamp(channel, 0.08)
			for i, value := range channel {
				y := i / visionObservationWidth
				x := i % visionObservationWidth
				out[((b*visionObservationChannels+c)*paddedHeight+(y+visionPaddingTopBottom))*visionObservationWidth+x] = (value - mean) / std
			}
		}
		motion := tensor.Values[base+3*pixels : base+4*pixels]
		for i, value := range motion {
			y := i / visionObservationWidth
			x := i % visionObservationWidth
			out[((b*visionObservationChannels+3)*paddedHeight+(y+visionPaddingTopBottom))*visionObservationWidth+x] = (value - 0.5) * 2
		}
	}
	return out
}

func referenceSampleMeanStdClamp(values []float32, minStd float32) (float32, float32) {
	var mean float64
	for _, value := range values {
		mean += float64(value)
	}
	mean /= float64(len(values))
	if len(values) == 1 {
		return float32(mean), minStd
	}
	var variance float64
	for _, value := range values {
		delta := float64(value) - mean
		variance += delta * delta
	}
	variance /= float64(len(values) - 1)
	std := float32(math.Sqrt(variance))
	if std < minStd {
		std = minStd
	}
	return float32(mean), std
}

func referenceVisionConv2D(t *testing.T, input []float32, batch, inChannels, inHeight, inWidth, outChannels, kernel, stride, padding int, weight, bias []float32) ([]float32, int, int) {
	t.Helper()
	outHeight := (inHeight+2*padding-kernel)/stride + 1
	outWidth := (inWidth+2*padding-kernel)/stride + 1
	out := make([]float32, batch*outChannels*outHeight*outWidth)
	for b := 0; b < batch; b++ {
		for oc := 0; oc < outChannels; oc++ {
			for oy := 0; oy < outHeight; oy++ {
				inputY := oy*stride - padding
				for ox := 0; ox < outWidth; ox++ {
					inputX := ox*stride - padding
					sum := float64(bias[oc])
					for ic := 0; ic < inChannels; ic++ {
						weightBase := ((oc*inChannels + ic) * kernel) * kernel
						inputBase := ((b*inChannels + ic) * inHeight) * inWidth
						for ky := 0; ky < kernel; ky++ {
							y := inputY + ky
							for kx := 0; kx < kernel; kx++ {
								x := inputX + kx
								if y < 0 || y >= inHeight || x < 0 || x >= inWidth {
									continue
								}
								sum += float64(input[inputBase+y*inWidth+x]) * float64(weight[weightBase+ky*kernel+kx])
							}
						}
					}
					out[((b*outChannels+oc)*outHeight+oy)*outWidth+ox] = float32(sum)
				}
			}
		}
	}
	return out, outHeight, outWidth
}

func referenceVisionGroupNormSiLU(values []float32, batch, channels, height, width, groups int, gamma, beta []float32, eps float32) {
	hw := height * width
	channelsPerGroup := channels / groups
	for b := 0; b < batch; b++ {
		for g := 0; g < groups; g++ {
			start := g * channelsPerGroup
			end := start + channelsPerGroup
			count := channelsPerGroup * hw
			var mean float64
			for c := start; c < end; c++ {
				base := ((b*channels + c) * height) * width
				for i := 0; i < hw; i++ {
					mean += float64(values[base+i])
				}
			}
			mean /= float64(count)
			var variance float64
			for c := start; c < end; c++ {
				base := ((b*channels + c) * height) * width
				for i := 0; i < hw; i++ {
					delta := float64(values[base+i]) - mean
					variance += delta * delta
				}
			}
			invStd := 1 / math.Sqrt(variance/float64(count)+float64(eps))
			for c := start; c < end; c++ {
				base := ((b*channels + c) * height) * width
				for i := 0; i < hw; i++ {
					normalized := (float64(values[base+i]) - mean) * invStd
					values[base+i] = referenceSiLU(float32(normalized)*gamma[c] + beta[c])
				}
			}
		}
	}
}

func referenceSiLU(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}
