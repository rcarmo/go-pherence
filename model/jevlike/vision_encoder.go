// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT

package jevlike

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
)

const (
	visionObservationChannels = 4
	visionObservationHeight   = 120
	visionObservationWidth    = 160
	visionPaddingTopBottom    = 4
	visionStemGroups          = 4
	visionStemEpsilon         = float32(1e-5)
	visionStem1Channels       = 16
	visionStem1Kernel         = 5
	visionStem1Stride         = 2
	visionStem1Padding        = 2
	visionStem2Kernel         = 3
	visionStem2Stride         = 2
	visionStem2Padding        = 1
	visionPatchKernel         = 4
	visionPatchStride         = 4
	visionPatchRows           = 8
	visionPatchColumns        = 10
	visionPositionParamName   = "positions"
)

// VisionEncoder is the native visual stem used by the upstream Jev-like vision
// scorer. It implements only the observation encoder: RGB normalisation,
// motion-channel scaling, two Conv+GroupNorm+SiLU stages, and a final patch
// projection.
type VisionEncoder struct {
	Width int `json:"width"`

	Positions []float32 `json:"positions"`

	StemConv1Weight []float32 `json:"stem_conv1_weight"`
	StemConv1Bias   []float32 `json:"stem_conv1_bias"`
	StemNorm1Weight []float32 `json:"stem_norm1_weight"`
	StemNorm1Bias   []float32 `json:"stem_norm1_bias"`

	StemConv2Weight []float32 `json:"stem_conv2_weight"`
	StemConv2Bias   []float32 `json:"stem_conv2_bias"`
	StemNorm2Weight []float32 `json:"stem_norm2_weight"`
	StemNorm2Bias   []float32 `json:"stem_norm2_bias"`

	PatchWeight []float32 `json:"patch_weight"`
	PatchBias   []float32 `json:"patch_bias"`
}

// NewVisionEncoder allocates a zero-initialized encoder with identity affine
// parameters for both GroupNorm stages and fixed 8x10 positions for 120x160
// observations.
func NewVisionEncoder(width int) (*VisionEncoder, error) {
	if width <= 0 {
		return nil, fmt.Errorf("invalid vision encoder width=%d", width)
	}
	positions, err := Position2D(visionPatchRows, visionPatchColumns, width)
	if err != nil {
		return nil, err
	}
	return &VisionEncoder{
		Width:           width,
		Positions:       positions,
		StemConv1Weight: make([]float32, visionStem1Channels*visionObservationChannels*visionStem1Kernel*visionStem1Kernel),
		StemConv1Bias:   make([]float32, visionStem1Channels),
		StemNorm1Weight: filledFloat32(visionStem1Channels, 1),
		StemNorm1Bias:   make([]float32, visionStem1Channels),
		StemConv2Weight: make([]float32, width*visionStem1Channels*visionStem2Kernel*visionStem2Kernel),
		StemConv2Bias:   make([]float32, width),
		StemNorm2Weight: filledFloat32(width, 1),
		StemNorm2Bias:   make([]float32, width),
		PatchWeight:     make([]float32, width*width*visionPatchKernel*visionPatchKernel),
		PatchBias:       make([]float32, width),
	}, nil
}

// Validate checks encoder dimensions, affine parameters, and fixed positions.
func (m *VisionEncoder) Validate() error {
	if m == nil {
		return fmt.Errorf("vision encoder is nil")
	}
	if m.Width <= 0 {
		return fmt.Errorf("invalid vision encoder width=%d", m.Width)
	}
	if m.Width%visionStemGroups != 0 {
		return fmt.Errorf("vision encoder width=%d must be divisible by %d", m.Width, visionStemGroups)
	}
	if err := m.validateFixedPositions(); err != nil {
		return err
	}
	if len(m.StemConv1Weight) != visionStem1Channels*visionObservationChannels*visionStem1Kernel*visionStem1Kernel || len(m.StemConv1Bias) != visionStem1Channels {
		return fmt.Errorf("invalid vision stem1 conv sizes weight=%d bias=%d", len(m.StemConv1Weight), len(m.StemConv1Bias))
	}
	if len(m.StemNorm1Weight) != visionStem1Channels || len(m.StemNorm1Bias) != visionStem1Channels {
		return fmt.Errorf("invalid vision stem1 groupnorm sizes weight=%d bias=%d", len(m.StemNorm1Weight), len(m.StemNorm1Bias))
	}
	if len(m.StemConv2Weight) != m.Width*visionStem1Channels*visionStem2Kernel*visionStem2Kernel || len(m.StemConv2Bias) != m.Width {
		return fmt.Errorf("invalid vision stem2 conv sizes weight=%d bias=%d width=%d", len(m.StemConv2Weight), len(m.StemConv2Bias), m.Width)
	}
	if len(m.StemNorm2Weight) != m.Width || len(m.StemNorm2Bias) != m.Width {
		return fmt.Errorf("invalid vision stem2 groupnorm sizes weight=%d bias=%d width=%d", len(m.StemNorm2Weight), len(m.StemNorm2Bias), m.Width)
	}
	if len(m.PatchWeight) != m.Width*m.Width*visionPatchKernel*visionPatchKernel || len(m.PatchBias) != m.Width {
		return fmt.Errorf("invalid vision patch sizes weight=%d bias=%d width=%d", len(m.PatchWeight), len(m.PatchBias), m.Width)
	}
	return nil
}

func (m *VisionEncoder) validateFixedPositions() error {
	positions, err := Position2D(visionPatchRows, visionPatchColumns, m.Width)
	if err != nil {
		return err
	}
	if len(m.Positions) != len(positions) {
		return fmt.Errorf("invalid vision positions length=%d, want %d", len(m.Positions), len(positions))
	}
	for i := range positions {
		if math.Abs(float64(m.Positions[i]-positions[i])) > 1e-6 {
			return fmt.Errorf("vision positions must equal fixed Position2D(%d,%d,%d)", visionPatchRows, visionPatchColumns, m.Width)
		}
	}
	return nil
}

// NamedParameters returns flattened arrays named like the upstream Python
// state_dict for the encoder subset.
func (m *VisionEncoder) NamedParameters() []NamedParameter {
	return []NamedParameter{
		{Name: visionPositionParamName, Shape: []int{visionPatchRows * visionPatchColumns, m.Width}, Values: append([]float32(nil), m.Positions...)},
		{Name: "stem.0.weight", Shape: []int{visionStem1Channels, visionObservationChannels, visionStem1Kernel, visionStem1Kernel}, Values: append([]float32(nil), m.StemConv1Weight...)},
		{Name: "stem.0.bias", Shape: []int{visionStem1Channels}, Values: append([]float32(nil), m.StemConv1Bias...)},
		{Name: "stem.1.weight", Shape: []int{visionStem1Channels}, Values: append([]float32(nil), m.StemNorm1Weight...)},
		{Name: "stem.1.bias", Shape: []int{visionStem1Channels}, Values: append([]float32(nil), m.StemNorm1Bias...)},
		{Name: "stem.3.weight", Shape: []int{m.Width, visionStem1Channels, visionStem2Kernel, visionStem2Kernel}, Values: append([]float32(nil), m.StemConv2Weight...)},
		{Name: "stem.3.bias", Shape: []int{m.Width}, Values: append([]float32(nil), m.StemConv2Bias...)},
		{Name: "stem.4.weight", Shape: []int{m.Width}, Values: append([]float32(nil), m.StemNorm2Weight...)},
		{Name: "stem.4.bias", Shape: []int{m.Width}, Values: append([]float32(nil), m.StemNorm2Bias...)},
		{Name: "patch.weight", Shape: []int{m.Width, m.Width, visionPatchKernel, visionPatchKernel}, Values: append([]float32(nil), m.PatchWeight...)},
		{Name: "patch.bias", Shape: []int{m.Width}, Values: append([]float32(nil), m.PatchBias...)},
	}
}

// NamedParameterMap returns flattened arrays keyed like the upstream Python
// state_dict.
func (m *VisionEncoder) NamedParameterMap() map[string][]float32 {
	params := m.NamedParameters()
	out := make(map[string][]float32, len(params))
	for _, param := range params {
		out[param.Name] = append([]float32(nil), param.Values...)
	}
	return out
}

// LoadNamedParameters copies provided named arrays into the encoder. The fixed
// positions buffer may be supplied, but it must match the deterministic 8x10
// Position2D table for this width.
func (m *VisionEncoder) LoadNamedParameters(params map[string][]float32) error {
	if err := m.Validate(); err != nil {
		return err
	}
	specs := map[string]*[]float32{
		"stem.0.weight": &m.StemConv1Weight,
		"stem.0.bias":   &m.StemConv1Bias,
		"stem.1.weight": &m.StemNorm1Weight,
		"stem.1.bias":   &m.StemNorm1Bias,
		"stem.3.weight": &m.StemConv2Weight,
		"stem.3.bias":   &m.StemConv2Bias,
		"stem.4.weight": &m.StemNorm2Weight,
		"stem.4.bias":   &m.StemNorm2Bias,
		"patch.weight":  &m.PatchWeight,
		"patch.bias":    &m.PatchBias,
	}
	for name, values := range params {
		if name == visionPositionParamName {
			if len(values) != len(m.Positions) {
				return fmt.Errorf("vision encoder parameter %q length=%d, want %d", name, len(values), len(m.Positions))
			}
			for i := range values {
				if math.Abs(float64(values[i]-m.Positions[i])) > 1e-6 {
					return fmt.Errorf("vision encoder parameter %q does not match fixed Position2D(%d,%d,%d)", name, visionPatchRows, visionPatchColumns, m.Width)
				}
			}
			continue
		}
		dst, ok := specs[name]
		if !ok {
			return fmt.Errorf("unknown vision encoder parameter %q", name)
		}
		if len(values) != len(*dst) {
			return fmt.Errorf("vision encoder parameter %q length=%d, want %d", name, len(values), len(*dst))
		}
		copy(*dst, values)
	}
	return nil
}

// Encode packs caller-supplied observations through ObservationTensor and
// returns patch features with one row per batch item plus the fixed 8x10
// position table.
func (m *VisionEncoder) Encode(items []PackedObservation) ([][][]float32, []float32, error) {
	tensor, err := ObservationTensor(items)
	if err != nil {
		return nil, nil, err
	}
	return m.EncodeTensor(tensor)
}

// EncodeTensor runs the native visual stem over one contiguous NCHW tensor
// shaped [batch,4,120,160] scaled like ObservationTensor.
func (m *VisionEncoder) EncodeTensor(observations ObservationTensorResult) ([][][]float32, []float32, error) {
	if err := m.Validate(); err != nil {
		return nil, nil, err
	}
	batch, err := validateVisionObservationTensor(observations)
	if err != nil {
		return nil, nil, err
	}
	prepared := make([]float32, batch*visionObservationChannels*(visionObservationHeight+2*visionPaddingTopBottom)*visionObservationWidth)
	if err := m.prepareObservations(prepared, observations); err != nil {
		return nil, nil, err
	}
	stem1, stem1H, stem1W, err := visionConv2D(prepared, batch, visionObservationChannels, visionObservationHeight+2*visionPaddingTopBottom, visionObservationWidth, visionStem1Channels, visionStem1Kernel, visionStem1Stride, visionStem1Padding, m.StemConv1Weight, m.StemConv1Bias)
	if err != nil {
		return nil, nil, err
	}
	if err = visionGroupNormSiLU(stem1, batch, visionStem1Channels, stem1H, stem1W, visionStemGroups, m.StemNorm1Weight, m.StemNorm1Bias, visionStemEpsilon); err != nil {
		return nil, nil, err
	}
	stem2, stem2H, stem2W, err := visionConv2D(stem1, batch, visionStem1Channels, stem1H, stem1W, m.Width, visionStem2Kernel, visionStem2Stride, visionStem2Padding, m.StemConv2Weight, m.StemConv2Bias)
	if err != nil {
		return nil, nil, err
	}
	if err = visionGroupNormSiLU(stem2, batch, m.Width, stem2H, stem2W, visionStemGroups, m.StemNorm2Weight, m.StemNorm2Bias, visionStemEpsilon); err != nil {
		return nil, nil, err
	}
	patches, patchH, patchW, err := visionConv2D(stem2, batch, m.Width, stem2H, stem2W, m.Width, visionPatchKernel, visionPatchStride, 0, m.PatchWeight, m.PatchBias)
	if err != nil {
		return nil, nil, err
	}
	if patchH != visionPatchRows || patchW != visionPatchColumns {
		return nil, nil, fmt.Errorf("vision patch grid=%dx%d, want %dx%d", patchH, patchW, visionPatchRows, visionPatchColumns)
	}
	features := make([][][]float32, batch)
	for b := 0; b < batch; b++ {
		features[b] = make([][]float32, patchH*patchW)
		for y := 0; y < patchH; y++ {
			for x := 0; x < patchW; x++ {
				token := y*patchW + x
				row := make([]float32, m.Width)
				for c := 0; c < m.Width; c++ {
					row[c] = patches[((b*m.Width+c)*patchH+y)*patchW+x]
				}
				features[b][token] = row
			}
		}
	}
	return features, append([]float32(nil), m.Positions...), nil
}

func validateVisionObservationTensor(observations ObservationTensorResult) (int, error) {
	if observations.Shape[0] <= 0 {
		return 0, fmt.Errorf("vision encoder needs at least one observation")
	}
	if observations.Shape[1] != visionObservationChannels || observations.Shape[2] != visionObservationHeight || observations.Shape[3] != visionObservationWidth {
		return 0, fmt.Errorf("vision encoder tensor shape=%v, want [batch %d %d %d]", observations.Shape, visionObservationChannels, visionObservationHeight, visionObservationWidth)
	}
	want := observations.Shape[0] * observations.Shape[1] * observations.Shape[2] * observations.Shape[3]
	if len(observations.Values) != want {
		return 0, fmt.Errorf("vision encoder tensor length=%d, want %d", len(observations.Values), want)
	}
	return observations.Shape[0], nil
}

func (m *VisionEncoder) prepareObservations(dst []float32, observations ObservationTensorResult) error {
	batch := observations.Shape[0]
	pixels := visionObservationHeight * visionObservationWidth
	paddedHeight := visionObservationHeight + 2*visionPaddingTopBottom
	for b := 0; b < batch; b++ {
		base := b * visionObservationChannels * pixels
		for c := 0; c < 3; c++ {
			channel := observations.Values[base+c*pixels : base+(c+1)*pixels]
			mean, std := sampleMeanStdClamp(channel, 0.08)
			for i, value := range channel {
				y := i / visionObservationWidth
				x := i % visionObservationWidth
				dst[((b*visionObservationChannels+c)*paddedHeight+(y+visionPaddingTopBottom))*visionObservationWidth+x] = (value - mean) / std
			}
		}
		motion := observations.Values[base+3*pixels : base+4*pixels]
		for i, value := range motion {
			y := i / visionObservationWidth
			x := i % visionObservationWidth
			dst[((b*visionObservationChannels+3)*paddedHeight+(y+visionPaddingTopBottom))*visionObservationWidth+x] = (value - 0.5) * 2.0
		}
	}
	return nil
}

func sampleMeanStdClamp(values []float32, minStd float32) (mean, std float32) {
	if len(values) == 0 {
		return 0, minStd
	}
	var sum float64
	for _, value := range values {
		sum += float64(value)
	}
	mean = float32(sum / float64(len(values)))
	if len(values) == 1 {
		return mean, minStd
	}
	var variance float64
	for _, value := range values {
		delta := float64(value - mean)
		variance += delta * delta
	}
	variance /= float64(len(values) - 1)
	std = float32(math.Sqrt(variance))
	if std < minStd {
		std = minStd
	}
	return mean, std
}

func visionConv2D(input []float32, batch, inChannels, inHeight, inWidth, outChannels, kernel, stride, padding int, weight, bias []float32) ([]float32, int, int, error) {
	if batch <= 0 || inChannels <= 0 || inHeight <= 0 || inWidth <= 0 || outChannels <= 0 || kernel <= 0 || stride <= 0 || padding < 0 {
		return nil, 0, 0, fmt.Errorf("invalid vision conv shape batch=%d in=%dx%dx%d out=%d kernel=%d stride=%d padding=%d", batch, inChannels, inHeight, inWidth, outChannels, kernel, stride, padding)
	}
	wantInput := batch * inChannels * inHeight * inWidth
	if len(input) != wantInput {
		return nil, 0, 0, fmt.Errorf("invalid vision conv input length=%d, want %d", len(input), wantInput)
	}
	weightCols := inChannels * kernel * kernel
	wantWeight := outChannels * weightCols
	if len(weight) != wantWeight || len(bias) != outChannels {
		return nil, 0, 0, fmt.Errorf("invalid vision conv parameter lengths weight=%d bias=%d want_weight=%d want_bias=%d", len(weight), len(bias), wantWeight, outChannels)
	}
	outHeight := (inHeight+2*padding-kernel)/stride + 1
	outWidth := (inWidth+2*padding-kernel)/stride + 1
	if outHeight <= 0 || outWidth <= 0 {
		return nil, 0, 0, fmt.Errorf("invalid vision conv output shape from in=%dx%d kernel=%d stride=%d padding=%d", inHeight, inWidth, kernel, stride, padding)
	}
	rows := batch * outHeight * outWidth
	col := make([]float32, rows*weightCols)
	for b := 0; b < batch; b++ {
		for oy := 0; oy < outHeight; oy++ {
			inputY := oy*stride - padding
			for ox := 0; ox < outWidth; ox++ {
				inputX := ox*stride - padding
				rowBase := ((b*outHeight+yIndex(oy))*outWidth + ox) * weightCols
				idx := rowBase
				for c := 0; c < inChannels; c++ {
					channelBase := ((b*inChannels + c) * inHeight) * inWidth
					for ky := 0; ky < kernel; ky++ {
						y := inputY + ky
						for kx := 0; kx < kernel; kx++ {
							x := inputX + kx
							if y >= 0 && y < inHeight && x >= 0 && x < inWidth {
								col[idx] = input[channelBase+y*inWidth+x]
							}
							idx++
						}
					}
				}
			}
		}
	}
	projected := make([]float32, rows*outChannels)
	if !visionGemmRows(projected, col, weight, rows, outChannels, weightCols) {
		return nil, 0, 0, fmt.Errorf("vision conv GEMM rejected rows=%d out_channels=%d cols=%d", rows, outChannels, weightCols)
	}
	for row := 0; row < rows; row++ {
		base := row * outChannels
		for c := 0; c < outChannels; c++ {
			projected[base+c] += bias[c]
		}
	}
	output := make([]float32, batch*outChannels*outHeight*outWidth)
	for b := 0; b < batch; b++ {
		for oy := 0; oy < outHeight; oy++ {
			for ox := 0; ox < outWidth; ox++ {
				rowBase := ((b*outHeight+yIndex(oy))*outWidth + ox) * outChannels
				for c := 0; c < outChannels; c++ {
					output[((b*outChannels+c)*outHeight+oy)*outWidth+ox] = projected[rowBase+c]
				}
			}
		}
	}
	return output, outHeight, outWidth, nil
}

func visionGemmRows(out, x, w []float32, batch, rows, cols int) bool {
	nOut, okOut := checked.MulInt(batch, rows)
	nX, okX := checked.MulInt(batch, cols)
	nW, okW := checked.MulInt(rows, cols)
	if batch <= 0 || rows <= 0 || cols <= 0 || !okOut || !okX || !okW || len(out) < nOut || len(x) < nX || len(w) < nW {
		return false
	}
	clear(out[:nOut])
	// Checked SGEMM has native Plan 9 dispatch; multi-row GemmRows is scalar.
	if simd.SgemmNTTo(out, x, w, batch, rows, cols, 1, cols, cols, rows) {
		return true
	}
	for b := 0; b < batch; b++ {
		xb := x[b*cols : (b+1)*cols]
		ob := out[b*rows : (b+1)*rows]
		for row := 0; row < rows; row++ {
			wrow := w[row*cols : (row+1)*cols]
			var sum float64
			for col, value := range xb {
				sum += float64(value) * float64(wrow[col])
			}
			ob[row] = float32(sum)
		}
	}
	return true
}

func visionGroupNormSiLU(values []float32, batch, channels, height, width, groups int, gamma, beta []float32, eps float32) error {
	if batch <= 0 || channels <= 0 || height <= 0 || width <= 0 || groups <= 0 {
		return fmt.Errorf("invalid vision groupnorm shape batch=%d channels=%d height=%d width=%d groups=%d", batch, channels, height, width, groups)
	}
	if channels%groups != 0 {
		return fmt.Errorf("vision groupnorm channels=%d not divisible by groups=%d", channels, groups)
	}
	if len(values) != batch*channels*height*width || len(gamma) != channels || len(beta) != channels {
		return fmt.Errorf("invalid vision groupnorm lengths values=%d gamma=%d beta=%d", len(values), len(gamma), len(beta))
	}
	hw := height * width
	channelsPerGroup := channels / groups
	for b := 0; b < batch; b++ {
		for g := 0; g < groups; g++ {
			channelStart := g * channelsPerGroup
			channelEnd := channelStart + channelsPerGroup
			count := channelsPerGroup * hw
			var mean float64
			for c := channelStart; c < channelEnd; c++ {
				base := ((b*channels + c) * height) * width
				for i := 0; i < hw; i++ {
					mean += float64(values[base+i])
				}
			}
			mean /= float64(count)
			var variance float64
			for c := channelStart; c < channelEnd; c++ {
				base := ((b*channels + c) * height) * width
				for i := 0; i < hw; i++ {
					delta := float64(values[base+i]) - mean
					variance += delta * delta
				}
			}
			invStd := 1 / math.Sqrt(variance/float64(count)+float64(eps))
			for c := channelStart; c < channelEnd; c++ {
				base := ((b*channels + c) * height) * width
				scale := gamma[c]
				shift := beta[c]
				for i := 0; i < hw; i++ {
					normalized := float32((float64(values[base+i]) - mean) * invStd)
					values[base+i] = siluFloat32(normalized*scale + shift)
				}
			}
		}
	}
	return nil
}

func siluFloat32(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

func yIndex(y int) int { return y }
