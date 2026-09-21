package jevlike

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type layerNormCache struct {
	xHat   []float32
	invStd float32
}

// Backward analytically differentiates AttentionHead.Forward.
func (h *AttentionHead) Backward(
	context [][][]float32,
	contextMask [][]bool,
	options [][][]float32,
	optionMask [][]bool,
	dLogits [][]float32,
) (map[string][]float32, [][][]float32, [][][]float32, error) {
	if err := h.Validate(); err != nil {
		return nil, nil, nil, err
	}
	rows, contextTokens, optionsPerRow, err := validateHeadInputs(context, contextMask, options, optionMask, h.Width)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(dLogits) != rows {
		return nil, nil, nil, fmtBackwardGradientShape(rows, len(dLogits), -1, -1, optionsPerRow)
	}
	for row := 0; row < rows; row++ {
		if len(dLogits[row]) != optionsPerRow {
			return nil, nil, nil, fmtBackwardGradientShape(rows, len(dLogits), row, len(dLogits[row]), optionsPerRow)
		}
	}

	dContextNormWeight := make([]float32, len(h.ContextNormWeight))
	dContextNormBias := make([]float32, len(h.ContextNormBias))
	dOptionNormWeight := make([]float32, len(h.OptionNormWeight))
	dOptionNormBias := make([]float32, len(h.OptionNormBias))
	dQueryWeight := make([]float32, len(h.QueryWeight))
	dKeyWeight := make([]float32, len(h.KeyWeight))
	dValueWeight := make([]float32, len(h.ValueWeight))

	dContext := zeroLike3D(context)
	dOptions := zeroLike3D(options)
	rankScale := float32(math.Sqrt(float64(h.Rank)))

	for row := 0; row < rows; row++ {
		contextNormalized := make([][]float32, contextTokens)
		contextNormCache := make([]layerNormCache, contextTokens)
		keys := make([][]float32, contextTokens)
		values := make([][]float32, contextTokens)
		contextNormUpstream := make([][]float32, contextTokens)
		for token := 0; token < contextTokens; token++ {
			contextNormalized[token], contextNormCache[token] = layerNormForwardCached(context[row][token], h.ContextNormWeight, h.ContextNormBias)
			keys[token] = linearNoBias(h.KeyWeight, h.Rank, h.Width, contextNormalized[token])
			values[token] = linearNoBias(h.ValueWeight, h.Rank, h.Width, contextNormalized[token])
			contextNormUpstream[token] = make([]float32, h.Width)
		}

		for option := 0; option < optionsPerRow; option++ {
			optionNormalized, optionNormCache := layerNormForwardCached(options[row][option], h.OptionNormWeight, h.OptionNormBias)
			query := linearNoBias(h.QueryWeight, h.Rank, h.Width, optionNormalized)

			scores := make([]float32, contextTokens)
			for token := 0; token < contextTokens; token++ {
				scores[token] = dotFloat32(query, keys[token]) / rankScale
				if !contextMask[row][token] {
					scores[token] = maskedFillValue
				}
			}
			attention := softmaxFloat32(scores)
			attended := make([]float32, h.Rank)
			for token, weight := range attention {
				for dim := 0; dim < h.Rank; dim++ {
					attended[dim] += weight * values[token][dim]
				}
			}

			upstream := dLogits[row][option]
			if !optionMask[row][option] || upstream == 0 {
				continue
			}

			dQuery := make([]float32, h.Rank)
			dAttended := make([]float32, h.Rank)
			for dim := 0; dim < h.Rank; dim++ {
				dQuery[dim] = upstream * attended[dim] / rankScale
				dAttended[dim] = upstream * query[dim] / rankScale
			}

			dAttention := make([]float32, contextTokens)
			for token := 0; token < contextTokens; token++ {
				dAttention[token] = dotFloat32(dAttended, values[token])
				dValue := make([]float32, h.Rank)
				for dim := 0; dim < h.Rank; dim++ {
					dValue[dim] = attention[token] * dAttended[dim]
				}
				accumulateLinearNoBiasBackward(h.ValueWeight, h.Rank, h.Width, contextNormalized[token], dValue, dValueWeight, contextNormUpstream[token])
			}

			dScores := softmaxBackward(attention, dAttention)
			for token := 0; token < contextTokens; token++ {
				if !contextMask[row][token] {
					continue
				}
				dScore := dScores[token]
				if dScore == 0 {
					continue
				}
				dKey := make([]float32, h.Rank)
				for dim := 0; dim < h.Rank; dim++ {
					dQuery[dim] += dScore * keys[token][dim] / rankScale
					dKey[dim] = dScore * query[dim] / rankScale
				}
				accumulateLinearNoBiasBackward(h.KeyWeight, h.Rank, h.Width, contextNormalized[token], dKey, dKeyWeight, contextNormUpstream[token])
			}

			optionNormUpstream := make([]float32, h.Width)
			accumulateLinearNoBiasBackward(h.QueryWeight, h.Rank, h.Width, optionNormalized, dQuery, dQueryWeight, optionNormUpstream)
			dOption, dGamma, dBeta := layerNormBackward(optionNormCache, h.OptionNormWeight, optionNormUpstream)
			for dim := 0; dim < h.Width; dim++ {
				dOptions[row][option][dim] += dOption[dim]
				dOptionNormWeight[dim] += dGamma[dim]
				dOptionNormBias[dim] += dBeta[dim]
			}
		}

		for token := 0; token < contextTokens; token++ {
			dToken, dGamma, dBeta := layerNormBackward(contextNormCache[token], h.ContextNormWeight, contextNormUpstream[token])
			for dim := 0; dim < h.Width; dim++ {
				dContext[row][token][dim] += dToken[dim]
				dContextNormWeight[dim] += dGamma[dim]
				dContextNormBias[dim] += dBeta[dim]
			}
		}
	}

	paramGrads := map[string][]float32{
		defaultHeadPrefix + ".context_norm.weight": dContextNormWeight,
		defaultHeadPrefix + ".context_norm.bias":   dContextNormBias,
		defaultHeadPrefix + ".option_norm.weight":  dOptionNormWeight,
		defaultHeadPrefix + ".option_norm.bias":    dOptionNormBias,
		defaultHeadPrefix + ".query.weight":        dQueryWeight,
		defaultHeadPrefix + ".key.weight":          dKeyWeight,
		defaultHeadPrefix + ".value.weight":        dValueWeight,
	}
	return paramGrads, dContext, dOptions, nil
}

func fmtBackwardGradientShape(rows, gotRows, badRow, gotCols, wantCols int) error {
	if badRow < 0 {
		return fmt.Errorf("jevlike attention gradient row mismatch logits=%d want=%d", gotRows, rows)
	}
	return fmt.Errorf("jevlike attention gradient shape mismatch row=%d logits=%d want=%d", badRow, gotCols, wantCols)
}

func zeroLike3D(values [][][]float32) [][][]float32 {
	out := make([][][]float32, len(values))
	for i := range values {
		out[i] = make([][]float32, len(values[i]))
		for j := range values[i] {
			out[i][j] = make([]float32, len(values[i][j]))
		}
	}
	return out
}

func layerNormForwardCached(input, gamma, beta []float32) ([]float32, layerNormCache) {
	out := make([]float32, len(input))
	cache := layerNormCache{xHat: make([]float32, len(input))}
	if len(input) == 0 {
		return out, cache
	}
	var mean float64
	for _, value := range input {
		mean += float64(value)
	}
	mean /= float64(len(input))
	var variance float64
	for _, value := range input {
		delta := float64(value) - mean
		variance += delta * delta
	}
	variance /= float64(len(input))
	invStd := float32(1 / math.Sqrt(variance+float64(layerNormEpsilon)))
	cache.invStd = invStd
	for i, value := range input {
		xHat := float32((float64(value) - mean) * float64(invStd))
		cache.xHat[i] = xHat
		normalized := xHat
		if len(gamma) != 0 {
			normalized *= gamma[i]
		}
		if len(beta) != 0 {
			normalized += beta[i]
		}
		out[i] = normalized
	}
	return out, cache
}

func layerNormBackward(cache layerNormCache, gamma, dOutput []float32) ([]float32, []float32, []float32) {
	n := len(cache.xHat)
	dInput := make([]float32, n)
	dGamma := make([]float32, n)
	dBeta := make([]float32, n)
	if n == 0 {
		return dInput, dGamma, dBeta
	}
	dXHat := make([]float32, n)
	var sumDXHat float64
	var sumDXHatXHat float64
	for i := 0; i < n; i++ {
		dBeta[i] = dOutput[i]
		dGamma[i] = dOutput[i] * cache.xHat[i]
		dXHat[i] = dOutput[i]
		if len(gamma) != 0 {
			dXHat[i] *= gamma[i]
		}
		sumDXHat += float64(dXHat[i])
		sumDXHatXHat += float64(dXHat[i]) * float64(cache.xHat[i])
	}
	invN := float32(1) / float32(n)
	for i := 0; i < n; i++ {
		term := float32(float64(float32(n)*dXHat[i]) - sumDXHat - float64(cache.xHat[i])*sumDXHatXHat)
		dInput[i] = invN * cache.invStd * term
	}
	return dInput, dGamma, dBeta
}

func softmaxBackward(output, dOutput []float32) []float32 {
	dInput := make([]float32, len(output))
	var weightedSum float64
	for i := range output {
		weightedSum += float64(output[i]) * float64(dOutput[i])
	}
	for i := range output {
		dInput[i] = output[i] * (dOutput[i] - float32(weightedSum))
	}
	return dInput
}

func accumulateLinearNoBiasBackward(weight []float32, outDim, inDim int, input, dOutput, dWeight, dInput []float32) {
	for row := 0; row < outDim; row++ {
		base := row * inDim
		grad := dOutput[row]
		// Row-local gradient updates are independent SAXPY operations. Keep
		// row accumulation order while dispatching arithmetic to Plan 9.
		simd.Saxpy(grad, input[:inDim], dWeight[base:base+inDim])
		simd.Saxpy(grad, weight[base:base+inDim], dInput[:inDim])
	}
}
