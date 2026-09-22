package pockettts

import (
	"fmt"
	"math"
)

// EOSLossAndGradient implements upstream's EOS boundary exactly: target one on
// the first invalid frame, target zero elsewhere, reduced over valid frames
// plus that first invalid frame. The first frame is never an EOS target.
func EOSLossAndGradient(logits []float32, mask []bool) (float64, []float32, error) {
	if len(logits) == 0 || len(logits) != len(mask) {
		return 0, nil, fmt.Errorf("invalid Pocket TTS EOS loss shape logits=%d mask=%d", len(logits), len(mask))
	}
	seenPadding := false
	for i, valid := range mask {
		if !valid {
			seenPadding = true
		} else if seenPadding {
			return 0, nil, fmt.Errorf("Pocket TTS EOS mask is not a contiguous prefix at position %d", i)
		}
	}
	active := make([]bool, len(mask))
	active[0] = mask[0]
	for i := 1; i < len(mask); i++ {
		active[i] = mask[i-1]
	}
	count := 0
	for _, enabled := range active {
		if enabled {
			count++
		}
	}
	if count == 0 {
		return 0, nil, fmt.Errorf("Pocket TTS EOS loss has no active positions")
	}
	gradient := make([]float32, len(logits))
	var loss float64
	inv := 1 / float64(count)
	for i, enabled := range active {
		if !enabled {
			continue
		}
		logit := float64(logits[i])
		if !finite64(logit) {
			return 0, nil, fmt.Errorf("Pocket TTS EOS logit %d is non-finite", i)
		}
		target := i > 0 && !mask[i]
		if target {
			loss += softplus64(-logit)
			gradient[i] = float32((sigmoid64(logit) - 1) * inv)
		} else {
			loss += softplus64(logit)
			gradient[i] = float32(sigmoid64(logit) * inv)
		}
	}
	return loss * inv, gradient, nil
}

// FlowMatchingLossAndGradient implements the upstream OT flow-matching loss
// for already sampled times/noise. Rows are flattened [N,C]; the returned
// gradient is d(mean(loss))/d(prediction).
func FlowMatchingLossAndGradient(prediction, noise, target, times []float32, channels int, sigMin float32) (float64, []float32, error) {
	if channels <= 0 || len(prediction) == 0 || len(prediction)%channels != 0 || len(noise) != len(prediction) || len(target) != len(prediction) || len(times) != len(prediction)/channels || !isFinite(sigMin) || sigMin < 0 || sigMin >= 1 {
		return 0, nil, fmt.Errorf("invalid Pocket TTS flow-matching loss shape")
	}
	rows := len(times)
	gradient := make([]float32, len(prediction))
	var total float64
	for row := 0; row < rows; row++ {
		if !isFinite(times[row]) || times[row] < 0 || times[row] > 1 {
			return 0, nil, fmt.Errorf("Pocket TTS flow time %d is outside [0,1]", row)
		}
		for channel := 0; channel < channels; channel++ {
			i := row*channels + channel
			if !isFinite(prediction[i]) || !isFinite(noise[i]) || !isFinite(target[i]) {
				return 0, nil, fmt.Errorf("Pocket TTS flow row %d channel %d is non-finite", row, channel)
			}
			desired := target[i] - (1-sigMin)*noise[i]
			difference := prediction[i] - desired
			total += float64(difference) * float64(difference)
			gradient[i] = 2 * difference / float32(rows*channels)
		}
	}
	return total / float64(rows*channels), gradient, nil
}

// LSDDiagonalLossAndGradient implements the released model's instantaneous
// s==t LSD term. prediction and desiredVelocity are [N,C]. If normalize is
// true, logVariance is one scalar per row and the returned logVariance
// gradient follows flow*exp(logvar)/C-logvar.
func LSDDiagonalLossAndGradient(prediction, desiredVelocity, logVariance []float32, channels int, normalize bool) (loss float64, dPrediction, dLogVariance []float32, err error) {
	if channels <= 0 || len(prediction) == 0 || len(prediction)%channels != 0 || len(desiredVelocity) != len(prediction) {
		return 0, nil, nil, fmt.Errorf("invalid Pocket TTS LSD diagonal loss shape")
	}
	rows := len(prediction) / channels
	if normalize && len(logVariance) != rows {
		return 0, nil, nil, fmt.Errorf("invalid Pocket TTS LSD log-variance shape=%d want=%d", len(logVariance), rows)
	}
	if !normalize && len(logVariance) != 0 {
		return 0, nil, nil, fmt.Errorf("Pocket TTS LSD log-variance supplied without normalization")
	}
	dPrediction = make([]float32, len(prediction))
	if normalize {
		dLogVariance = make([]float32, rows)
	}
	for row := 0; row < rows; row++ {
		rowSquare := 0.0
		for channel := 0; channel < channels; channel++ {
			i := row*channels + channel
			if !isFinite(prediction[i]) || !isFinite(desiredVelocity[i]) {
				return 0, nil, nil, fmt.Errorf("Pocket TTS LSD row %d channel %d is non-finite", row, channel)
			}
			difference := prediction[i] - desiredVelocity[i]
			rowSquare += float64(difference) * float64(difference)
		}
		scale := 1.0
		rowLoss := rowSquare
		if normalize {
			logvar := float64(logVariance[row])
			if !finite64(logvar) {
				return 0, nil, nil, fmt.Errorf("Pocket TTS LSD log-variance %d is non-finite", row)
			}
			scale = math.Exp(logvar) / float64(channels)
			rowLoss = rowSquare*scale - logvar
			dLogVariance[row] = float32((rowSquare*scale - 1) / float64(rows))
		}
		loss += rowLoss
		for channel := 0; channel < channels; channel++ {
			i := row*channels + channel
			dPrediction[i] = float32(2 * float64(prediction[i]-desiredVelocity[i]) * scale / float64(rows))
		}
	}
	return loss / float64(rows), dPrediction, dLogVariance, nil
}

func softplus64(x float64) float64 {
	if x > 0 {
		return x + math.Log1p(math.Exp(-x))
	}
	return math.Log1p(math.Exp(x))
}

func sigmoid64(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}
	z := math.Exp(x)
	return z / (1 + z)
}
