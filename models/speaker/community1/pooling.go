package community1

import (
	"context"
	"fmt"
	"math"
	"math/bits"
)

// StatsPoolConfig describes one window of channel-major [Features,Frames]
// frame embeddings. Flatten WeSpeaker [dimension,channel,frames] into that layout
// WITHOUT transposing the time dimension. Input is not PCM or Fbank features.
// With nil masks, Speakers and MaskFrames must be zero and one unweighted output
// is returned. Otherwise masks have shape [Speakers,MaskFrames], values [0,1].
type StatsPoolConfig struct{ Features, Frames, Speakers, MaskFrames int }

// StatsPoolResult contains speaker-major [speakers,2*features] concatenated mean
// then standard deviation, plus support metadata AFTER nearest mask resizing.
// Unweighted output has one row. WeightSum excludes epsilon; NonzeroFrames is
// the count of positive mask weights, not a duration or global speaker identity.
// In particular zero support has zero statistics, but MUST NOT automatically be
// treated as a usable speaker embedding. Downstream admission remains separate.
// All returned slices are owned; no state is retained or sources mutated.
type StatsPoolResult struct {
	Statistics    []float32
	WeightSum     []float32
	NonzeroFrames []int
}

// StatsPool implements pinned pyannote StatsPool used by WeSpeaker TSTP. Bounds:
// features1..4096, sequence frames1..4096, masks1..8, mask frames1..4096. An
// unweighted single frame is rejected because unbiased std would be undefined.
// Weighted zero/single support follows the upstream epsilon formula exactly in
// structure. Values/weights must be finite; weighted masks must be in [0,1].
//
// Masks use legacy Torch nearest indexing with float32 scale=M/T and
// floor(float32(t*scale)), clamped to M-1. Float32 boundary rounding matters
// (2->82 maps frame41 to mask0); this is NOT exact integer-ratio flooring,
// linear or nearest-exact interpolation. This is an index-space resize,
// not source-PCM/receptive-field/timestamp alignment. Mask generation and any
// exclusion of overlapping speakers happen before this boundary.
//
// Multiple masks share immutable frame features; no expanded features or
// per-speaker CNN computation is performed here. Cancellation is checked in
// bounded reduction blocks. No partial result escapes on error. No specialised
// SIMD kernel, full WeSpeaker embedding or performance result is claimed.
func StatsPool(ctx context.Context, features, masks []float32, cfg StatsPoolConfig) (*StatsPoolResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c := cfg
	if c.Features < 1 || c.Features > 4096 || c.Frames < 1 || c.Frames > 4096 || len(features) != c.Features*c.Frames {
		return nil, fmt.Errorf("invalid statistics pool feature geometry")
	}
	weighted := masks != nil
	speakers := c.Speakers
	if !weighted {
		if c.Speakers != 0 || c.MaskFrames != 0 || c.Frames < 2 {
			return nil, fmt.Errorf("unweighted pool requires no mask geometry and at least two frames")
		}
		speakers = 1
	} else if c.Speakers < 1 || c.Speakers > 8 || c.MaskFrames < 1 || c.MaskFrames > 4096 || len(masks) != c.Speakers*c.MaskFrames {
		return nil, fmt.Errorf("invalid statistics pool mask geometry")
	}
	if err := finitePool(ctx, features, false); err != nil {
		return nil, err
	}
	if err := finitePool(ctx, masks, true); err != nil {
		return nil, err
	}
	result := &StatsPoolResult{Statistics: make([]float32, speakers*2*c.Features), WeightSum: make([]float32, speakers), NonzeroFrames: make([]int, speakers)}
	// Reusable mask/product rows. Feature arrays are never broadcast/copied per
	// mask. PyTorch materialises each product then reduces it with its pinned
	// float32 CPU sum tree; preserving that explicit order avoids ISA-dependent
	// compiler reduction choices while producing the same result on every Go
	// architecture.
	var weights, products []float32
	if weighted {
		weights = make([]float32, c.Frames)
		products = make([]float32, c.Frames)
	}
	for speaker := 0; speaker < speakers; speaker++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		v1, v2 := float32(c.Frames), float32(c.Frames)
		if weighted {
			support := 0
			for frame := 0; frame < c.Frames; frame++ {
				if frame%256 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				value := masks[speaker*c.MaskFrames+poolMaskIndex(frame, c.MaskFrames, c.Frames)]
				weights[frame] = value
				products[frame] = value * value
				if value > 0 {
					support++
				}
			}
			sum := torchSumF32(weights)
			result.WeightSum[speaker] = sum
			result.NonzeroFrames[speaker] = support
			v1 = sum + float32(1e-8)
			v2 = torchSumF32(products)
		} else {
			result.WeightSum[speaker] = float32(c.Frames)
			result.NonzeroFrames[speaker] = c.Frames
		}
		denominator := float64(c.Frames - 1)
		if weighted {
			// Preserve float32 cancellation and the two explicit epsilon additions.
			denominator = float64(v1 - v2/v1 + float32(1e-8))
			if denominator <= 0 || math.IsNaN(denominator) || math.IsInf(denominator, 0) {
				return nil, fmt.Errorf("invalid weighted variance denominator")
			}
		}
		for feature := 0; feature < c.Features; feature++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			row := features[feature*c.Frames : (feature+1)*c.Frames]
			mean := float64(0)
			if weighted {
				for i, value := range row {
					if i%256 == 0 {
						if err := ctx.Err(); err != nil {
							return nil, err
						}
					}
					products[i] = value * weights[i]
				}
				mean = float64(torchSumF32(products) / v1)
			} else {
				for i, value := range row {
					if i%256 == 0 {
						if err := ctx.Err(); err != nil {
							return nil, err
						}
					}
					mean += float64(value)
				}
				mean /= float64(c.Frames)
			}
			variance := float64(0)
			if weighted {
				for i, value := range row {
					if i%256 == 0 {
						if err := ctx.Err(); err != nil {
							return nil, err
						}
					}
					delta := value - float32(mean)
					products[i] = (delta * delta) * weights[i]
				}
				variance = float64(torchSumF32(products) / float32(denominator))
			} else {
				for i, value := range row {
					if i%256 == 0 {
						if err := ctx.Err(); err != nil {
							return nil, err
						}
					}
					delta := float64(value) - mean
					variance += delta * delta
				}
				variance /= denominator
			}
			std := float32(math.Sqrt(variance))
			mu := float32(mean)
			if math.IsNaN(float64(mu)) || math.IsInf(float64(mu), 0) || math.IsNaN(float64(std)) || math.IsInf(float64(std), 0) {
				return nil, fmt.Errorf("non-finite statistics pool reduction")
			}
			offset := speaker * 2 * c.Features
			result.Statistics[offset+feature] = mu
			result.Statistics[offset+c.Features+feature] = std
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

// torchSumF32 reproduces the pinned PyTorch CPU contiguous float32 sum tree:
// eight scalar lanes per vector, four interleaved vector accumulators, and the
// four-level cascade from ATen/native/cpu/SumKernel.cpp. Scalar remainder values
// are accumulated before the eight lanes. The explicit tree is architecture
// independent and is bounded here to the StatsPool limit of 4096 elements.
func torchSumF32(values []float32) float32 {
	const (
		lanes  = 8
		rows   = 4
		levels = 4
	)
	// Below one hardware vector, ATen takes scalar_inner_sum. row_sum still
	// uses four interleaved accumulators rather than a single serial total.
	if len(values) < lanes {
		var partial [rows]float32
		groups := len(values) / rows
		for group := 0; group < groups; group++ {
			for row := 0; row < rows; row++ {
				partial[row] += values[group*rows+row]
			}
		}
		for i := groups * rows; i < len(values); i++ {
			partial[0] += values[i]
		}
		for row := 1; row < rows; row++ {
			partial[0] += partial[row]
		}
		return partial[0]
	}
	vectors := len(values) / lanes
	groups := vectors / rows
	levelPower := max(4, bits.Len(uint(max(groups-1, 0)))/levels)
	levelStep := 1 << levelPower
	levelMask := levelStep - 1
	var sums [levels][rows][lanes]float32
	group := 0
	for group+levelStep <= groups {
		for end := group + levelStep; group < end; group++ {
			for row := 0; row < rows; row++ {
				base := (group*rows + row) * lanes
				for lane := 0; lane < lanes; lane++ {
					sums[0][row][lane] += values[base+lane]
				}
			}
		}
		for level := 1; level < levels; level++ {
			for row := 0; row < rows; row++ {
				for lane := 0; lane < lanes; lane++ {
					sums[level][row][lane] += sums[level-1][row][lane]
					sums[level-1][row][lane] = 0
				}
			}
			if group&(levelMask<<(level*levelPower)) != 0 {
				break
			}
		}
	}
	for ; group < groups; group++ {
		for row := 0; row < rows; row++ {
			base := (group*rows + row) * lanes
			for lane := 0; lane < lanes; lane++ {
				sums[0][row][lane] += values[base+lane]
			}
		}
	}
	// multi_row_sum folds its cascade levels before row_sum adds vectors that
	// did not fill a complete four-vector group. Keep that order exactly: the
	// two additions do not commute under float32 rounding.
	for level := 1; level < levels; level++ {
		for row := 0; row < rows; row++ {
			for lane := 0; lane < lanes; lane++ {
				sums[0][row][lane] += sums[level][row][lane]
			}
		}
	}
	for vector := groups * rows; vector < vectors; vector++ {
		base := vector * lanes
		for lane := 0; lane < lanes; lane++ {
			sums[0][0][lane] += values[base+lane]
		}
	}
	for row := 1; row < rows; row++ {
		for lane := 0; lane < lanes; lane++ {
			sums[0][0][lane] += sums[0][row][lane]
		}
	}
	var result float32
	for i := vectors * lanes; i < len(values); i++ {
		result += values[i]
	}
	for lane := 0; lane < lanes; lane++ {
		result += sums[0][0][lane]
	}
	return result
}

func finitePool(ctx context.Context, values []float32, mask bool) error {
	for i, value := range values {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || (mask && (value < 0 || value > 1)) {
			return fmt.Errorf("invalid statistics pool values/masks")
		}
	}
	return ctx.Err()
}

// Torch CPU nearest_idx helpers retain identity and exact 2x upsampling fast
// paths. Other ratios round both scale and product to float32 before flooring.
func poolMaskIndex(frame, sourceFrames, targetFrames int) int {
	if sourceFrames == targetFrames {
		return frame
	}
	if targetFrames == 2*sourceFrames {
		return frame / 2
	}
	scale := float32(sourceFrames) / float32(targetFrames)
	return min(int(float32(float32(frame)*scale)), sourceFrames-1)
}
