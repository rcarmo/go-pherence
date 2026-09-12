// Package community1 contains components for the Community-1 diarization port.
// This package does not yet implement a complete diarization pipeline.
package community1

import (
	"context"
	"fmt"
	"math"
)

// The class order and permutation convention follow pyannote.audio Powerset,
// pinned in testdata/powerset-reference.json. See NOTICE for MIT attribution.

const MaxPowersetFrames = 4096

// Powerset describes local-speaker subsets ordered first by cardinality, then
// lexicographically (Python itertools.combinations). It is immutable after
// construction. Speakers are window-local slots, not global speaker identities.
// Bounds of eight slots/256 classes prevent unbounded combinatorial allocation;
// callers must use the actual segmentation checkpoint's declared geometry.
type Powerset struct {
	speakers, maxActive int
	masks               []uint16
}

func NewPowerset(speakers, maxActive int) (*Powerset, error) {
	if speakers < 1 || speakers > 8 || maxActive < 1 || maxActive > speakers {
		return nil, fmt.Errorf("powerset requires 1..8 speakers and 1..speakers active slots")
	}
	p := &Powerset{speakers: speakers, maxActive: maxActive}
	var combinations func(int, int, uint16)
	combinations = func(next, remaining int, mask uint16) {
		if remaining == 0 {
			p.masks = append(p.masks, mask)
			return
		}
		for slot := next; slot <= speakers-remaining; slot++ {
			combinations(slot+1, remaining-1, mask|1<<slot)
		}
	}
	for size := 0; size <= maxActive; size++ {
		combinations(0, size, 0)
	}
	return p, nil
}

func (p *Powerset) Speakers() int {
	if p == nil {
		return 0
	}
	return p.speakers
}
func (p *Powerset) Classes() int {
	if p == nil {
		return 0
	}
	return len(p.masks)
}

// PowersetMode selects the upstream conversion contract.
type PowersetMode uint8

const (
	// PowersetHard selects argmax, resolving ties to the first class. Scores may
	// be finite logits or log probabilities; -Inf denotes an impossible class.
	PowersetHard PowersetMode = iota
	// PowersetSoft sums exp(log_probability) for classes containing each slot.
	// Inputs must be nonpositive log probabilities, with -Inf allowed. There is
	// no softmax/renormalisation; normalised input is the caller's responsibility.
	PowersetSoft
)

// Decode converts frame-major [frames,classes] to owned [frames,speakers].
// It validates dimensions and ALL scores before computing output, rejects
// NaN/+Inf and all-impossible rows, and never returns partial output on error.
// A zero frame count with an empty input is valid. Frames are segmentation
// frames, not PCM samples; no timestamps, thresholds or clustering are applied.
// Cancellation is checked every row. Input must remain immutable for the call.
func (p *Powerset) Decode(ctx context.Context, scores []float32, frames int, mode PowersetMode) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if p == nil || p.speakers < 1 || p.speakers > 8 || len(p.masks) == 0 || frames < 0 || frames > MaxPowersetFrames || len(scores) != frames*len(p.masks) {
		return nil, fmt.Errorf("invalid powerset geometry")
	}
	if mode != PowersetHard && mode != PowersetSoft {
		return nil, fmt.Errorf("invalid powerset decode mode")
	}
	for frame := 0; frame < frames; frame++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		possible := false
		for _, score := range scores[frame*len(p.masks) : (frame+1)*len(p.masks)] {
			if math.IsNaN(float64(score)) || math.IsInf(float64(score), 1) || (mode == PowersetSoft && score > 0) {
				return nil, fmt.Errorf("invalid powerset score at frame %d", frame)
			}
			if !math.IsInf(float64(score), -1) {
				possible = true
			}
		}
		if !possible {
			return nil, fmt.Errorf("all powerset classes impossible at frame %d", frame)
		}
	}
	out := make([]float32, frames*p.speakers)
	for frame := 0; frame < frames; frame++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		row := scores[frame*len(p.masks) : (frame+1)*len(p.masks)]
		dst := out[frame*p.speakers : (frame+1)*p.speakers]
		if mode == PowersetHard {
			best := 0
			for i := 1; i < len(row); i++ {
				if row[i] > row[best] {
					best = i
				}
			}
			for slot := range dst {
				if p.masks[best]&(1<<slot) != 0 {
					dst[slot] = 1
				}
			}
		} else {
			for class, score := range row {
				probability := float32(math.Exp(float64(score)))
				for slot := range dst {
					if p.masks[class]&(1<<slot) != 0 {
						dst[slot] += probability
					}
				}
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// ClassPermutation returns indexes for permuting powerset score columns so
// decodedOutput[:,j] == decodedInput[:,slots[j]] (apart from hard argmax ties,
// which always follow the score-column order). slots must be a permutation of
// 0..Speakers()-1. The returned array is owned; no global assignment is inferred.
func (p *Powerset) ClassPermutation(slots []int) ([]int, error) {
	if p == nil || p.speakers < 1 || len(slots) != p.speakers || len(p.masks) == 0 {
		return nil, fmt.Errorf("invalid powerset permutation geometry")
	}
	seen := uint16(0)
	for _, slot := range slots {
		if slot < 0 || slot >= p.speakers || seen&(1<<slot) != 0 {
			return nil, fmt.Errorf("invalid speaker permutation")
		}
		seen |= 1 << slot
	}
	indexes := make(map[uint16]int, len(p.masks))
	for i, mask := range p.masks {
		indexes[mask] = i
	}
	out := make([]int, len(p.masks))
	for i, mask := range p.masks {
		mapped := uint16(0)
		for slot, source := range slots {
			if mask&(1<<slot) != 0 {
				mapped |= 1 << source
			}
		}
		index, ok := indexes[mapped]
		if !ok {
			return nil, fmt.Errorf("missing permuted powerset class")
		}
		out[i] = index
	}
	return out, nil
}
