// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT
// Powerset reconstruction/aggregation follows pyannote sources in NOTICE.
package community1

import (
	"context"
	"fmt"
	"math"
)

// ReconstructionTiePolicy is explicit because NumPy's default argsort has
// ISA/version-dependent non-stable tie ordering. RejectAmbiguousTies (zero)
// refuses a frame whose cutoff splits equal scores. LowestIndexTies is a
// deterministic alternative, NOT claimed to match upstream tie identities.
type ReconstructionTiePolicy int

const (
	RejectAmbiguousTies ReconstructionTiePolicy = iota
	LowestIndexTies
)

// ReconstructionConfig describes powerset binary/NaN segmentation windows
// [Chunks,Frames,Speakers] and labels [Chunks,Speakers]. Geometry is seconds on
// the canonical PCM timeline; no source-container PTS/edit-list mapping here.
// FrameStart is deliberately absent: upstream aggregation resets it to Start.
// Hamming/warmup disabled, matching Community-1 apply's (0,0) warmup.
// Bounds: chunks1..4096, frames1..4096, speakers1..8, start0..14400,
// chunk duration/step in(0,30], frame duration/step in[1e-6,1], duration>=step,
// whole extent<=14430s, <=1e6 output frames, <=2^24 input/output elements,
// and outputFrames*classes*classes<=2^28 bounds the per-frame sorting work.
// MaxSpeakers1..64 caps rounded counts; -2 labels are discarded. Other labels
// must be0..63. Gap classes are preserved like max(labels)+1, not compacted.
type ReconstructionConfig struct {
	Chunks, Frames, Speakers                                  int
	Start, ChunkDuration, ChunkStep, FrameDuration, FrameStep float64
	MaxSpeakers                                               int
	TiePolicy                                                 ReconstructionTiePolicy
}

// ActivityTimeline owns overlap-add activations [Frames,Classes], counts and
// full/exclusive binary activity on the same frame grid. Counts are rounded
// float32 observation averages (nearest/even), capped by MaxSpeakers. Activity
// scores are SUMS, not averages; missing observations contribute zero. Classes
// can be padded above max label to fit counts, exactly as source to_diarization.
// No normalization/permutation/speaker naming happens here. AmbiguousFrames
// reports union of full/exclusive cutoff ties resolved by LowestIndexTies.
// A silent/all-discarded input can have Classes=0, with empty activity buffers.
type ActivityTimeline struct {
	Frames, Classes                 int
	Start, FrameDuration, FrameStep float64
	Counts                          []int
	Activations                     []float32
	Full, Exclusive                 []uint8
	AmbiguousFrames                 []int
}

// ReconstructPowerset matches the bounded binary powerset/no-warmup subset:
// ignore inactive local speakers (sum==0), max local scores for shared cluster,
// overlap-add at closest-frame rounded starts, separately average local counts,
// select top count / top min(count,1). NaN in a local count removes that window
// observation; NaN in a shared-cluster max removes that cluster observation.
// Input values must be0/1/NaN, not soft/Inf. No thresholded soft segmentation,
// interval smoothing, crop regridding, neural inference or hidden runtime.
// Source input mutation is NOT reproduced: all sources stay immutable.
func ReconstructPowerset(ctx context.Context, segmentations []float32, labels []int, cfg ReconstructionConfig) (*ActivityTimeline, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	starts, total, err := reconstructionGrid(cfg)
	if err != nil {
		return nil, err
	}
	c := cfg
	if len(segmentations) != c.Chunks*c.Frames*c.Speakers || len(labels) != c.Chunks*c.Speakers {
		return nil, fmt.Errorf("invalid reconstruction input length")
	}
	if err := checkBinarySegmentations(ctx, segmentations); err != nil {
		return nil, err
	}
	effective := append([]int(nil), labels...)
	classes := 0
	for chunk := 0; chunk < c.Chunks; chunk++ {
		var sum [8]float32
		for s := 0; s < c.Speakers; s++ {
			label := labels[chunk*c.Speakers+s]
			if label < -2 || label == -1 || label > 63 {
				return nil, fmt.Errorf("invalid reconstruction label")
			}
		}
		for frame := 0; frame < c.Frames; frame++ {
			if frame%256 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			for s := 0; s < c.Speakers; s++ {
				sum[s] += segmentations[(chunk*c.Frames+frame)*c.Speakers+s]
			}
		}
		for s := 0; s < c.Speakers; s++ {
			index := chunk*c.Speakers + s
			if sum[s] == 0 {
				effective[index] = -2
			}
			classes = max(classes, effective[index]+1)
		}
	}
	// Count aggregation is independent of clustering. Missing values are omitted,
	// not zero observations in the average denominator.
	sums := make([]float32, total)
	observations := make([]int, total)
	for chunk, start := range starts {
		for frame := 0; frame < c.Frames; frame++ {
			if frame%256 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			sum := float32(0)
			for s := 0; s < c.Speakers; s++ {
				sum += segmentations[(chunk*c.Frames+frame)*c.Speakers+s]
			}
			if !math.IsNaN(float64(sum)) {
				sums[start+frame] += sum
				observations[start+frame]++
			}
		}
	}
	result := &ActivityTimeline{Frames: total, Start: c.Start, FrameDuration: c.FrameDuration, FrameStep: c.FrameStep, Counts: make([]int, total)}
	for i := range sums {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if observations[i] > 0 {
			result.Counts[i] = min(c.MaxSpeakers, int(math.RoundToEven(float64(sums[i]/float32(observations[i])))))
		}
		classes = max(classes, result.Counts[i])
	}
	if int64(total)*int64(classes) > 1<<24 || int64(total)*int64(classes)*int64(classes) > 1<<28 {
		return nil, fmt.Errorf("reconstruction output element bound")
	}
	result.Classes = classes
	result.Activations = make([]float32, total*classes)
	result.Full = make([]uint8, total*classes)
	result.Exclusive = make([]uint8, total*classes)
	for chunk, start := range starts {
		for frame := 0; frame < c.Frames; frame++ {
			if frame%256 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			var maximum [64]float32
			var seen, unknown [64]bool
			for s := 0; s < c.Speakers; s++ {
				k := effective[chunk*c.Speakers+s]
				if k < 0 {
					continue
				}
				value := segmentations[(chunk*c.Frames+frame)*c.Speakers+s]
				seen[k] = true
				if math.IsNaN(float64(value)) {
					unknown[k] = true
				} else {
					maximum[k] = max(maximum[k], value)
				}
			}
			for k := 0; k < classes; k++ {
				if seen[k] && !unknown[k] {
					result.Activations[(start+frame)*classes+k] += maximum[k]
				}
			}
		}
	}
	var order [64]int
	for frame, count := range result.Counts {
		if frame%128 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if count == 0 {
			continue
		}
		row := result.Activations[frame*classes : (frame+1)*classes]
		for k := 0; k < classes; k++ {
			order[k] = k
		}
		// Stable insertion sort on at most64 indices: no per-frame closure or
		// reflection allocation, and ties retain increasing original index.
		for i := 1; i < classes; i++ {
			key, j := order[i], i
			for j > 0 && row[order[j-1]] < row[key] {
				order[j] = order[j-1]
				j--
			}
			order[j] = key
		}
		fullTie := count < classes && row[order[count-1]] == row[order[count]]
		exclusiveTie := classes > 1 && row[order[0]] == row[order[1]]
		if fullTie || exclusiveTie {
			if c.TiePolicy == RejectAmbiguousTies {
				return nil, fmt.Errorf("ambiguous reconstruction cutoff tie at frame %d", frame)
			}
			result.AmbiguousFrames = append(result.AmbiguousFrames, frame)
		}
		for i := 0; i < count; i++ {
			result.Full[frame*classes+order[i]] = 1
		}
		result.Exclusive[frame*classes+order[0]] = 1
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}
func reconstructionGrid(c ReconstructionConfig) ([]int, int, error) {
	if c.Chunks < 1 || c.Chunks > 4096 || c.Frames < 1 || c.Frames > 4096 || c.Speakers < 1 || c.Speakers > 8 || int64(c.Chunks)*int64(c.Frames)*int64(c.Speakers) > 1<<24 || c.MaxSpeakers < 1 || c.MaxSpeakers > 64 || (c.TiePolicy != RejectAmbiguousTies && c.TiePolicy != LowestIndexTies) {
		return nil, 0, fmt.Errorf("invalid reconstruction geometry/policy")
	}
	for _, v := range []float64{c.Start, c.ChunkDuration, c.ChunkStep, c.FrameDuration, c.FrameStep} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, 0, fmt.Errorf("nonfinite reconstruction geometry")
		}
	}
	if c.Start < 0 || c.Start > 14400 || c.ChunkDuration <= 0 || c.ChunkDuration > 30 || c.ChunkStep <= 0 || c.ChunkStep > 30 || c.FrameDuration < 1e-6 || c.FrameDuration > 1 || c.FrameStep < 1e-6 || c.FrameStep > c.FrameDuration {
		return nil, 0, fmt.Errorf("unsupported reconstruction timing")
	}
	// Do not algebraically simplify these expressions: source float64 rounding
	// and nearest/even boundary placement are part of this component's contract.
	end := c.Start + c.ChunkDuration + float64(c.Chunks-1)*c.ChunkStep
	if end > 14430 {
		return nil, 0, fmt.Errorf("reconstruction extent bound")
	}
	totalFloat := math.RoundToEven(((end+.5*c.FrameDuration)-c.Start-.5*c.FrameDuration)/c.FrameStep) + 1
	if totalFloat < 1 || totalFloat > 1e6 {
		return nil, 0, fmt.Errorf("reconstruction frame bound")
	}
	total := int(totalFloat)
	starts := make([]int, c.Chunks)
	for chunk := range starts {
		time := c.Start + float64(chunk)*c.ChunkStep
		value := math.RoundToEven(((time + .5*c.FrameDuration) - c.Start - .5*c.FrameDuration) / c.FrameStep)
		if value < 0 || value > float64(total) || value+float64(c.Frames) > float64(total) {
			return nil, 0, fmt.Errorf("reconstruction window exceeds output grid")
		}
		starts[chunk] = int(value)
	}
	return starts, total, nil
}
