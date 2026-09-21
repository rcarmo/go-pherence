// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT
// Mask selection and clustering admission follow the references in NOTICE.
package community1

import (
	"context"
	"fmt"
	"math"
)

// EmbeddingMaskConfig describes one binary segmentation window. Input layout
// is [Frames,Speakers]. Sample counts refer to the same canonical PCM window,
// including its padding, NOT source-media timestamps. MinimumSamples must come
// from a separately qualified embedding configuration; no model is probed here.
// Bounds: frames1..4096, speakers1..8, both sample counts1..480000.
type EmbeddingMaskConfig struct {
	Frames, Speakers              int
	WindowSamples, MinimumSamples int
	ExcludeOverlap                bool
}

// EmbeddingMasks contains owned speaker-major [Speakers,Frames] masks for
// WeSpeaker Forward/ForwardEmbedding. MinimumCleanFrames is -1 when exclusion
// is disabled. UsedOverlapExcluded is true only when the clean mask was chosen.
// SelectedFrames counts segmentation frames, NOT post-resize feature support,
// samples, minimum-duration admission, or global speaker identities.
type EmbeddingMasks struct {
	Masks               []float32
	MinimumCleanFrames  int
	UsedOverlapExcluded []bool
	SelectedFrames      []int
}

// SelectEmbeddingMasks reproduces SpeakerDiarization.get_embeddings selection:
// use overlap-free speech only if its count is STRICTLY GREATER than
// ceil(Frames*MinimumSamples/WindowSamples); otherwise use all speech. Bounds
// make the integer product safe on 32-bit architectures. Exclusion never drops
// overlapping frames when clean support is equal to the minimum.
//
// Inputs must be binary 0/1 or NaN (unknown/partial stitching); Inf/soft masks
// fail. A frame with any NaN is not clean for ANY speaker. Only after clean-mask
// selection arithmetic are NaNs replaced with zero, matching the source order.
// This chooses masks; it does not admit embeddings for clustering. Empty masks
// remain empty even though the raw projection can return finite bias for them.
// No source mutation, retained state, CNN work, or source-time mapping occurs.
func SelectEmbeddingMasks(ctx context.Context, segmentations []float32, cfg EmbeddingMaskConfig) (*EmbeddingMasks, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if cfg.Frames < 1 || cfg.Frames > 4096 || cfg.Speakers < 1 || cfg.Speakers > 8 || cfg.WindowSamples < 1 || cfg.WindowSamples > 480000 || cfg.MinimumSamples < 1 || cfg.MinimumSamples > 480000 || len(segmentations) != cfg.Frames*cfg.Speakers {
		return nil, fmt.Errorf("invalid embedding mask geometry")
	}
	if err := checkBinarySegmentations(ctx, segmentations); err != nil {
		return nil, err
	}
	minimum := -1
	if cfg.ExcludeOverlap {
		product := int64(cfg.Frames) * int64(cfg.MinimumSamples)
		minimum = int((product + int64(cfg.WindowSamples) - 1) / int64(cfg.WindowSamples))
	}
	result := &EmbeddingMasks{Masks: make([]float32, len(segmentations)), MinimumCleanFrames: minimum, UsedOverlapExcluded: make([]bool, cfg.Speakers), SelectedFrames: make([]int, cfg.Speakers)}
	var cleanCounts [8]int
	for frame := 0; frame < cfg.Frames; frame++ {
		if frame%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		row := segmentations[frame*cfg.Speakers : (frame+1)*cfg.Speakers]
		active := float32(0)
		for _, value := range row {
			active += value
		}
		if active < 2 {
			for speaker, value := range row {
				if value == 1 {
					result.Masks[speaker*cfg.Frames+frame] = 1
					cleanCounts[speaker]++
				}
			}
		}
	}
	for speaker := 0; speaker < cfg.Speakers; speaker++ {
		useClean := cfg.ExcludeOverlap && cleanCounts[speaker] > minimum
		result.UsedOverlapExcluded[speaker] = useClean
		if useClean {
			result.SelectedFrames[speaker] = cleanCounts[speaker]
			continue
		}
		for frame := 0; frame < cfg.Frames; frame++ {
			if frame%256 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			value := segmentations[frame*cfg.Speakers+speaker]
			if math.IsNaN(float64(value)) {
				value = 0
			}
			result.Masks[speaker*cfg.Frames+frame] = value
			result.SelectedFrames[speaker] += int(value)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

// ClusteringFilterConfig describes embeddings [Chunks,Speakers,Dimension] and
// binary segmentations [Chunks,Frames,Speakers]. Bounds: chunks/frames1..4096,
// speakers1..8, dimension1..512; each input has at most 2^24 elements.
// MinActiveRatio is explicit, finite and in [0,1]; the upstream default is 0.2.
// Zero deliberately allows zero clean support, just as upstream. Production
// callers must bind this policy to a validated pipeline configuration.
type ClusteringFilterConfig struct {
	Chunks, Frames, Speakers, Dimension int
	MinActiveRatio                      float64
}

// ClusteringEmbeddings holds owned selected rows in stable chunk/speaker order.
// Indices refer to the original window-local speakers, not global identities.
// Empty admission returns zero-length slices. Rejected rows can still be used
// later for centroid assignment, as in the reference; this is a training filter.
type ClusteringEmbeddings struct {
	Embeddings                   []float32
	ChunkIndices, SpeakerIndices []int
}

// FilterClusteringEmbeddings reproduces BaseClustering.filter_embeddings for
// binary/NaN masks and finite/NaN embeddings: count ONLY single-speaker frames,
// admit counts >= float32(MinActiveRatio*Frames), and exclude any NaN embedding
// row. The float32 threshold matches NumPy comparison against float32 counts.
// A NaN segmentation poisons that speaker's full-window clean count even on
// non-clean frames (NaN*0 remains NaN). Other speakers lose only that frame.
// Unlike the raw reference, Inf embeddings/segmentations and soft masks fail
// explicitly before selection. No cosine normalisation or clustering occurs.
// With the default 0.2 ratio, empty/overlap-only masks are never training rows,
// even when the raw embedding is finite projection bias.
func FilterClusteringEmbeddings(ctx context.Context, embeddings, segmentations []float32, cfg ClusteringFilterConfig) (*ClusteringEmbeddings, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c := cfg
	if c.Chunks < 1 || c.Chunks > 4096 || c.Frames < 1 || c.Frames > 4096 || c.Speakers < 1 || c.Speakers > 8 || c.Dimension < 1 || c.Dimension > 512 || math.IsNaN(c.MinActiveRatio) || math.IsInf(c.MinActiveRatio, 0) || c.MinActiveRatio < 0 || c.MinActiveRatio > 1 {
		return nil, fmt.Errorf("invalid clustering filter geometry/policy")
	}
	rows := c.Chunks * c.Speakers
	if rows*c.Frames > 1<<24 || rows*c.Dimension > 1<<24 || len(embeddings) != rows*c.Dimension || len(segmentations) != rows*c.Frames {
		return nil, fmt.Errorf("invalid clustering filter input length/bound")
	}
	if err := checkBinarySegmentations(ctx, segmentations); err != nil {
		return nil, err
	}
	for i, value := range embeddings {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("infinite clustering embedding")
		}
	}
	result := &ClusteringEmbeddings{ChunkIndices: make([]int, 0, rows), SpeakerIndices: make([]int, 0, rows)}
	threshold := float32(c.MinActiveRatio * float64(c.Frames))
	for chunk := 0; chunk < c.Chunks; chunk++ {
		var counts [8]int
		var unknown [8]bool
		for frame := 0; frame < c.Frames; frame++ {
			if frame%256 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			row := segmentations[(chunk*c.Frames+frame)*c.Speakers : (chunk*c.Frames+frame+1)*c.Speakers]
			active := float32(0)
			for speaker, value := range row {
				active += value
				if math.IsNaN(float64(value)) {
					unknown[speaker] = true
				}
			}
			if active == 1 {
				for speaker, value := range row {
					counts[speaker] += int(value)
				}
			}
		}
		for speaker := 0; speaker < c.Speakers; speaker++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if unknown[speaker] || float32(counts[speaker]) < threshold {
				continue
			}
			valid := true
			for _, value := range embeddings[(chunk*c.Speakers+speaker)*c.Dimension : (chunk*c.Speakers+speaker+1)*c.Dimension] {
				if math.IsNaN(float64(value)) {
					valid = false
					break
				}
			}
			if valid {
				result.ChunkIndices = append(result.ChunkIndices, chunk)
				result.SpeakerIndices = append(result.SpeakerIndices, speaker)
			}
		}
	}
	result.Embeddings = make([]float32, len(result.ChunkIndices)*c.Dimension)
	for i, chunk := range result.ChunkIndices {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		offset := (chunk*c.Speakers + result.SpeakerIndices[i]) * c.Dimension
		copy(result.Embeddings[i*c.Dimension:(i+1)*c.Dimension], embeddings[offset:offset+c.Dimension])
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

func checkBinarySegmentations(ctx context.Context, values []float32) error {
	for i, value := range values {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if value != 0 && value != 1 && !math.IsNaN(float64(value)) {
			return fmt.Errorf("segmentations must be binary or NaN")
		}
	}
	return ctx.Err()
}
