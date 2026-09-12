// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT
// Community-1 postprocessing order follows pyannote sources in NOTICE.
package community1

import (
	"context"
	"errors"
	"fmt"
	"math"
)

var (
	// ErrNoTrainingEmbeddings means speech was detected but no valid, sufficiently
	// clean embedding row was admitted. Upstream averages an empty array to NaNs;
	// this checked API deliberately fails rather than inventing a centroid.
	ErrNoTrainingEmbeddings = errors.New("speech detected without training embeddings")
	// ErrSpeakerCountFallbackRequired means source would invoke KMeans. That
	// fallback is not implemented here; no silent speaker-count substitution.
	ErrSpeakerCountFallbackRequired = errors.New("speaker-count KMeans fallback required")
)

// PostprocessConfig is explicit inference/postprocessing policy. Geometry and
// tie rules come from Reconstruction. EmbeddingDimension1..512. MinSpeakers
// is1..64; NumSpeakers=0 means automatic, otherwise1..64 overrides both bounds
// as upstream. Reconstruction.MaxSpeakers is otherwise the upper bound.
// AHCThreshold>=0, Fa/Fb>0, finite; MinDurationOff in[0,30]. Training admission
// uses the source's default clean-speech ratio0.2. VBx uses20iterations/epsilon
// 1e-4/smoothing7. No hidden config parsing or reference/model inference.
type PostprocessConfig struct {
	Reconstruction                               ReconstructionConfig
	EmbeddingDimension, MinSpeakers, NumSpeakers int
	AHCThreshold, Fa, Fb, MinDurationOff         float64
	Constrained                                  bool
}

// PostprocessResult owns all arrays, with local hard labels after inactive-row
// suppression. Centroid rows retain cluster-index order; no SPEAKER_XX naming
// or centroid/annotation-label reordering. Timeline may pad extra class columns
// when counts exceed surviving centroids. ConstraintSatisfied concerns centroid
// count; a one-training-row source fallback can return false without KMeans.
// Path is "silence", "single-training-row" or "clustered". Silence returns
// zero centroids, -2 hard labels, an empty activity width and empty turn lists.
type PostprocessResult struct {
	Path                                            string
	TrainingRows, Clusters                          int
	ConstraintSatisfied                             bool
	TrainingChunks, TrainingSpeakers, InitialLabels []int
	Centroids, SoftScores                           []float64
	HardLabels                                      []int
	Timeline                                        *ActivityTimeline
	FullTurns, ExclusiveTurns                       []SpeakerTurn
}

// PostprocessObserver is a synchronous stage notification, not a partial result.
// Error aborts and is preserved; context is checked before/after the callback.
// Stages on the main path: count, filter, ahc, plda, vbx, centroids, assignment,
// reconstruction, turns. Silence exits after count; single-row skips ahc..vbx.
type PostprocessObserver func(stage string) error

func PostprocessCommunity1(ctx context.Context, segmentations, embeddings []float32, plda *PreparedPLDA, cfg PostprocessConfig) (*PostprocessResult, error) {
	return PostprocessCommunity1Observed(ctx, segmentations, embeddings, plda, cfg, nil)
}

// PostprocessCommunity1Observed connects previously checked components without
// running any neural/frontend/media code. Inputs are binary/NaN segmentation
// [Chunks,Frames,Speakers] and embeddings [Chunks,Speakers,EmbeddingDimension].
// Embeddings are widened exactly from float32 into the float64 clustering path;
// this is NOT bit-exact float32 NumPy-normalisation parity. NaN embeddings may
// be filtered, but on the multirow path ALL original rows must have finite
// nonzero cosine norms, including inactive rows, as required by checked
// AssignCosineSpeakers. Undefined cosine is an error, not hidden imputation.
// Silence accepts nil embeddings/model; non-silent calls require full shape.
//
// Caps: <=512 admitted AHC rows and <=64 initial slots, plus the existing
// component memory/work/timing bounds. Full/exclusive turns use frame centres,
// no source-PTS mapping/clipping/naming. Positive gap filling can overlap even
// exclusive intervals. No PCM entry point, services, GPU, KMeans or quality
// claim. Inputs/model must remain immutable; no partial result escapes errors.
func PostprocessCommunity1Observed(ctx context.Context, segmentations, embeddings []float32, plda *PreparedPLDA, cfg PostprocessConfig, observe PostprocessObserver) (*PostprocessResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c := cfg.Reconstruction
	minSpeakers, maxSpeakers := cfg.MinSpeakers, c.MaxSpeakers
	if cfg.EmbeddingDimension < 1 || cfg.EmbeddingDimension > 512 || minSpeakers < 1 || minSpeakers > 64 || maxSpeakers < 1 || maxSpeakers > 64 || cfg.NumSpeakers < 0 || cfg.NumSpeakers > 64 {
		return nil, fmt.Errorf("invalid postprocess dimension/count policy")
	}
	if cfg.NumSpeakers > 0 {
		minSpeakers, maxSpeakers = cfg.NumSpeakers, cfg.NumSpeakers
	}
	if minSpeakers > maxSpeakers {
		return nil, fmt.Errorf("invalid postprocess speaker range")
	}
	c.MaxSpeakers = maxSpeakers
	for _, value := range []float64{cfg.AHCThreshold, cfg.Fa, cfg.Fb, cfg.MinDurationOff} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("nonfinite postprocess policy")
		}
	}
	if cfg.AHCThreshold < 0 || cfg.Fa <= 0 || cfg.Fb <= 0 || cfg.MinDurationOff < 0 || cfg.MinDurationOff > 30 {
		return nil, fmt.Errorf("invalid postprocess numerical policy")
	}
	starts, frames, err := reconstructionGrid(c)
	if err != nil {
		return nil, err
	}
	if len(segmentations) != c.Chunks*c.Frames*c.Speakers {
		return nil, fmt.Errorf("postprocess segmentation length")
	}
	if err := checkBinarySegmentations(ctx, segmentations); err != nil {
		return nil, err
	}
	notify := func(stage string) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if observe != nil {
			if err := observe(stage); err != nil {
				return err
			}
		}
		return ctx.Err()
	}
	counts, err := powersetFrameCounts(ctx, segmentations, c, starts, frames)
	if err != nil {
		return nil, err
	}
	if err := notify("count"); err != nil {
		return nil, err
	}
	result := &PostprocessResult{HardLabels: make([]int, c.Chunks*c.Speakers), FullTurns: []SpeakerTurn{}, ExclusiveTurns: []SpeakerTurn{}}
	for i := range result.HardLabels {
		result.HardLabels[i] = -2
	}
	speaking := false
	for i, count := range counts {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if count > 0 {
			speaking = true
			break
		}
	}
	if !speaking {
		result.Path = "silence"
		result.Centroids = []float64{}
		result.SoftScores = []float64{}
		result.Timeline = &ActivityTimeline{Frames: frames, Start: c.Start, FrameDuration: c.FrameDuration, FrameStep: c.FrameStep, Counts: counts, Activations: []float32{}, Full: []uint8{}, Exclusive: []uint8{}}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return result, nil
	}
	rows := c.Chunks * c.Speakers
	dim := cfg.EmbeddingDimension
	if rows*dim > 1<<24 || len(embeddings) != rows*dim {
		return nil, fmt.Errorf("postprocess embedding length/bound")
	}
	train, err := FilterClusteringEmbeddings(ctx, embeddings, segmentations, ClusteringFilterConfig{c.Chunks, c.Frames, c.Speakers, dim, .2})
	if err != nil {
		return nil, err
	}
	result.TrainingRows = len(train.ChunkIndices)
	result.TrainingChunks = train.ChunkIndices
	result.TrainingSpeakers = train.SpeakerIndices
	if err := notify("filter"); err != nil {
		return nil, err
	}
	if result.TrainingRows == 0 {
		return nil, ErrNoTrainingEmbeddings
	}
	if result.TrainingRows > 512 {
		return nil, fmt.Errorf("postprocess AHC training row bound")
	}
	training, err := widenPostprocess(ctx, train.Embeddings)
	if err != nil {
		return nil, err
	}
	if result.TrainingRows == 1 {
		result.Path = "single-training-row"
		result.Clusters = 1
		result.Centroids = training
		if err := notify("centroids"); err != nil {
			return nil, err
		}
		// Reference returns zero hard labels and unit scores before inactive removal;
		// requested counts do not force KMeans on this early fallback.
		result.SoftScores = make([]float64, rows)
		for i := range result.HardLabels {
			result.HardLabels[i] = 0
			result.SoftScores[i] = 1
		}
	} else {
		result.Path = "clustered"
		if plda == nil || plda.cfg.InputDim != dim || plda.cfg.OutputDim < 1 {
			return nil, fmt.Errorf("postprocess PLDA dimension/model")
		}
		initial, err := CentroidAHC(ctx, training, CentroidAHCConfig{result.TrainingRows, dim, cfg.AHCThreshold})
		if err != nil {
			return nil, err
		}
		if initial.Clusters > 64 {
			return nil, fmt.Errorf("postprocess initial VBx slot bound")
		}
		result.InitialLabels = initial.Labels
		if err := notify("ahc"); err != nil {
			return nil, err
		}
		features, err := plda.Transform(ctx, training, result.TrainingRows)
		if err != nil {
			return nil, err
		}
		if err := notify("plda"); err != nil {
			return nil, err
		}
		policy := Community1VBxConfig(result.TrainingRows, plda.cfg.OutputDim)
		policy.Fa, policy.Fb = cfg.Fa, cfg.Fb
		vb, err := ClusterVBx(ctx, features, plda.Phi(), initial.Labels, policy)
		if err != nil {
			return nil, err
		}
		if err := notify("vbx"); err != nil {
			return nil, err
		}
		centers, err := ComputeVBxCentroids(ctx, training, vb.Responsibilities, vb.Priors, VBxCentroidConfig{result.TrainingRows, dim, vb.Speakers})
		if err != nil {
			return nil, err
		}
		result.Centroids = centers.Centroids
		result.Clusters = len(centers.SpeakerIndices)
		if err := notify("centroids"); err != nil {
			return nil, err
		}
		if result.Clusters < minSpeakers || result.Clusters > maxSpeakers {
			return nil, fmt.Errorf("%w: automatic=%d requested=%d..%d", ErrSpeakerCountFallbackRequired, result.Clusters, minSpeakers, maxSpeakers)
		}
		originals, err := widenPostprocess(ctx, embeddings)
		if err != nil {
			return nil, err
		}
		assignment, err := AssignCosineSpeakers(ctx, originals, result.Centroids, segmentations, CosineAssignmentConfig{c.Chunks, c.Speakers, result.Clusters, dim, c.Frames, cfg.Constrained})
		if err != nil {
			return nil, err
		}
		result.HardLabels = assignment.Labels
		result.SoftScores = assignment.Scores
	}
	if err := notify("assignment"); err != nil {
		return nil, err
	}
	result.ConstraintSatisfied = result.Clusters >= minSpeakers && result.Clusters <= maxSpeakers
	// This is required even on the one-training-row shortcut. Keep source's NaN
	// semantics: NaN sum is not zero, hence it is not an inactive-speaker signal.
	for chunk := 0; chunk < c.Chunks; chunk++ {
		var sum [8]float32
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
			if sum[s] == 0 {
				result.HardLabels[chunk*c.Speakers+s] = -2
			}
		}
	}
	// Reuse the already-computed counts; equivalent to ReconstructPowerset without
	// a second scan/allocation for counts and inactive labels.
	classes := 0
	for _, label := range result.HardLabels {
		classes = max(classes, label+1)
	}
	for i, count := range counts {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		classes = max(classes, count)
	}
	timeline := &ActivityTimeline{Frames: frames, Start: c.Start, FrameDuration: c.FrameDuration, FrameStep: c.FrameStep, Counts: counts}
	result.Timeline, err = reconstructActivity(ctx, segmentations, result.HardLabels, c, starts, timeline, classes)
	if err != nil {
		return nil, err
	}
	if err := notify("reconstruction"); err != nil {
		return nil, err
	}
	if result.Timeline.Classes > 0 {
		turns := BinaryTurnConfig{frames, result.Timeline.Classes, c.Start, c.FrameDuration, c.FrameStep, 0, cfg.MinDurationOff}
		result.FullTurns, err = BinaryActivityToTurns(ctx, result.Timeline.Full, turns)
		if err != nil {
			return nil, err
		}
		result.ExclusiveTurns, err = BinaryActivityToTurns(ctx, result.Timeline.Exclusive, turns)
		if err != nil {
			return nil, err
		}
	}
	if err := notify("turns"); err != nil {
		return nil, err
	}
	return result, nil
}
func widenPostprocess(ctx context.Context, values []float32) ([]float64, error) {
	out := make([]float64, len(values))
	for i, v := range values {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		out[i] = float64(v)
	}
	return out, ctx.Err()
}
