package nemotrondiarization

import (
	"context"
	"fmt"
	"math"
)

// StackingStream holds at most seven unprojected mel rows for one stream.
// AppendFeatures emits only complete groups of eight; Finish emits a padded
// final group. Returned projected embeddings are owned by the caller.
type StackingStream struct {
	Projection *StackingProjection
	Device     StackingDeviceProjector // optional per-request GPU projection
	ctx        context.Context
	pending    [stackWidth]float32
	frames     int
	closed     bool
}

// AppendFeatures accepts at most five seconds of 128-bin features per call.
// Invalid input leaves pending stream state unchanged. Projection errors
// close the stream because a consumed eight-frame group cannot be replayed.
func (s *StackingStream) AppendFeatures(features []float32) ([]float32, error) {
	if s == nil || s.Projection == nil || len(s.Projection.weight) != projectedWidth*stackWidth || s.closed || len(features) == 0 || len(features)%melBins != 0 || len(features)/melBins > 500 || s.frames < 0 || s.frames >= stackFrames {
		return nil, fmt.Errorf("invalid Nemotron streaming stacking features")
	}
	for _, value := range features {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron streaming stacking features")
		}
	}
	rows := len(features) / melBins
	groups := (s.frames + rows) / stackFrames
	stacked := make([]float32, 0, groups*stackWidth)
	for i := 0; i < rows; i++ {
		copy(s.pending[s.frames*melBins:(s.frames+1)*melBins], features[i*melBins:(i+1)*melBins])
		s.frames++
		if s.frames == stackFrames {
			stacked = append(stacked, s.pending[:]...)
			s.frames = 0
		}
	}
	if groups == 0 {
		return nil, nil
	}
	var projected []float32
	var err error
	if s.Device != nil {
		projected, err = s.Projection.ProjectWithDevice(s.ctx, stacked, groups*stackFrames, s.Device)
	} else {
		projected, err = s.Projection.Project(stacked, groups*stackFrames)
	}
	if err != nil {
		s.closed = true // pending was consumed; retry cannot replay this group
		return nil, err
	}
	return projected, nil
}

// Finish may be called once. An exact group boundary has no padded row.
func (s *StackingStream) Finish() ([]float32, error) {
	if s == nil || s.Projection == nil || len(s.Projection.weight) != projectedWidth*stackWidth || s.closed || s.frames < 0 || s.frames >= stackFrames {
		return nil, fmt.Errorf("invalid Nemotron streaming stacking finish")
	}
	var out []float32
	if s.frames > 0 {
		clear(s.pending[s.frames*melBins:])
		var projected []float32
		var err error
		if s.Device != nil {
			projected, err = s.Projection.ProjectWithDevice(s.ctx, s.pending[:], stackFrames, s.Device)
		} else {
			projected, err = s.Projection.Project(s.pending[:], stackFrames)
		}
		if err != nil {
			s.closed = true // padded terminal group cannot be replayed
			return nil, err
		}
		out = projected
	}
	s.frames = 0
	s.closed = true
	return out, nil
}
