// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT
// Binary activity to intervals follows pinned pyannote Binarize (NOTICE).
package community1

import (
	"context"
	"fmt"
	"math"
	"sort"
)

// SpeakerTurn is a canonical-timeline half-open interval labelled by cluster
// column. These are not source-media PTS, reference speaker names or word times.
type SpeakerTurn struct {
	Start, End float64
	Speaker    int
}

// BinaryTurnConfig controls the zero-padding/onset=.5/offset=.5 Binarize subset.
// Frame centres use 0.5*(frameStart + frameEnd), preserving source order.
// Starts0..14400, frame step[1e-6,1], duration[step,1], min durations0..30.
// Activity [Frames,Speakers], frames>=2 <=1e6, speakers1..64, elements<=2^24.
// Fewer than two frames fail explicitly (upstream has an undefined final t).
// At most100000 output turns; the bounded final sort is synchronous.
type BinaryTurnConfig struct {
	Frames, Speakers                                               int
	Start, FrameDuration, FrameStep, MinDurationOn, MinDurationOff float64
}

// BinaryActivityToTurns follows source binary state changes at frame CENTRES.
// A final active run ends at the LAST centre, not last centre+step or window
// end; a one-frame final activation becomes empty and is dropped. Segments of
// duration<=1e-6 are empty. Positive min-off fills same-speaker gaps strictly
// shorter than the collar (or gaps<=1e-6), then min-on removes short runs.
// Setting min-off can create overlap even from exclusive discrete activity;
// disjointness is guaranteed only at the discrete grid / zero-collar policy.
// No clipping to file duration, relabelling, caption alignment or source-time
// conversion. Returned turns are owned and sorted by start/end/speaker. The
// caller must qualify the grid before emitting production speaker-labelled VTT.
func BinaryActivityToTurns(ctx context.Context, activity []uint8, cfg BinaryTurnConfig) ([]SpeakerTurn, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c := cfg
	if c.Frames < 2 || c.Frames > 1e6 || c.Speakers < 1 || c.Speakers > 64 || int64(c.Frames)*int64(c.Speakers) > 1<<24 || len(activity) != c.Frames*c.Speakers {
		return nil, fmt.Errorf("invalid binary turn geometry")
	}
	for _, v := range []float64{c.Start, c.FrameDuration, c.FrameStep, c.MinDurationOn, c.MinDurationOff} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("nonfinite binary turn timing")
		}
	}
	if c.Start < 0 || c.Start > 14400 || c.FrameStep < 1e-6 || c.FrameStep > 1 || c.FrameDuration < c.FrameStep || c.FrameDuration > 1 || c.MinDurationOn < 0 || c.MinDurationOn > 30 || c.MinDurationOff < 0 || c.MinDurationOff > 30 || c.Start+float64(c.Frames-1)*c.FrameStep+c.FrameDuration > 14431 {
		return nil, fmt.Errorf("unsupported binary turn timing")
	}
	for i, value := range activity {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if value > 1 {
			return nil, fmt.Errorf("activity must be binary")
		}
	}
	centre := func(frame int) float64 {
		start := c.Start + float64(frame)*c.FrameStep
		end := start + c.FrameDuration
		return .5 * (start + end)
	}
	out := make([]SpeakerTurn, 0)
	appendTurn := func(turn SpeakerTurn) error {
		if len(out) >= 100000 {
			return fmt.Errorf("binary turn output limit")
		}
		out = append(out, turn)
		return nil
	}
	for speaker := 0; speaker < c.Speakers; speaker++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		// Keep only one pending interval per speaker for same-speaker gap filling.
		var pending SpeakerTurn
		hasPending := false
		emit := func(start, end float64) error {
			if end-start <= 1e-6 {
				return nil
			}
			next := SpeakerTurn{start, end, speaker}
			if hasPending && c.MinDurationOff > 0 && (start-pending.End <= 1e-6 || start-pending.End < c.MinDurationOff) {
				pending.End = end
				return nil
			}
			if hasPending && pending.End-pending.Start >= c.MinDurationOn {
				if err := appendTurn(pending); err != nil {
					return err
				}
			}
			pending = next
			hasPending = true
			return nil
		}
		active := activity[speaker] == 1
		start := centre(0)
		for frame := 1; frame < c.Frames; frame++ {
			if frame%256 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			value := activity[frame*c.Speakers+speaker]
			time := centre(frame)
			if active && value == 0 {
				if err := emit(start, time); err != nil {
					return nil, err
				}
				active = false
			} else if !active && value == 1 {
				start = time
				active = true
			}
		}
		if active {
			if err := emit(start, centre(c.Frames-1)); err != nil {
				return nil, err
			}
		}
		if hasPending && pending.End-pending.Start >= c.MinDurationOn {
			if err := appendTurn(pending); err != nil {
				return nil, err
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Start != out[j].Start {
			return out[i].Start < out[j].Start
		}
		if out[i].End != out[j].End {
			return out[i].End < out[j].End
		}
		return out[i].Speaker < out[j].Speaker
	})
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
