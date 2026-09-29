package nemotrondiarization

import (
	"context"
	"errors"
	"testing"
)

type failingStackProjector struct{}

func (failingStackProjector) Project(context.Context, []float32, []float32, int) ([]float32, error) {
	return nil, errors.New("projection failed")
}

func TestStackingStreamDeviceFailureClosesConsumedState(t *testing.T) {
	projection := &StackingProjection{weight: make([]float32, projectedWidth*stackWidth)}
	stream := &StackingStream{Projection: projection, Device: failingStackProjector{}, ctx: context.Background()}
	input := make([]float32, stackWidth)
	if _, err := stream.AppendFeatures(input); err == nil || !stream.closed {
		t.Fatal("device failure left consumed group replayable")
	}
	if _, err := stream.AppendFeatures(input); err == nil {
		t.Fatal("accepted append after projection failure")
	}
	stream = &StackingStream{Projection: projection, Device: failingStackProjector{}, ctx: context.Background()}
	if _, err := stream.AppendFeatures(input[:melBins]); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Finish(); err == nil || !stream.closed {
		t.Fatal("device failure left terminal group replayable")
	}
	if _, err := stream.Finish(); err == nil {
		t.Fatal("accepted finish after projection failure")
	}
}
