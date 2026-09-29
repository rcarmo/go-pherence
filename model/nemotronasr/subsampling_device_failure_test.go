package nemotronasr

import (
	"context"
	"errors"
	"testing"
)

type failingSubsamplingProjector struct{}

func (failingSubsamplingProjector) Project(context.Context, []float32, []float32, []float32, int) ([]float32, error) {
	return nil, errors.New("projection failed")
}

func TestReleasedSubsamplingDeviceFailurePreservesCache(t *testing.T) {
	model := releasedPCMGenerationModel(t)
	stream := &SubsamplingStream{Model: model.Subsampling, Projector: failingSubsamplingProjector{}}
	if _, err := stream.ForwardUnmaskedChunkContext(context.Background(), make([]float32, 25*128), 25); err == nil {
		t.Fatal("accepted failed device projection")
	}
	if stream.started || stream.mode != 0 {
		t.Fatal("device failure advanced subsampling state")
	}
	for _, previous := range stream.last {
		if len(previous) != 0 {
			t.Fatal("device failure advanced convolution cache")
		}
	}
	request := &PCMGenerationStream{Model: model, Projector: failingSubsamplingProjector{}}
	if _, _, err := request.AppendPCM(context.Background(), make([]float32, 80000)); err == nil || !request.closed {
		t.Fatal("failed device request retained consumed PCM state")
	}
	if _, _, err := request.AppendPCM(context.Background(), []float32{0}); err == nil {
		t.Fatal("accepted append after device failure")
	}
}
