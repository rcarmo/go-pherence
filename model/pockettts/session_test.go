package pockettts

import (
	"context"
	"errors"
	"testing"
)

func TestNilSessionRejectsGeneration(t *testing.T) {
	var session *Session
	noise := func(int, []float32) error { return nil }
	if _, err := session.GenerateInto(nil, nil, 1, 0, 1, 0, noise); err == nil {
		t.Fatal("accepted nil session")
	}
	if _, err := session.GenerateIntoContext(context.Background(), nil, nil, 1, 0, 1, 0, noise); err == nil {
		t.Fatal("accepted nil context session")
	}
	if _, err := session.GenerateIntoContext(nil, nil, nil, 1, 0, 1, 0, noise); err == nil {
		t.Fatal("accepted nil context")
	}
}

func TestCanceledSessionRejectsBeforeReset(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s := &Session{Generator: &GeneratorCPU{}, MaxFrames: 1, FlowState: &TransformerState{Layers: []TransformerLayerState{{Capacity: 4}}}, VoiceTemplate: &TransformerState{}}
	pcm := make([]float32, SamplesPerFrame)
	pcm[0] = 7
	count, err := s.GenerateIntoContext(ctx, pcm, []uint32{1}, 1, 0, 1, 0, func(int, []float32) error { return nil })
	if count != 0 || !errors.Is(err, context.Canceled) || pcm[0] != 7 {
		t.Fatalf("pre-canceled count=%d err=%v first=%g", count, err, pcm[0])
	}
}
