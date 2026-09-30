package nemotronasr

import (
	"context"
	"testing"
)

func TestPCMGenerationLanguagePromptValidation(t *testing.T) {
	model := &PCMGenerationModel{Subsampling: &Subsampling{}, Tower: &OfflineEncoderTower{}, Projection: &RNNTProjection{}, Decoder: &RNNTDecoder{}}
	for _, p := range []int{0, 13, 101, 127} {
		prompt := p
		s := &PCMGenerationStream{Model: model, PromptID: &prompt}
		if err := s.validate(context.Background()); err != nil {
			t.Fatal(p, err)
		}
	}
	for _, p := range []int{-1, 128} {
		prompt := p
		s := &PCMGenerationStream{Model: model, PromptID: &prompt}
		if _, _, err := s.AppendPCM(context.Background(), make([]float32, 80000)); err == nil {
			t.Fatal("invalid prompt accepted")
		}
		if s.closed || s.frontend.first || len(s.frontend.pending) != 0 {
			t.Fatal("validation mutated stream")
		}
	}
	s := &PCMGenerationStream{Model: model}
	if err := s.validate(context.Background()); err != nil {
		t.Fatal("nil default prompt rejected", err)
	}
}
