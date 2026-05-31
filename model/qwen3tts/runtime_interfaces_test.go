package qwen3tts

import (
	"errors"
	"testing"
)

func TestNotImplementedRuntimeContracts(t *testing.T) {
	rt := NewNotImplementedRuntime()
	if _, err := rt.ForwardSemantic(RuntimeRequestPlan{}); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("talker err=%v", err)
	}
	if _, err := rt.PredictAcoustic(RuntimeRequestPlan{}, nil); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("code predictor err=%v", err)
	}
	if _, err := rt.DecodeWaveform(RuntimeRequestPlan{}, nil); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("decoder err=%v", err)
	}
}
