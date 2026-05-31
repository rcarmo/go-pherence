package lfm2

import (
	"errors"
	"testing"
)

func TestNotImplementedRuntimeContracts(t *testing.T) {
	rt := NewNotImplementedRuntime()
	if _, err := rt.Embed(RuntimeRequestPlan{}, nil); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("embed err=%v", err)
	}
	if _, err := rt.ForwardConv(RuntimeRequestPlan{}, nil); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("conv err=%v", err)
	}
	if _, err := rt.ForwardAttention(RuntimeRequestPlan{}, nil); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("attention err=%v", err)
	}
	if _, err := rt.ForwardMoE(RuntimeRequestPlan{}, nil); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("moe err=%v", err)
	}
	if _, err := rt.Generate(RuntimeRequestPlan{}); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("generate err=%v", err)
	}
}
