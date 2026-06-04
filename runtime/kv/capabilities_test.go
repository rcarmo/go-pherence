package kv

import (
	"runtime"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func TestRuntimeTurboQuantCapabilitiesMirrorSIMD(t *testing.T) {
	got := RuntimeTurboQuantCapabilities()
	caps := simd.RuntimeCapabilities()
	if got.Arch != runtime.GOARCH || got.Arch != caps.Arch {
		t.Fatalf("arch mismatch got=%q runtime=%q caps=%q", got.Arch, runtime.GOARCH, caps.Arch)
	}
	if got.Rotation != caps.HasDot || got.Vec != caps.HasVec || got.AVX2 != caps.HasAVX2 || got.NEON != caps.HasNEON || got.RVV != caps.HasRVV {
		t.Fatalf("capability mismatch got=%+v simd=%+v", got, caps)
	}
}
