package pockettts

import (
	"strings"
	"testing"
)

func TestMimiEncoderRejectsOverlapBeforeMutation(t *testing.T) {
	m := &MimiEncoderCPU{FrameSize: SamplesPerFrame, Downsample: CausalConv1D{Out: 2}}
	workspace := &MimiEncoderWorkspace{
		encoder:     m,
		MaxFrames:   1,
		ChunkFrames: 1,
		ConvScratch: &Scratch{values: make([]float32, 1), used: 1},
		Initial:     StreamingConvState{Previous: []float32{7}, First: false},
	}
	assertRejected := func(name string, out, audio []float32) {
		t.Helper()
		before := append([]float32(nil), out...)
		err := m.EncodeInto(out, audio, workspace)
		if err == nil || !strings.Contains(err.Error(), "overlap") {
			t.Fatalf("%s: error=%v", name, err)
		}
		assertSliceClose(t, name+" output", out, before, 0)
		if workspace.ConvScratch.used != 1 || workspace.Initial.Previous[0] != 7 || workspace.Initial.First {
			t.Fatalf("%s mutated workspace", name)
		}
	}

	shared := make([]float32, SamplesPerFrame+2)
	assertRejected("input/output", shared[1:3], shared[:SamplesPerFrame])

	audio, out := make([]float32, SamplesPerFrame), []float32{3, 4}
	workspace.A = audio
	assertRejected("input/workspace", out, audio)
	workspace.A = out
	assertRejected("output/workspace", out, audio)
}

func TestMimiEncoderWorkspaceUsesLargestIntermediate(t *testing.T) {
	conv := func(in, out, kernel, stride int) CausalConv1D {
		return CausalConv1D{In: in, Out: out, Kernel: kernel, Stride: stride, Dilation: 1}
	}
	stage := func(in, out, stride int) seanetEncoderStageCPU {
		return seanetEncoderStageCPU{
			Residual: seanetResidualCPU{Conv1: conv(in, in, 1, 1), Conv2: conv(in, in, 1, 1)},
			Down:     conv(in, out, 2*stride, stride),
		}
	}
	m := &MimiEncoderCPU{
		Initial:    conv(1, 1, 1, 1),
		Stages:     []seanetEncoderStageCPU{stage(1, 64, 2), stage(64, 2, 5), stage(2, 16, 12)},
		Final:      conv(16, 16, 1, 1),
		Downsample: conv(16, 2, 32, 16),
		Transformer: &TransformerCPU{
			Width: 16, Heads: 1, HeadDim: 16, MaxPeriod: 10_000,
			Layers: []TransformerLayerCPU{{FC1: LinearF32{Out: 32}}},
		},
		FrameSize: SamplesPerFrame,
	}
	workspace, err := m.NewWorkspace(1)
	if err != nil {
		t.Fatal(err)
	}
	const largest = 64 * (SamplesPerFrame / 2)
	if len(workspace.A) != largest || len(workspace.B) != largest || len(workspace.Residual) != largest {
		t.Fatalf("intermediate capacities A=%d B=%d residual=%d want=%d", len(workspace.A), len(workspace.B), len(workspace.Residual), largest)
	}

	m.Transformer.Width--
	if _, err := m.NewWorkspace(1); err == nil {
		t.Fatal("accepted transformer/final projection mismatch")
	}
}
