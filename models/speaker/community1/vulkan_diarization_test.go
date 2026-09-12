package community1

import (
	"context"
	"errors"
	"io"
	"reflect"
	"testing"
)

type fakeVulkanDiarizationSegmentation struct {
	grid       SincNetGrid
	stats      VulkanSegmentationStats
	calls      int
	closeCalls int
	closeErr   error
}

func (f *fakeVulkanDiarizationSegmentation) ForwardPCM(context.Context, []float32, SincNetMode, HeadMode) (*SegmentationPCMResult, error) {
	f.calls++
	return &SegmentationPCMResult{Grid: f.grid, Classes: f.stats.Classes, LogProbabilities: []float32{0, -1}}, nil
}
func (f *fakeVulkanDiarizationSegmentation) Close() error {
	f.closeCalls++
	if f.closeErr != nil && f.closeCalls == 1 {
		return f.closeErr
	}
	return nil
}
func (f *fakeVulkanDiarizationSegmentation) Stats() VulkanSegmentationStats { return f.stats }
func (f *fakeVulkanDiarizationSegmentation) Grid() SincNetGrid              { return f.grid }

type fakeVulkanDiarizationEmbedding struct {
	stats      VulkanEmbeddingStats
	calls      int
	closeCalls int
}

func (f *fakeVulkanDiarizationEmbedding) Forward(context.Context, []float32, int, []float32, int, int) (*WeSpeakerEmbeddingResult, error) {
	f.calls++
	return &WeSpeakerEmbeddingResult{Embeddings: []float32{1}, WeightSum: []float32{1}, NonzeroFrames: []int{1}}, nil
}
func (f *fakeVulkanDiarizationEmbedding) Close() error                { f.closeCalls++; return nil }
func (f *fakeVulkanDiarizationEmbedding) Stats() VulkanEmbeddingStats { return f.stats }

type sliceDiarizationPCM []float32

func (p sliceDiarizationPCM) ReadSamplesAt(_ context.Context, dst []float32, off int64) (int, error) {
	if off < 0 || off >= int64(len(p)) {
		return 0, io.EOF
	}
	n := copy(dst, p[off:])
	if int(off)+n == len(p) {
		return n, io.EOF
	}
	return n, nil
}

func TestVulkanDiarizationInjectedLifecycleAndRun(t *testing.T) {
	model, cfg, _ := diarizationFixture(t, false)
	segSource := model.segmentation.checkpoint
	embSource := model.embedding.model
	grid, err := segSource.Grid(2960)
	if err != nil {
		t.Fatal(err)
	}
	seg := &fakeVulkanDiarizationSegmentation{grid: grid, stats: VulkanSegmentationStats{Frames: grid.Frames, Classes: 7, MaxActive: 2}}
	emb := &fakeVulkanDiarizationEmbedding{}
	calls := []string{}
	owner, err := newVulkanDiarization(context.Background(), segSource, model.segmentation.frontend.filters, embSource, nil, 2960, vulkanDiarizationFactories{
		segmentation: func(context.Context, *SegmentationCheckpoint, []float32, int) (vulkanDiarizationSegmentation, error) {
			calls = append(calls, "seg")
			return seg, nil
		},
		embedding: func(context.Context, *WeSpeakerResNet34, int) (vulkanDiarizationEmbedding, error) {
			calls = append(calls, "emb")
			return emb, nil
		},
	})
	if err != nil || !reflect.DeepEqual(calls, []string{"seg", "emb"}) || owner.Stats().WindowSamples != 2960 {
		t.Fatal(owner, err, calls)
	}
	// A mismatched fixed-window policy is rejected before either native child.
	cfg.WindowSamples++
	if result, err := owner.RunPCM(context.Background(), sliceDiarizationPCM(make([]float32, 2960)), 2960, cfg, SincNetScalarFMA, HeadScalar); result != nil || err == nil || seg.calls != 0 || emb.calls != 0 {
		t.Fatal(result, err, seg.calls, emb.calls)
	}
	if err := owner.Close(); err != nil || seg.closeCalls != 1 || emb.closeCalls != 1 {
		t.Fatal(err, seg.closeCalls, emb.closeCalls)
	}
	if err := owner.Close(); err != nil || seg.closeCalls != 1 || emb.closeCalls != 1 {
		t.Fatal("non-idempotent close", err)
	}
}

func TestVulkanDiarizationRollbackAndRetry(t *testing.T) {
	model, _, _ := diarizationFixture(t, false)
	seg := &fakeVulkanDiarizationSegmentation{closeErr: io.ErrClosedPipe}
	stop := errors.New("embedding construction failed")
	owner, err := newVulkanDiarization(context.Background(), model.segmentation.checkpoint, model.segmentation.frontend.filters, model.embedding.model, nil, 2960, vulkanDiarizationFactories{
		segmentation: func(context.Context, *SegmentationCheckpoint, []float32, int) (vulkanDiarizationSegmentation, error) {
			return seg, nil
		},
		embedding: func(context.Context, *WeSpeakerResNet34, int) (vulkanDiarizationEmbedding, error) { return nil, stop },
	})
	if owner == nil || !errors.Is(err, stop) || !errors.Is(err, io.ErrClosedPipe) || seg.closeCalls != 1 {
		t.Fatal(owner, err, seg.closeCalls)
	}
	if err := owner.Close(); err != nil || seg.closeCalls != 2 {
		t.Fatal(err, seg.closeCalls)
	}
}
