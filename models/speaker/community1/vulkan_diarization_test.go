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
	result     *SegmentationPCMResult
	calls      int
	closeCalls int
	closeErr   error
}

func (f *fakeVulkanDiarizationSegmentation) ForwardPCM(context.Context, []float32, SincNetMode, HeadMode) (*SegmentationPCMResult, error) {
	f.calls++
	if f.result != nil {
		return f.result, nil
	}
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
	result     *WeSpeakerEmbeddingResult
	calls      int
	closeCalls int
}

func (f *fakeVulkanDiarizationEmbedding) Forward(context.Context, []float32, int, []float32, int, int) (*WeSpeakerEmbeddingResult, error) {
	f.calls++
	if f.result != nil {
		return f.result, nil
	}
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
	stop := errors.New("embedding construction failed")
	for _, retained := range []bool{false, true} {
		t.Run(map[bool]string{false: "rollback-complete", true: "rollback-retained"}[retained], func(t *testing.T) {
			seg := &fakeVulkanDiarizationSegmentation{}
			if retained {
				seg.closeErr = io.ErrClosedPipe
			}
			owner, err := newVulkanDiarization(context.Background(), model.segmentation.checkpoint, model.segmentation.frontend.filters, model.embedding.model, nil, 2960, vulkanDiarizationFactories{
				segmentation: func(context.Context, *SegmentationCheckpoint, []float32, int) (vulkanDiarizationSegmentation, error) {
					return seg, nil
				},
				embedding: func(context.Context, *WeSpeakerResNet34, int) (vulkanDiarizationEmbedding, error) { return nil, stop },
			})
			if !errors.Is(err, stop) || seg.closeCalls != 1 {
				t.Fatal(owner, err, seg.closeCalls)
			}
			if !retained {
				if owner != nil {
					t.Fatal("completed rollback returned owner", owner)
				}
				return
			}
			if owner == nil || !errors.Is(err, io.ErrClosedPipe) {
				t.Fatal(owner, err)
			}
			if err := owner.Close(); err != nil || seg.closeCalls != 2 {
				t.Fatal(err, seg.closeCalls)
			}
		})
	}
}

func TestVulkanDiarizationFinalCancellationReturnsNoPartialOutput(t *testing.T) {
	model, cfg, _ := diarizationFixture(t, false)
	grid, err := model.segmentation.checkpoint.Grid(cfg.WindowSamples)
	if err != nil {
		t.Fatal(err)
	}
	local := model.segmentation.checkpoint.cfg.Head.Speakers
	dim := model.embedding.model.cfg.EmbedDim
	scores := make([]float32, grid.Frames*7)
	for frame := 0; frame < grid.Frames; frame++ {
		for class := 1; class < 7; class++ {
			scores[frame*7+class] = -1
		}
	}
	seg := &fakeVulkanDiarizationSegmentation{grid: grid, stats: VulkanSegmentationStats{Frames: grid.Frames, Classes: 7, MaxActive: 2}, result: &SegmentationPCMResult{Grid: grid, Classes: 7, LogProbabilities: scores}}
	emb := &fakeVulkanDiarizationEmbedding{result: &WeSpeakerEmbeddingResult{Embeddings: make([]float32, local*dim), WeightSum: make([]float32, local), NonzeroFrames: make([]int, local)}}
	owner, err := newVulkanDiarization(context.Background(), model.segmentation.checkpoint, model.segmentation.frontend.filters, model.embedding.model, nil, cfg.WindowSamples, vulkanDiarizationFactories{
		segmentation: func(context.Context, *SegmentationCheckpoint, []float32, int) (vulkanDiarizationSegmentation, error) {
			return seg, nil
		},
		embedding: func(context.Context, *WeSpeakerResNet34, int) (vulkanDiarizationEmbedding, error) { return emb, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer owner.Close()
	pcm := sliceDiarizationPCM(make([]float32, cfg.WindowSamples))
	count := newPowersetContext(0)
	result, err := owner.RunPCM(count, pcm, int64(len(pcm)), cfg, SincNetScalarFMA, HeadScalar)
	count.cancel()
	if err != nil || result == nil || count.calls < 1 {
		t.Fatal(result, err, count.calls)
	}
	ctx := newPowersetContext(count.calls)
	result, err = owner.RunPCM(ctx, pcm, int64(len(pcm)), cfg, SincNetScalarFMA, HeadScalar)
	ctx.cancel()
	if result != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("late cancellation returned result", result, err)
	}
}

func TestVulkanDiarizationLateCancellationReturnsNilAfterRollback(t *testing.T) {
	model, _, _ := diarizationFixture(t, false)
	grid, err := model.segmentation.checkpoint.Grid(2960)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	seg := &fakeVulkanDiarizationSegmentation{grid: grid}
	emb := &fakeVulkanDiarizationEmbedding{}
	owner, err := newVulkanDiarization(ctx, model.segmentation.checkpoint, model.segmentation.frontend.filters, model.embedding.model, nil, 2960, vulkanDiarizationFactories{
		segmentation: func(context.Context, *SegmentationCheckpoint, []float32, int) (vulkanDiarizationSegmentation, error) {
			return seg, nil
		},
		embedding: func(context.Context, *WeSpeakerResNet34, int) (vulkanDiarizationEmbedding, error) {
			cancel()
			return emb, nil
		},
	})
	if owner != nil || !errors.Is(err, context.Canceled) || seg.closeCalls != 1 || emb.closeCalls != 1 {
		t.Fatal(owner, err, seg.closeCalls, emb.closeCalls)
	}
}
