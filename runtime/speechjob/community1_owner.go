//go:build linux && amd64

package speechjob

import (
	"context"
	"fmt"
	"sync"

	c1 "github.com/rcarmo/go-pherence/model/speaker/community1"
)

// Community1Status is a detached owner-lifetime snapshot, not model progress.
type Community1Status struct {
	Running, Stopping, Poisoned, Closed bool
	ErrorCode                           string
}

// Community1Owner exclusively owns one pure-Go experimental diarization graph.
// RunPCM is synchronous and uses call-local scratch, so no native/device work can
// survive its return. Copies share state. Transfer all model aliases exclusively
// if deterministic graph-reference release at Close is required.
type Community1Owner struct {
	s     *community1OwnerState
	stage Stage
}
type community1OwnerState struct {
	mu      sync.Mutex
	gate    chan struct{}
	status  Community1Status
	run     communityInfer
	release func()
}

// NewOwnedCommunity1Stage adds serial lifetime ownership around the existing
// whole-result stage. It changes no model mode, tolerance, tie policy or output
// identity and starts no workers/services. The raw NewCommunity1Stage remains.
func NewOwnedCommunity1Stage(model *c1.ExperimentalDiarization, cfg Community1StageConfig) (*Community1Owner, error) {
	if model == nil || model.ValidateOwned() != nil {
		return nil, ErrConfiguration
	}
	s := newCommunity1Owner(func(ctx context.Context, r c1.DiarizationPCMReader, total int64) (*c1.DiarizationPCMResult, error) {
		if cfg.OverlapBranches {
			return model.RunPCMOverlapped(ctx, r, total, cfg.PCM, cfg.SegmentationModes, cfg.EmbeddingMode)
		}
		return model.RunPCM(ctx, r, total, cfg.PCM, cfg.SegmentationModes, cfg.EmbeddingMode)
	}, model.ReleaseOwnedModels)
	if e := validateCommunityConfig(cfg); e != nil {
		return nil, e
	}
	return &Community1Owner{s: s, stage: community1Stage(cfg, s.infer)}, nil
}
func newCommunity1Owner(run communityInfer, release func()) *community1OwnerState {
	return &community1OwnerState{gate: make(chan struct{}, 1), run: run, release: release}
}
func (o *Community1Owner) Stage() Stage {
	if o == nil {
		return Stage{}
	}
	return o.stage
}
func (o *Community1Owner) Status() Community1Status {
	if o == nil || o.s == nil {
		return Community1Status{Closed: true}
	}
	o.s.mu.Lock()
	defer o.s.mu.Unlock()
	return o.s.status
}
func (s *community1OwnerState) acquire(ctx context.Context) error {
	if ctx == nil {
		return ErrConfiguration
	}
	if e := ctx.Err(); e != nil {
		return e
	}
	select {
	case s.gate <- struct{}{}:
		if e := ctx.Err(); e != nil {
			<-s.gate
			return e
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func (s *community1OwnerState) infer(ctx context.Context, r c1.DiarizationPCMReader, total int64) (result *c1.DiarizationPCMResult, err error) {
	if e := s.acquire(ctx); e != nil {
		return nil, e
	}
	defer func() { <-s.gate }()
	s.mu.Lock()
	if s.status.Stopping || s.status.Closed || s.status.Poisoned {
		s.mu.Unlock()
		return nil, ErrClosed
	}
	s.status.Running = true
	s.mu.Unlock()
	defer func() { s.mu.Lock(); s.status.Running = false; s.mu.Unlock() }()
	returned := false
	err = callStage(func() error { var e error; result, e = s.run(ctx, r, total); returned = true; return e })
	if !returned {
		s.mu.Lock()
		s.status.Poisoned = true
		s.status.ErrorCode = "inference_panicked"
		s.mu.Unlock()
		return nil, err
	}
	if err != nil {
		return nil, err
	}
	return result, nil
}

// Close stops admission, waits for a synchronous RunPCM call to return, then
// clears this composition's retained graph references. Timeout preserves them.
// A poisoned owner is still releasable because panic unwound all call-local Go
// scratch; unlike Vulkan there is no native asynchronous submission.
func (o *Community1Owner) Close(ctx context.Context) error {
	if o == nil || o.s == nil {
		return nil
	}
	s := o.s
	s.mu.Lock()
	if s.status.Closed {
		s.mu.Unlock()
		return nil
	}
	s.mu.Unlock()
	if ctx == nil {
		return ErrConfiguration
	}
	if e := ctx.Err(); e != nil {
		return e
	}
	s.mu.Lock()
	if s.status.Closed {
		s.mu.Unlock()
		return nil
	}
	s.status.Stopping = true
	s.mu.Unlock()
	if e := s.acquire(ctx); e != nil {
		return e
	}
	defer func() { <-s.gate }()
	s.mu.Lock()
	if s.status.Closed {
		s.mu.Unlock()
		return nil
	}
	s.mu.Unlock()
	returned := false
	err := callStage(func() error { s.release(); returned = true; return nil })
	if !returned {
		s.mu.Lock()
		s.status.Poisoned = true
		s.status.ErrorCode = "release_panicked"
		s.mu.Unlock()
		return fmt.Errorf("Community-1 owner release: %w", err)
	}
	s.mu.Lock()
	s.status.Closed = true
	s.status.ErrorCode = ""
	s.mu.Unlock()
	return nil
}
