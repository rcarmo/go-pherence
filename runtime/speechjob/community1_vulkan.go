//go:build linux && amd64

package speechjob

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	c1 "github.com/rcarmo/go-pherence/models/speaker/community1"
)

// VulkanCommunity1StageConfig explicitly binds the Community-1 model identity
// to a Vulkan backend/device identity. It never selects a fallback. DrainPoll
// bounds each proof-of-idle attempt after any inference return.
type VulkanCommunity1StageConfig struct {
	Community         Community1StageConfig
	AllowExperimental bool
	BackendSHA256     string
	DeviceIdentity    string
	DrainPoll         time.Duration
}

// VulkanCommunity1Status is a detached lifetime snapshot, not inference progress.
type VulkanCommunity1Status struct {
	Running, Draining, Quarantined, Stopping, Closed bool
	ErrorCode                                        string
}

// VulkanCommunity1Stage exclusively owns one hybrid model. The caller owns the
// process-global Vulkan context and must not run another Vulkan owner concurrently.
// Cancellation never releases the stage/resource lease until a fresh-context
// drain proves all submitted native work complete. Uncertain/device-lost state
// blocks forever for process-level recovery rather than falling back to CPU.
type VulkanCommunity1Stage struct {
	s     *vulkanCommunity1State
	stage Stage
}

type vulkanCommunity1State struct {
	mu             sync.Mutex
	gate           chan struct{}
	status         VulkanCommunity1Status
	run            communityInfer
	drain          func(context.Context, time.Duration) error
	closeModel     func() error
	poll           time.Duration
	quarantineHold chan struct{}
}

func NewVulkanCommunity1Stage(model *c1.VulkanDiarization, cfg VulkanCommunity1StageConfig) (*VulkanCommunity1Stage, error) {
	if model == nil || model.Stats().WindowSamples != cfg.Community.PCM.WindowSamples || !cfg.AllowExperimental || !validHash(cfg.BackendSHA256) || cfg.DeviceIdentity == "" || len(cfg.DeviceIdentity) > 256 || cfg.DrainPoll < time.Millisecond || cfg.DrainPoll > 30*time.Second {
		return nil, ErrConfiguration
	}
	if err := validateCommunityConfig(cfg.Community); err != nil {
		return nil, err
	}
	if cfg.Community.SegmentationModes.LSTM != c1.LSTMSIMD || cfg.Community.EmbeddingMode != c1.WeSpeakerBlockGEMM {
		return nil, ErrConfiguration
	}
	identity, err := json.Marshal(struct {
		Backend, Device string
		Stats           c1.VulkanDiarizationStats
	}{cfg.BackendSHA256, cfg.DeviceIdentity, model.Stats()})
	if err != nil {
		return nil, err
	}
	stageCfg := cfg.Community
	stageCfg.ExecutionBackendSHA256 = hash(identity)
	s := newVulkanCommunity1Owner(cfg.DrainPoll, func(ctx context.Context, r c1.DiarizationPCMReader, total int64) (*c1.DiarizationPCMResult, error) {
		return model.RunPCM(ctx, r, total, stageCfg.PCM, stageCfg.SegmentationModes.SincNet, stageCfg.SegmentationModes.Head)
	}, vk.VulkanDrain, model.Close)
	return &VulkanCommunity1Stage{s: s, stage: community1Stage(stageCfg, s.infer)}, nil
}

func newVulkanCommunity1Owner(poll time.Duration, run communityInfer, drain func(context.Context, time.Duration) error, closeModel func() error) *vulkanCommunity1State {
	return &vulkanCommunity1State{gate: make(chan struct{}, 1), run: run, drain: drain, closeModel: closeModel, poll: poll, quarantineHold: make(chan struct{})}
}

func (o *VulkanCommunity1Stage) Stage() Stage {
	if o == nil {
		return Stage{}
	}
	return o.stage
}
func (o *VulkanCommunity1Stage) Status() VulkanCommunity1Status {
	if o == nil || o.s == nil {
		return VulkanCommunity1Status{Closed: true}
	}
	o.s.mu.Lock()
	defer o.s.mu.Unlock()
	return o.s.status
}
func (s *vulkanCommunity1State) acquire(ctx context.Context) error {
	if ctx == nil {
		return ErrConfiguration
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case s.gate <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-s.gate
			return err
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func (s *vulkanCommunity1State) quarantine(code string) {
	s.mu.Lock()
	s.status.Quarantined = true
	s.status.Draining = true
	s.status.ErrorCode = code
	s.mu.Unlock()
	<-s.quarantineHold
}
func (s *vulkanCommunity1State) settle(cause error) {
	if errors.Is(cause, vk.ErrVulkanDeviceLost) || errors.Is(cause, vk.ErrVulkanUncertain) {
		s.quarantine("device_quarantined")
		return
	}
	s.mu.Lock()
	s.status.Draining = true
	s.mu.Unlock()
	for {
		started := time.Now()
		returned := false
		err := callStage(func() error { e := s.drain(context.Background(), s.poll); returned = true; return e })
		if !returned {
			s.quarantine("drain_panicked")
			return
		}
		if err == nil {
			s.mu.Lock()
			s.status.Draining = false
			s.mu.Unlock()
			return
		}
		if errors.Is(err, vk.ErrVulkanUncertain) || errors.Is(err, vk.ErrVulkanDeviceLost) {
			s.quarantine("device_quarantined")
			return
		}
		if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
			s.quarantine("drain_failed")
			return
		}
		if delay := s.poll - time.Since(started); delay > 0 {
			time.Sleep(delay)
		}
	}
}
func (s *vulkanCommunity1State) infer(ctx context.Context, reader c1.DiarizationPCMReader, total int64) (result *c1.DiarizationPCMResult, err error) {
	if err := s.acquire(ctx); err != nil {
		return nil, err
	}
	defer func() { <-s.gate }()
	s.mu.Lock()
	if s.status.Stopping || s.status.Closed {
		s.mu.Unlock()
		return nil, ErrClosed
	}
	s.status.Running = true
	s.mu.Unlock()
	defer func() { s.mu.Lock(); s.status.Running = false; s.mu.Unlock() }()
	returned := false
	err = callStage(func() error { var e error; result, e = s.run(ctx, reader, total); returned = true; return e })
	if !returned {
		s.quarantine("inference_panicked")
		return nil, err
	}
	s.settle(err)
	return result, err
}

// Close stops admission, waits for active inference/drain, and retries model
// closure without discarding unresolved resources. It never closes the global
// Vulkan device. A quarantined call can end only through process teardown.
func (o *VulkanCommunity1Stage) Close(ctx context.Context) error {
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
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	s.status.Stopping = true
	s.mu.Unlock()
	if err := s.acquire(ctx); err != nil {
		return err
	}
	defer func() { <-s.gate }()
	s.mu.Lock()
	closed := s.status.Closed
	s.mu.Unlock()
	if closed {
		return nil
	}
	returned := false
	err := callStage(func() error { e := s.closeModel(); returned = true; return e })
	if !returned {
		s.quarantine("close_panicked")
		return err
	}
	if err != nil {
		s.mu.Lock()
		s.status.ErrorCode = "close_failed"
		s.mu.Unlock()
		return fmt.Errorf("Vulkan Community-1 model close: %w", err)
	}
	s.mu.Lock()
	s.status.Closed = true
	s.status.ErrorCode = ""
	s.mu.Unlock()
	return nil
}
