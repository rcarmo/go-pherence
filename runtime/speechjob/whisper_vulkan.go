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
	"github.com/rcarmo/go-pherence/model/whisper"
)

// VulkanWhisperStageConfig opts into the existing resident F32 encoder/host
// decoder, never an automatic CPU fallback. BackendSHA256 attests immutable
// device/driver/shader/kernel/precision identity; it is part of checkpoint keys.
// DrainPoll bounds EACH wait, not total shutdown. Device loss/uncertainty holds
// the active job and admission until process teardown. No forced idle is inferred.
type VulkanWhisperStageConfig struct {
	Whisper           WhisperStageConfig
	AllowExperimental bool
	BackendSHA256     string
	DrainPoll         time.Duration
}

// VulkanWhisperStatus is a detached owner-lifetime snapshot, not progress/RSS.
type VulkanWhisperStatus struct {
	Running, Draining, Quarantined, Stopping, Closed bool
	ErrorCode                                        string
}

// VulkanWhisperStage owns exclusive stage access and encoder teardown after a
// successful constructor. The caller still owns the host model/tokenizer and
// global Vulkan context; neither may be closed/mutated while this owner is live.
// Do not use the encoder through other references or run other global Vulkan
// clients concurrently. Copies share state. No VulkanInit, models or work start.
type VulkanWhisperStage struct {
	s      *vulkanWhisperStageState
	stages []Stage
}
type vulkanWhisperStageState struct {
	mu             sync.Mutex
	gate           chan struct{}
	status         VulkanWhisperStatus
	drain          func(context.Context, time.Duration) error
	closeEncoder   func() error
	poll           time.Duration
	quarantineHold chan struct{} // never signalled by production; process teardown only
}

func NewVulkanWhisperWindowStage(model *whisper.Whisper, tokenizer *whisper.Tokenizer, encoder *whisper.VulkanEncoder, cfg VulkanWhisperStageConfig) (*VulkanWhisperStage, error) {
	return NewVulkanWhisperWindowStages(model, tokenizer, encoder, []VulkanWhisperStageConfig{cfg})
}

// NewVulkanWhisperWindowStages binds several immutable language profiles to one
// serial resident encoder owner. All configs must attest the same backend and
// drain policy. The returned owner closes the encoder exactly once.
func NewVulkanWhisperWindowStages(model *whisper.Whisper, tokenizer *whisper.Tokenizer, encoder *whisper.VulkanEncoder, cfgs []VulkanWhisperStageConfig) (*VulkanWhisperStage, error) {
	if len(cfgs) < 1 || len(cfgs) > 32 || encoder == nil {
		return nil, ErrConfiguration
	}
	first := cfgs[0]
	if !first.AllowExperimental || !validHash(first.BackendSHA256) || first.DrainPoll < time.Millisecond || first.DrainPoll > 30*time.Second {
		return nil, ErrConfiguration
	}
	validate := func() error { return model.ValidatePCMVulkanHostDecoder(context.Background(), encoder) }
	if e := validate(); e != nil {
		return nil, e
	}
	identity, e := json.Marshal(struct {
		Backend string
		Stats   whisper.VulkanEncoderStats
	}{first.BackendSHA256, encoder.Stats()})
	if e != nil {
		return nil, e
	}
	s := newVulkanWhisperOwner(first.DrainPoll, vk.VulkanDrain, encoder.Close)
	binding := &residentStageBinding{identity: hash(identity), validate: validate, encoder: encoder, wrap: s.wrap}
	stages := make([]Stage, 0, len(cfgs))
	for _, cfg := range cfgs {
		if !cfg.AllowExperimental || cfg.BackendSHA256 != first.BackendSHA256 || cfg.DrainPoll != first.DrainPoll {
			return nil, ErrConfiguration
		}
		st, err := newWhisperWindowStage(model, tokenizer, cfg.Whisper, binding)
		if err != nil {
			return nil, err
		}
		stages = append(stages, st)
	}
	return &VulkanWhisperStage{s: s, stages: stages}, nil
}
func newVulkanWhisperOwner(poll time.Duration, drain func(context.Context, time.Duration) error, closeEncoder func() error) *vulkanWhisperStageState {
	return &vulkanWhisperStageState{gate: make(chan struct{}, 1), drain: drain, closeEncoder: closeEncoder, poll: poll, quarantineHold: make(chan struct{})}
}

// Stage returns a copy whose closure retains this owner. Keep the owner handle
// for Status/Close. Its version differs from host stages even for identical text.
func (o *VulkanWhisperStage) Stage() Stage {
	if o == nil || len(o.stages) == 0 {
		return Stage{}
	}
	return o.stages[0]
}

// Stages returns copies of all language-specific stages owned by this encoder.
func (o *VulkanWhisperStage) Stages() []Stage {
	if o == nil {
		return nil
	}
	return append([]Stage(nil), o.stages...)
}
func (o *VulkanWhisperStage) Status() VulkanWhisperStatus {
	if o == nil || o.s == nil {
		return VulkanWhisperStatus{Closed: true}
	}
	o.s.mu.Lock()
	defer o.s.mu.Unlock()
	return o.s.status
}
func (s *vulkanWhisperStageState) acquire(ctx context.Context) error {
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
func (s *vulkanWhisperStageState) quarantine(code string) {
	s.mu.Lock()
	s.status.Quarantined = true
	s.status.Draining = true
	s.status.ErrorCode = code
	s.mu.Unlock()
	// Returning would release Store.Run/HTTP/queue resource admission while native
	// work may survive. No API can declare a lost/uncertain device safe in-process.
	<-s.quarantineHold
}
func (s *vulkanWhisperStageState) settle(cause error) {
	if errors.Is(cause, vk.ErrVulkanDeviceLost) || errors.Is(cause, vk.ErrVulkanUncertain) {
		s.quarantine("device_quarantined")
		return
	}
	s.mu.Lock()
	s.status.Draining = true
	s.mu.Unlock()
	for {
		started := time.Now()
		panicked := true
		e := callStage(func() error { e := s.drain(context.Background(), s.poll); panicked = false; return e })
		if panicked {
			s.quarantine("drain_panicked")
			return
		}
		if e == nil {
			s.mu.Lock()
			s.status.Draining = false
			s.mu.Unlock()
			return
		}
		// Retry only an explicitly bounded context timeout/cancellation. A fatal
		// driver/helper error can be joined with ErrVulkanInFlight; that is still
		// quarantine, never an unbounded retry loop. Fresh nil proves idle.
		if errors.Is(e, vk.ErrVulkanUncertain) || errors.Is(e, vk.ErrVulkanDeviceLost) {
			s.quarantine("device_quarantined")
			return
		}
		if !errors.Is(e, context.DeadlineExceeded) && !errors.Is(e, context.Canceled) {
			s.quarantine("drain_failed")
			return
		}
		// Protect against immediate-return mocks/drivers without spinning. Request
		// cancellation intentionally cannot end native ownership retention.
		if delay := s.poll - time.Since(started); delay > 0 {
			time.Sleep(delay)
		}
	}
}
func (s *vulkanWhisperStageState) wrap(infer windowInfer) windowInfer {
	return func(ctx context.Context, source whisper.SampleReader, total, first int64, emit func(whisper.WindowTranscript) error) (err error) {
		if e := s.acquire(ctx); e != nil {
			return e
		}
		defer func() { <-s.gate }()
		s.mu.Lock()
		if s.status.Stopping || s.status.Closed {
			s.mu.Unlock()
			return ErrClosed
		}
		s.status.Running = true
		s.mu.Unlock()
		defer func() { s.mu.Lock(); s.status.Running = false; s.mu.Unlock() }()
		// Contain a panic here, before the generic executor can drop native ownership.
		returned := false
		err = callStage(func() error { e := infer(ctx, source, total, first, emit); returned = true; return e })
		if !returned {
			s.quarantine("inference_panicked")
			return err
		}
		s.settle(err)
		return err
	}
}

// Close permanently stops new inference, waits for this owner's active work,
// then closes the resident encoder. Timeout leaves all ownership intact. Failed
// encoder teardown is retryable; this method never closes the global device or
// host decoder. Quarantined active work can only end with process teardown.
func (o *VulkanWhisperStage) Close(ctx context.Context) error {
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
	s.mu.Lock()
	s.status.Stopping = true
	s.mu.Unlock()
	if e := s.acquire(ctx); e != nil {
		return e
	}
	defer func() { <-s.gate }()
	s.mu.Lock()
	closed := s.status.Closed
	s.mu.Unlock()
	if closed {
		return nil
	}
	returned := false
	e := callStage(func() error { e := s.closeEncoder(); returned = true; return e })
	if !returned {
		s.quarantine("close_panicked")
		return e
	}
	if e != nil {
		s.mu.Lock()
		s.status.ErrorCode = "close_failed"
		s.mu.Unlock()
		return fmt.Errorf("Vulkan Whisper encoder close: %w", e)
	}
	s.mu.Lock()
	s.status.Closed = true
	s.status.ErrorCode = ""
	s.mu.Unlock()
	return nil
}
