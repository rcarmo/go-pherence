//go:build linux && amd64

package speechjob

import (
	"context"
	"errors"
	"io"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/models/whisper"
	"github.com/rcarmo/go-pherence/runtime/resourcebudget"
)

func mockVulkanJob(poll time.Duration, drain func(context.Context, time.Duration) error, closeEncoder func() error, infer windowInfer) *VulkanWhisperStage {
	s := newVulkanWhisperOwner(poll, drain, closeEncoder)
	st := whisperWindowStage(hash([]byte("mock-vulkan-job-v1")), 320, 160, 4096, 128<<10, 448, 51865, s.wrap(infer))
	return &VulkanWhisperStage{s: s, stage: st}
}
func waitOwner(t *testing.T, o *VulkanWhisperStage, predicate func(VulkanWhisperStatus) bool) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if predicate(o.Status()) {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("owner state", o.Status())
}
func TestVulkanJobCancellationRetainsAdmissionAndStore(t *testing.T) {
	s, _ := openTest(t)
	job := createTest(t, s)
	b, e := resourcebudget.New(resourcebudget.Config{Capacity: resourcebudget.Resources{CPUSlots: 2, MemoryBytes: 100}, MaxActive: 1, MaxWaiting: 1})
	if e != nil {
		t.Fatal(e)
	}
	defer b.Close()
	admit, _ := b.Admission(resourcebudget.Resources{CPUSlots: 2, MemoryBytes: 100})
	entered, draining, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()
	var closes atomic.Int32
	owner := mockVulkanJob(time.Millisecond, func(ctx context.Context, _ time.Duration) error {
		if ctx.Err() != nil {
			t.Error("drain inherited cancelled request")
		}
		close(draining)
		<-release
		return nil
	}, func() error { closes.Add(1); return nil }, func(ctx context.Context, _ whisper.SampleReader, _, _ int64, _ func(whisper.WindowTranscript) error) error {
		close(entered)
		<-ctx.Done()
		return errors.Join(ctx.Err(), vk.ErrVulkanInFlight)
	})
	q := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), admit, func(Manifest) ([]Stage, error) { return []Stage{fixturePCMStage(1001), owner.Stage()}, nil })
	q.Enqueue(context.Background(), job.ID, false)
	q.Start(context.Background())
	<-entered
	q.Cancel(context.Background(), job.ID)
	<-draining
	if !owner.Status().Draining || b.Snapshot().Active != 1 {
		t.Fatal(owner.Status(), b.Snapshot())
	}
	if e := s.Close(); !errors.Is(e, ErrBusy) {
		t.Fatal("store released during native drain", e)
	}
	if e := s.Delete(context.Background(), job.ID); !errors.Is(e, ErrBusy) {
		t.Fatal("media deleted during drain", e)
	}
	timeout, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if e := q.Shutdown(timeout); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
	if e := owner.Close(timeout); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
	if closes.Load() != 0 {
		t.Fatal("encoder closed before fence proof")
	}
	close(release)
	q.Shutdown(context.Background())
	entries, e := q.List()
	if e != nil || entries[0].Status != QueueCancelled {
		t.Fatal(entries, e)
	}
	if b.Snapshot().Active != 0 {
		t.Fatal("lease not drained", b.Snapshot())
	}
	if e = owner.Close(context.Background()); e != nil || closes.Load() != 1 {
		t.Fatal(e, closes.Load())
	}
}
func TestVulkanJobJournalResumeAfterDrain(t *testing.T) {
	s, _ := openTest(t)
	job := createTest(t, s)
	var starts []int64
	var calls int
	var polls int
	infer := func(ctx context.Context, r whisper.SampleReader, total, first int64, emit func(whisper.WindowTranscript) error) error {
		calls++
		if calls == 1 {
			e := fixtureWindows(&starts, 2)(ctx, r, total, first, emit)
			return errors.Join(e, vk.ErrVulkanInFlight)
		}
		return fixtureWindows(&starts, -1)(ctx, r, total, first, emit)
	}
	owner := mockVulkanJob(time.Millisecond, func(context.Context, time.Duration) error {
		polls++
		if polls == 1 {
			return errors.Join(context.DeadlineExceeded, vk.ErrVulkanInFlight)
		}
		return nil
	}, func() error { return nil }, infer)
	stages := []Stage{fixturePCMStage(1001), owner.Stage()}
	job, e := s.Run(context.Background(), job.ID, config, stages, nil)
	if !errors.Is(e, vk.ErrVulkanInFlight) || job.Status != Failed || len(job.Checkpoints) != 1 || polls != 2 {
		t.Fatal(job, e, polls)
	}
	job, e = s.Run(context.Background(), job.ID, config, stages, nil)
	if e != nil || job.Status != Complete || len(starts) != 2 || starts[1] != 2 || polls != 3 {
		t.Fatal(job, e, starts, polls)
	}
	r, e := s.OpenCheckpoint(context.Background(), job.ID, "asr-windows")
	if readAll(t, r, e) == "" {
		t.Fatal("no retained windows")
	}
	if e = owner.Close(context.Background()); e != nil {
		t.Fatal(e)
	}
}
func TestVulkanJobCloseSerialisesAndRetainsFailedResources(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()
	calls := 0
	s := newVulkanWhisperOwner(time.Millisecond, func(context.Context, time.Duration) error { return nil }, func() error {
		calls++
		if calls == 1 {
			return io.ErrClosedPipe
		}
		return nil
	})
	owner := &VulkanWhisperStage{s: s}
	copyOwner := *owner
	infer := s.wrap(func(context.Context, whisper.SampleReader, int64, int64, func(whisper.WindowTranscript) error) error {
		close(entered)
		<-release
		return nil
	})
	done := make(chan error, 1)
	go func() { done <- infer(context.Background(), nil, 0, 0, nil) }()
	<-entered
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if e := copyOwner.Close(ctx); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
	if calls != 0 {
		t.Fatal("close while active")
	}
	close(release)
	if e := <-done; e != nil {
		t.Fatal(e)
	}
	if e := infer(context.Background(), nil, 0, 0, nil); !errors.Is(e, ErrClosed) {
		t.Fatal("run after stopping", e)
	}
	if e := owner.Close(context.Background()); !errors.Is(e, io.ErrClosedPipe) {
		t.Fatal(e)
	}
	if e := copyOwner.Close(context.Background()); e != nil || calls != 2 {
		t.Fatal(e, calls)
	}
	if e := owner.Close(context.Background()); e != nil || calls != 2 || !owner.Status().Closed {
		t.Fatal(e, calls)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if e := owner.Close(cancelled); e != nil {
		t.Fatal("closed owner not idempotent", e)
	}
	unclosed := &VulkanWhisperStage{s: newVulkanWhisperOwner(time.Millisecond, func(context.Context, time.Duration) error { return nil }, func() error { return nil })}
	if e := unclosed.Close(nil); !errors.Is(e, ErrConfiguration) {
		t.Fatal(e)
	}
	var zero *VulkanWhisperStage
	if !zero.Status().Closed || zero.Stage().Run != nil || zero.Close(context.Background()) != nil {
		t.Fatal("zero owner")
	}
}
func TestVulkanJobBindingIdentityAndHostJournalUnchanged(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_NVIDIA", "1")
	t.Setenv("GO_PHERENCE_WHISPER_GPU_GRAPH", "0")
	t.Setenv("GO_PHERENCE_WHISPER_GPU_SELF_ATTN", "0")
	model, tok := jobToyWhisper()
	cfg := WhisperStageConfig{ModelSHA256: hash([]byte("toy")), RuntimeSHA256: hash([]byte("host")), Language: "pt", MaxWindowBytes: 4096, MaxResultBytes: 128 << 10}
	host, e := NewWhisperWindowStage(model, tok, cfg)
	if e != nil {
		t.Fatal(e)
	}
	// Private test seam only uses CPU toy inference; public constructor requires a
	// concrete whisper.VulkanEncoder and cannot inject another neural runtime.
	s := newVulkanWhisperOwner(time.Millisecond, func(context.Context, time.Duration) error { return nil }, func() error { return nil })
	binding := &residentStageBinding{identity: hash([]byte("device-v1")), validate: model.ValidatePCMHostOnly, wrap: s.wrap}
	a, e := newWhisperWindowStage(model, tok, cfg, binding)
	if e != nil || a.Version == host.Version {
		t.Fatal(e)
	}
	binding.identity = hash([]byte("device-v2"))
	changed, e := newWhisperWindowStage(model, tok, cfg, binding)
	if e != nil || a.Version == changed.Version {
		t.Fatal(e)
	}
	store, _ := openTest(t)
	job := createTest(t, store)
	job, e = store.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(321), a}, nil)
	if e != nil || job.Status != Complete {
		t.Fatal(job, e)
	}
	if _, e = store.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(321), host}, nil); !errors.Is(e, ErrConfiguration) {
		t.Fatal("host resumed device journal", e)
	}
	if _, e = store.Run(context.Background(), job.ID, config, []Stage{fixturePCMStage(321), changed}, nil); !errors.Is(e, ErrConfiguration) {
		t.Fatal("changed backend accepted", e)
	}
	good := VulkanWhisperStageConfig{Whisper: cfg, AllowExperimental: true, BackendSHA256: hash([]byte("backend")), DrainPoll: time.Millisecond}
	for _, kind := range []string{"nil", "zero", "no-consent", "hash", "poll", "nilmodel"} {
		c := good
		encoder := &whisper.VulkanEncoder{}
		m := model
		switch kind {
		case "nil":
			encoder = nil
		case "no-consent":
			c.AllowExperimental = false
		case "hash":
			c.BackendSHA256 = "bad"
		case "poll":
			c.DrainPoll = 0
		case "nilmodel":
			m = nil
		}
		if _, e := NewVulkanWhisperWindowStage(m, tok, encoder, c); e == nil {
			t.Fatal(kind)
		}
	}
}
func TestVulkanJobFatalDrainErrorQuarantines(t *testing.T) {
	for _, mode := range []string{"plain", "joined-inflight"} {
		t.Run(mode, func(t *testing.T) {
			entered := make(chan struct{})
			s := newVulkanWhisperOwner(time.Millisecond, func(context.Context, time.Duration) error {
				close(entered)
				e := errors.New("wait helper unavailable")
				if mode == "joined-inflight" {
					return errors.Join(vk.ErrVulkanInFlight, e)
				}
				return e
			}, func() error { return nil })
			done := make(chan struct{})
			go func() { defer close(done); s.settle(nil) }()
			<-entered
			deadline := time.Now().Add(time.Second)
			for {
				if s.status.Quarantined && s.status.ErrorCode == "drain_failed" {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal(s.status)
				}
				time.Sleep(time.Millisecond)
			}
			select {
			case <-done:
				t.Fatal("fatal drain returned")
			default:
			}
		})
	}
}
func TestVulkanJobClosePanicQuarantines(t *testing.T) {
	s := newVulkanWhisperOwner(time.Millisecond, func(context.Context, time.Duration) error { return nil }, func() error { panic("fixture") })
	o := &VulkanWhisperStage{s: s}
	done := make(chan struct{})
	go func() { defer close(done); _ = o.Close(context.Background()) }()
	waitOwner(t, o, func(s VulkanWhisperStatus) bool { return s.Quarantined && s.ErrorCode == "close_panicked" })
	select {
	case <-done:
		t.Fatal("close panic released owner")
	default:
	}
}
func TestVulkanJobDrainIgnoresCancelledRequestAndFastTimeouts(t *testing.T) {
	polls := 0
	s := newVulkanWhisperOwner(time.Millisecond, func(ctx context.Context, budget time.Duration) error {
		if ctx.Err() != nil || budget != time.Millisecond {
			t.Fatal("invalid drain context")
		}
		polls++
		if polls < 3 {
			return context.DeadlineExceeded
		}
		return nil
	}, func() error { return nil })
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if e := s.wrap(func(context.Context, whisper.SampleReader, int64, int64, func(whisper.WindowTranscript) error) error {
		t.Fatal("cancelled inference ran")
		return nil
	})(ctx, nil, 0, 0, nil); !errors.Is(e, context.Canceled) || polls != 0 {
		t.Fatal(e, polls)
	}
	if e := s.acquire(nil); !errors.Is(e, ErrConfiguration) {
		t.Fatal(e)
	}
	// Error type from gate timeout may omit InFlight; fresh nil drain proves idle.
	s.settle(nil)
	if polls != 3 || s.status.Draining {
		t.Fatal(polls, s.status)
	}
}
