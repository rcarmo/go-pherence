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

	c1 "github.com/rcarmo/go-pherence/models/speaker/community1"
	"github.com/rcarmo/go-pherence/runtime/resourcebudget"
)

func waitCommunity(t *testing.T, o *Community1Owner, p func(Community1Status) bool) {
	t.Helper()
	d := time.Now().Add(time.Second)
	for time.Now().Before(d) {
		if p(o.Status()) {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal(o.Status())
}
func TestCommunityOwnerCancelDrainAndClose(t *testing.T) {
	entered, drain := make(chan struct{}), make(chan struct{})
	defer func() {
		select {
		case <-drain:
		default:
			close(drain)
		}
	}()
	var releases atomic.Int32
	s := newCommunity1Owner(func(ctx context.Context, _ c1.DiarizationPCMReader, _ int64) (*c1.DiarizationPCMResult, error) {
		close(entered)
		<-ctx.Done()
		<-drain
		return nil, ctx.Err()
	}, func() { releases.Add(1) })
	o := &Community1Owner{s: s}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { _, e := s.infer(ctx, nil, 0); done <- e }()
	<-entered
	cancel()
	waitCommunity(t, o, func(x Community1Status) bool { return x.Running })
	timeout, stop := context.WithTimeout(context.Background(), time.Millisecond)
	defer stop()
	if e := o.Close(timeout); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
	if releases.Load() != 0 {
		t.Fatal("released active graph")
	}
	close(drain)
	if e := <-done; !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	if e := o.Close(context.Background()); e != nil || releases.Load() != 1 {
		t.Fatal(e, releases.Load())
	}
	if e := o.Close(timeout); e != nil {
		t.Fatal(e)
	}
}
func TestCommunityOwnerSerialAndPanicPoison(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	calls := 0
	s := newCommunity1Owner(func(context.Context, c1.DiarizationPCMReader, int64) (*c1.DiarizationPCMResult, error) {
		calls++
		if calls == 1 {
			close(entered)
			<-release
			return nil, nil
		}
		panic("fixture")
	}, func() {})
	first := make(chan error, 1)
	go func() { _, e := s.infer(context.Background(), nil, 0); first <- e }()
	<-entered
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if _, e := s.infer(ctx, nil, 0); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
	close(release)
	if e := <-first; e != nil {
		t.Fatal(e)
	}
	if _, e := s.infer(context.Background(), nil, 0); e == nil {
		t.Fatal("panic escaped as success")
	}
	o := &Community1Owner{s: s}
	if !o.Status().Poisoned || o.Status().ErrorCode != "inference_panicked" {
		t.Fatal(o.Status())
	}
	if _, e := s.infer(context.Background(), nil, 0); !errors.Is(e, ErrClosed) {
		t.Fatal(e)
	}
	if e := o.Close(context.Background()); e != nil {
		t.Fatal(e)
	}
}
func TestCommunityOwnerReleasePanicRetains(t *testing.T) {
	calls := 0
	s := newCommunity1Owner(func(context.Context, c1.DiarizationPCMReader, int64) (*c1.DiarizationPCMResult, error) {
		return nil, nil
	}, func() {
		calls++
		if calls == 1 {
			panic("fixture")
		}
	})
	o := &Community1Owner{s: s}
	if e := o.Close(context.Background()); e == nil || !o.Status().Poisoned || o.Status().Closed {
		t.Fatal(e, o.Status())
	}
	if e := o.Close(context.Background()); e != nil || !o.Status().Closed || calls != 2 {
		t.Fatal(e, o.Status(), calls)
	}
	var zero *Community1Owner
	if !zero.Status().Closed || zero.Stage().Run != nil || zero.Close(context.Background()) != nil {
		t.Fatal("zero")
	}
	fresh := &Community1Owner{s: newCommunity1Owner(func(context.Context, c1.DiarizationPCMReader, int64) (*c1.DiarizationPCMResult, error) {
		return nil, nil
	}, func() {})}
	if e := fresh.Close(nil); !errors.Is(e, ErrConfiguration) || fresh.Status().Stopping {
		t.Fatal(e, fresh.Status())
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if e := fresh.Close(cancelled); !errors.Is(e, context.Canceled) || fresh.Status().Stopping {
		t.Fatal(e, fresh.Status())
	}
	if _, e := fresh.s.infer(context.Background(), nil, 0); e != nil {
		t.Fatal("rejected after invalid close", e)
	}
	fresh.Close(context.Background())
}
func TestCommunityOwnerQueueCancellationRetainsResources(t *testing.T) {
	store, _ := openTest(t)
	job := createTest(t, store)
	entered, drain := make(chan struct{}), make(chan struct{})
	defer func() {
		select {
		case <-drain:
		default:
			close(drain)
		}
	}()
	s := newCommunity1Owner(func(ctx context.Context, _ c1.DiarizationPCMReader, _ int64) (*c1.DiarizationPCMResult, error) {
		close(entered)
		<-ctx.Done()
		<-drain
		return nil, ctx.Err()
	}, func() {})
	cfg := Community1StageConfig{AllowExperimental: true, SegmentationSHA256: hash([]byte("s")), EmbeddingSHA256: hash([]byte("e")), PLDASHA256: hash([]byte("p")), FiltersSHA256: hash([]byte("f")), RuntimeSHA256: hash([]byte("r")), PCM: c1.DiarizationPCMConfig{WindowSamples: 400, StepSamples: 1, MinimumEmbeddingSamples: 1, MinSpeakers: 1, MaxSpeakers: 1, Fa: 1, Fb: 1, TiePolicy: c1.RejectAmbiguousTies}, SegmentationModes: c1.SegmentationModes{SincNet: c1.SincNetScalarFMA, LSTM: c1.LSTMScalar, Head: c1.HeadScalar}, EmbeddingMode: c1.WeSpeakerBlockScalar, MaxResultBytes: 1024}
	cfg.PCM.StepSamples = 400
	owner := &Community1Owner{s: s, stage: community1Stage(cfg, s.infer)}
	budget, e := resourcebudget.New(resourcebudget.Config{Capacity: resourcebudget.Resources{CPUSlots: 2, MemoryBytes: 100}, MaxActive: 1, MaxWaiting: 1})
	if e != nil {
		t.Fatal(e)
	}
	admit, _ := budget.Admission(resourcebudget.Resources{CPUSlots: 2, MemoryBytes: 100})
	q := openQueueTest(t, store, filepath.Join(t.TempDir(), "queue"), admit, func(Manifest) ([]Stage, error) { return []Stage{fixturePCMStage(800), owner.Stage()}, nil })
	q.Enqueue(context.Background(), job.ID, false)
	q.Start(context.Background())
	<-entered
	q.Cancel(context.Background(), job.ID)
	if budget.Snapshot().Active != 1 {
		t.Fatal("resource released before synchronous return")
	}
	if e := store.Close(); !errors.Is(e, ErrBusy) {
		t.Fatal(e)
	}
	timeout, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if e := q.Shutdown(timeout); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
	if e := owner.Close(timeout); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
	close(drain)
	q.Shutdown(context.Background())
	if budget.Snapshot().Active != 0 {
		t.Fatal(budget.Snapshot())
	}
	if e := owner.Close(context.Background()); e != nil {
		t.Fatal(e)
	}
	budget.Shutdown(context.Background())
}
func TestCommunityOwnerStageIdentityMatchesRaw(t *testing.T) {
	cfg := Community1StageConfig{AllowExperimental: true, SegmentationSHA256: hash([]byte("s")), EmbeddingSHA256: hash([]byte("e")), PLDASHA256: hash([]byte("p")), FiltersSHA256: hash([]byte("f")), RuntimeSHA256: hash([]byte("r")), PCM: c1.DiarizationPCMConfig{WindowSamples: 400, StepSamples: 1, MinimumEmbeddingSamples: 1, MinSpeakers: 1, MaxSpeakers: 1, Fa: 1, Fb: 1, TiePolicy: c1.RejectAmbiguousTies}, SegmentationModes: c1.SegmentationModes{SincNet: c1.SincNetScalarFMA, LSTM: c1.LSTMScalar, Head: c1.HeadScalar}, EmbeddingMode: c1.WeSpeakerBlockScalar, MaxResultBytes: 1024}
	raw := community1Stage(cfg, func(context.Context, c1.DiarizationPCMReader, int64) (*c1.DiarizationPCMResult, error) {
		return nil, io.EOF
	})
	s := newCommunity1Owner(func(context.Context, c1.DiarizationPCMReader, int64) (*c1.DiarizationPCMResult, error) {
		return nil, io.EOF
	}, func() {})
	owned := community1Stage(cfg, s.infer)
	if raw.Version != owned.Version {
		t.Fatal("owner changed numerical identity")
	}
}
