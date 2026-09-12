//go:build linux && amd64

package speechjob

import (
	"context"
	"errors"
	"io"
	"sync/atomic"
	"testing"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	c1 "github.com/rcarmo/go-pherence/models/speaker/community1"
)

func validCommunityVulkanConfig() VulkanCommunity1StageConfig {
	key := hash([]byte("fixture"))
	return VulkanCommunity1StageConfig{Community: Community1StageConfig{AllowExperimental: true, SegmentationSHA256: key, EmbeddingSHA256: key, PLDASHA256: key, FiltersSHA256: key, RuntimeSHA256: key, ModelIdentitySHA256: key, PCM: c1.DiarizationPCMConfig{WindowSamples: 400, StepSamples: 400, MinimumEmbeddingSamples: 1, MinSpeakers: 1, MaxSpeakers: 1, Fa: 1, Fb: 1, TiePolicy: c1.RejectAmbiguousTies}, SegmentationModes: c1.SegmentationModes{SincNet: c1.SincNetScalarFMA, LSTM: c1.LSTMScalar, Head: c1.HeadScalar}, EmbeddingMode: c1.WeSpeakerBlockScalar, MaxResultBytes: 1024}, AllowExperimental: true, BackendSHA256: key, DeviceIdentity: "fixture-device", DrainPoll: time.Millisecond}
}

func TestVulkanCommunityOwnerCancellationDrainsBeforeReturn(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()
	var drains atomic.Int32
	s := newVulkanCommunity1Owner(time.Millisecond, func(ctx context.Context, _ c1.DiarizationPCMReader, _ int64) (*c1.DiarizationPCMResult, error) {
		close(entered)
		<-ctx.Done()
		return nil, errors.Join(ctx.Err(), vk.ErrVulkanInFlight)
	}, func(ctx context.Context, _ time.Duration) error {
		if ctx.Err() != nil {
			t.Error("drain inherited request cancellation")
		}
		drains.Add(1)
		<-release
		return nil
	}, func() error { return nil })
	o := &VulkanCommunity1Stage{s: s}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { _, err := s.infer(ctx, nil, 0); done <- err }()
	<-entered
	cancel()
	deadline := time.Now().Add(time.Second)
	for !o.Status().Draining && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if !o.Status().Running || !o.Status().Draining || drains.Load() != 1 {
		t.Fatal(o.Status(), drains.Load())
	}
	close(release)
	if err := <-done; !errors.Is(err, context.Canceled) || !errors.Is(err, vk.ErrVulkanInFlight) {
		t.Fatal(err)
	}
	if err := o.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestVulkanCommunityOwnerCloseRetryAndIdentity(t *testing.T) {
	cfg := validCommunityVulkanConfig()
	var closes atomic.Int32
	s := newVulkanCommunity1Owner(time.Millisecond, func(context.Context, c1.DiarizationPCMReader, int64) (*c1.DiarizationPCMResult, error) {
		return nil, io.EOF
	}, func(context.Context, time.Duration) error { return nil }, func() error {
		if closes.Add(1) == 1 {
			return io.ErrClosedPipe
		}
		return nil
	})
	o := &VulkanCommunity1Stage{s: s, stage: community1Stage(cfg.Community, s.infer)}
	if err := o.Close(context.Background()); err == nil || o.Status().Closed || o.Status().ErrorCode != "close_failed" {
		t.Fatal(err, o.Status())
	}
	if err := o.Close(context.Background()); err != nil || !o.Status().Closed || closes.Load() != 2 {
		t.Fatal(err, o.Status(), closes.Load())
	}
	plain := community1Stage(cfg.Community, func(context.Context, c1.DiarizationPCMReader, int64) (*c1.DiarizationPCMResult, error) {
		return nil, io.EOF
	})
	accelerated := cfg.Community
	accelerated.ExecutionBackendSHA256 = cfg.BackendSHA256
	device := community1Stage(accelerated, func(context.Context, c1.DiarizationPCMReader, int64) (*c1.DiarizationPCMResult, error) {
		return nil, io.EOF
	})
	if plain.Version == device.Version {
		t.Fatal("backend identity absent from stage key")
	}
}

func TestVulkanCommunityConfigRequiresExplicitBinding(t *testing.T) {
	cfg := validCommunityVulkanConfig()
	for _, kind := range []string{"consent", "hash", "device", "poll", "community"} {
		bad := cfg
		switch kind {
		case "consent":
			bad.AllowExperimental = false
		case "hash":
			bad.BackendSHA256 = "bad"
		case "device":
			bad.DeviceIdentity = ""
		case "poll":
			bad.DrainPoll = 0
		case "community":
			bad.Community.AllowExperimental = false
		}
		if kind == "community" {
			if err := validateCommunityConfig(bad.Community); err == nil {
				t.Fatal(kind)
			}
			continue
		}
		if !bad.AllowExperimental || !validHash(bad.BackendSHA256) || bad.DeviceIdentity == "" || bad.DrainPoll < time.Millisecond {
			continue
		}
		t.Fatal("invalid fixture accepted", kind)
	}
	var zero *VulkanCommunity1Stage
	if !zero.Status().Closed || zero.Stage().Run != nil || zero.Close(context.Background()) != nil {
		t.Fatal("zero owner")
	}
}
