//go:build linux && amd64

package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	c1 "github.com/rcarmo/go-pherence/models/speaker/community1"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

type communityRuntime struct {
	loadSeg        func(context.Context, c1.SegmentationTensorSource, c1.SegmentationLoadConfig) (*c1.SegmentationCheckpoint, error)
	newSeg         func(context.Context, *c1.SegmentationCheckpoint, []float32) (*c1.ExperimentalSegmentation, error)
	loadEmb        func(context.Context, c1.WeSpeakerTensorSource, c1.WeSpeakerResNetConfig, string) (*c1.WeSpeakerResNet34, error)
	newEmb         func(context.Context, *c1.WeSpeakerResNet34) (*c1.ExperimentalEmbedding, error)
	loadPLDA       func(context.Context, io.ReaderAt, int64, io.ReaderAt, int64, c1.PLDAConfig) (*c1.RawPLDAPreparation, error)
	newModel       func(context.Context, *c1.ExperimentalSegmentation, *c1.ExperimentalEmbedding, *c1.PreparedPLDA) (*c1.ExperimentalDiarization, error)
	newOwner       func(*c1.ExperimentalDiarization, speechjob.Community1StageConfig) (stageOwner, error)
	initVulkan     func() bool
	deviceName     func() string
	newVulkanModel func(context.Context, *c1.SegmentationCheckpoint, []float32, *c1.WeSpeakerResNet34, *c1.PreparedPLDA, int) (*c1.VulkanDiarization, error)
	newVulkanOwner func(*c1.VulkanDiarization, speechjob.VulkanCommunity1StageConfig) (stageOwner, error)
}

func defaultCommunityRuntime() communityRuntime {
	return communityRuntime{
		loadSeg: c1.LoadSegmentationSource, newSeg: c1.NewExperimentalSegmentation,
		loadEmb: c1.LoadWeSpeakerResNetSource, newEmb: c1.NewExperimentalEmbedding,
		loadPLDA: c1.LoadRawPLDANPZ, newModel: c1.NewExperimentalDiarization,
		newOwner: func(m *c1.ExperimentalDiarization, c speechjob.Community1StageConfig) (stageOwner, error) {
			return speechjob.NewOwnedCommunity1Stage(m, c)
		},
		initVulkan: vk.VulkanInit, deviceName: vk.VulkanDeviceName, newVulkanModel: c1.NewVulkanDiarization,
		newVulkanOwner: func(m *c1.VulkanDiarization, c speechjob.VulkanCommunity1StageConfig) (stageOwner, error) {
			return speechjob.NewVulkanCommunity1Stage(m, c)
		},
	}
}

func closeCommunityVulkanModel(model interface{ Close() error }, poll time.Duration, drain func(context.Context, time.Duration) error) {
	for {
		if err := model.Close(); err == nil {
			return
		}
		// Constructor cancellation can leave one accepted submission retained.
		// Prove it idle on a fresh context before retrying destruction. Fatal or
		// uncertain device state deliberately blocks process startup/exit.
		if err := drain(context.Background(), poll); errors.Is(err, vk.ErrVulkanDeviceLost) || errors.Is(err, vk.ErrVulkanUncertain) {
			select {}
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// prepareCommunity validates all bounded immutable assets and exact tensor
// metadata/values under the server's existing loading resource reservation.
// Sources close before return; returned models own copies.
func prepareCommunity(ctx context.Context, c ServerConfig, r communityRuntime) (stageOwner, error) {
	x := c.Profile.Community
	if x == nil {
		return nil, nil
	}
	vulkanSettings := x.Vulkan
	if r.loadSeg == nil || r.loadEmb == nil || r.loadPLDA == nil || vulkanSettings == nil && (r.newSeg == nil || r.newEmb == nil || r.newModel == nil || r.newOwner == nil) || vulkanSettings != nil && (r.initVulkan == nil || r.deviceName == nil || r.newVulkanModel == nil || r.newVulkanOwner == nil) {
		return nil, fmt.Errorf("Community-1 runtime constructors required")
	}
	cfg, e := speechCommunityConfig(*x, c.RuntimeSHA256)
	if e != nil {
		return nil, e
	}
	assets := []Asset{x.Segmentation, x.Filters, x.Embedding, x.XVectorTransform, x.PLDA}
	caps := []int64{2 << 30, 64 << 20, 2 << 30, 64 << 20, 64 << 20}
	for i, a := range assets {
		if e = verifyAsset(ctx, a, caps[i], false); e != nil {
			return nil, e
		}
	}
	segFile, e := safetensors.Open(x.Segmentation.Path)
	if e != nil {
		return nil, fmt.Errorf("Community-1 segmentation source rejected")
	}
	seg, e := r.loadSeg(ctx, segFile, communitySegConfig(x.SegmentationConfig))
	closeErr := segFile.Close()
	if e = errors.Join(e, closeErr); e != nil {
		return nil, e
	}
	filterFile, e := safetensors.Open(x.Filters.Path)
	if e != nil {
		return nil, fmt.Errorf("Community-1 filter source rejected")
	}
	filters, shape, e := filterFile.GetFloat32("sincnet.filters")
	if e == nil && (len(shape) != 2 || shape[0] != 80 || shape[1] != 251) {
		e = fmt.Errorf("Community-1 lowered filter shape rejected")
	}
	ownedFilters := append([]float32(nil), filters...)
	e = errors.Join(e, filterFile.Close())
	if e != nil {
		return nil, e
	}
	var segmentation *c1.ExperimentalSegmentation
	if vulkanSettings == nil {
		segmentation, e = r.newSeg(ctx, seg, ownedFilters)
		if e != nil {
			clear(ownedFilters)
			return nil, e
		}
	}
	embFile, e := safetensors.Open(x.Embedding.Path)
	if e != nil {
		clear(ownedFilters)
		return nil, fmt.Errorf("Community-1 embedding source rejected")
	}
	embModel, e := r.loadEmb(ctx, embFile, communityEmbedConfig(x.EmbeddingConfig), x.EmbeddingPrefix)
	e = errors.Join(e, embFile.Close())
	if e != nil {
		clear(ownedFilters)
		return nil, e
	}
	var embedding *c1.ExperimentalEmbedding
	if vulkanSettings == nil {
		embedding, e = r.newEmb(ctx, embModel)
		if e != nil {
			clear(ownedFilters)
			return nil, e
		}
	}
	xv, e := regularFile(x.XVectorTransform.Path, 64<<20)
	if e != nil {
		clear(ownedFilters)
		return nil, e
	}
	plda, e := regularFile(x.PLDA.Path, 64<<20)
	if e != nil {
		xv.Close()
		clear(ownedFilters)
		return nil, e
	}
	xs, _ := xv.Stat()
	ps, _ := plda.Stat()
	prepared, e := r.loadPLDA(ctx, xv, xs.Size(), plda, ps.Size(), communityPLDAConfig(x.PLDAConfig))
	e = errors.Join(e, xv.Close(), plda.Close())
	if e != nil {
		clear(ownedFilters)
		return nil, e
	}
	if vulkanSettings != nil {
		if !r.initVulkan() {
			clear(ownedFilters)
			return nil, fmt.Errorf("experimental Community-1 Vulkan initialisation failed")
		}
		deviceName := r.deviceName()
		if deviceName == "" || !strings.Contains(deviceName, vulkanSettings.DeviceContains) {
			clear(ownedFilters)
			return nil, fmt.Errorf("configured Community-1 Vulkan device identity rejected")
		}
		model, modelErr := r.newVulkanModel(ctx, seg, ownedFilters, embModel, prepared.Model, cfg.PCM.WindowSamples)
		clear(ownedFilters)
		if modelErr != nil {
			if model != nil {
				closeCommunityVulkanModel(model, time.Duration(vulkanSettings.DrainMilliseconds)*time.Millisecond, vk.VulkanDrain)
			}
			return nil, modelErr
		}
		// The resident owner copied all neural parameters. The server exclusively
		// owns these source graphs and will never use the CPU path in this profile.
		c1.ReleaseVulkanHostWeights(seg, embModel)
		owner, ownerErr := r.newVulkanOwner(model, speechjob.VulkanCommunity1StageConfig{Community: cfg, AllowExperimental: true, BackendSHA256: vulkanSettings.BackendSHA256, DeviceIdentity: deviceName, DrainPoll: time.Duration(vulkanSettings.DrainMilliseconds) * time.Millisecond})
		if ownerErr != nil {
			closeCommunityVulkanModel(model, time.Duration(vulkanSettings.DrainMilliseconds)*time.Millisecond, vk.VulkanDrain)
			return nil, ownerErr
		}
		return owner, nil
	}
	clear(ownedFilters)
	model, e := r.newModel(ctx, segmentation, embedding, prepared.Model)
	if e != nil {
		return nil, e
	}
	owner, e := r.newOwner(model, cfg)
	if e != nil {
		model.ReleaseOwnedModels()
		return nil, e
	}
	return owner, nil
}

// inspectCommunityMetadata validates inexpensive exact container headers during
// --check without loading tensor payloads or PLDA numeric arrays.
func inspectCommunityMetadata(ctx context.Context, c ServerConfig) error {
	x := c.Profile.Community
	if x == nil {
		return nil
	}
	for _, p := range []struct {
		a   Asset
		cap int64
	}{{x.Segmentation, 2 << 30}, {x.Filters, 64 << 20}, {x.Embedding, 2 << 30}, {x.XVectorTransform, 64 << 20}, {x.PLDA, 64 << 20}} {
		if e := verifyAsset(ctx, p.a, p.cap, false); e != nil {
			return e
		}
	}
	for _, a := range []Asset{x.Segmentation, x.Filters, x.Embedding} {
		f, e := safetensors.Open(a.Path)
		if e != nil {
			return e
		}
		infos := f.TensorInfos()
		if len(infos) < 1 || len(infos) > 4096 {
			f.Close()
			return fmt.Errorf("Community-1 tensor inventory rejected")
		}
		if e = f.Close(); e != nil {
			return e
		}
	}
	// NPZ readers validate their complete numeric schemas only during execution;
	// check ZIP local signature here after full hash verification.
	for _, a := range []Asset{x.XVectorTransform, x.PLDA} {
		f, e := os.Open(a.Path)
		if e != nil {
			return e
		}
		var sig [4]byte
		_, e = io.ReadFull(f, sig[:])
		e = errors.Join(e, f.Close())
		if e != nil || binary.LittleEndian.Uint32(sig[:]) != 0x04034b50 {
			return fmt.Errorf("Community-1 NPZ signature rejected")
		}
	}
	return nil
}
