//go:build linux && amd64

package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	c1 "github.com/rcarmo/go-pherence/models/speaker/community1"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

type communityRuntime struct {
	loadSeg  func(context.Context, c1.SegmentationTensorSource, c1.SegmentationLoadConfig) (*c1.SegmentationCheckpoint, error)
	newSeg   func(context.Context, *c1.SegmentationCheckpoint, []float32) (*c1.ExperimentalSegmentation, error)
	loadEmb  func(context.Context, c1.WeSpeakerTensorSource, c1.WeSpeakerResNetConfig, string) (*c1.WeSpeakerResNet34, error)
	newEmb   func(context.Context, *c1.WeSpeakerResNet34) (*c1.ExperimentalEmbedding, error)
	loadPLDA func(context.Context, io.ReaderAt, int64, io.ReaderAt, int64, c1.PLDAConfig) (*c1.RawPLDAPreparation, error)
	newModel func(context.Context, *c1.ExperimentalSegmentation, *c1.ExperimentalEmbedding, *c1.PreparedPLDA) (*c1.ExperimentalDiarization, error)
	newOwner func(*c1.ExperimentalDiarization, speechjob.Community1StageConfig) (stageOwner, error)
}

func defaultCommunityRuntime() communityRuntime {
	return communityRuntime{c1.LoadSegmentationSource, c1.NewExperimentalSegmentation, c1.LoadWeSpeakerResNetSource, c1.NewExperimentalEmbedding, c1.LoadRawPLDANPZ, c1.NewExperimentalDiarization, func(m *c1.ExperimentalDiarization, c speechjob.Community1StageConfig) (stageOwner, error) {
		return speechjob.NewOwnedCommunity1Stage(m, c)
	}}
}

// prepareCommunity validates all bounded immutable assets and exact tensor
// metadata/values under the server's existing loading resource reservation.
// Sources close before return; returned models own copies.
func prepareCommunity(ctx context.Context, c ServerConfig, r communityRuntime) (stageOwner, error) {
	x := c.Profile.Community
	if x == nil {
		return nil, nil
	}
	if r.loadSeg == nil || r.newSeg == nil || r.loadEmb == nil || r.newEmb == nil || r.loadPLDA == nil || r.newModel == nil || r.newOwner == nil {
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
	segmentation, e := r.newSeg(ctx, seg, ownedFilters)
	clear(ownedFilters)
	if e != nil {
		return nil, e
	}
	embFile, e := safetensors.Open(x.Embedding.Path)
	if e != nil {
		return nil, fmt.Errorf("Community-1 embedding source rejected")
	}
	embModel, e := r.loadEmb(ctx, embFile, communityEmbedConfig(x.EmbeddingConfig), x.EmbeddingPrefix)
	e = errors.Join(e, embFile.Close())
	if e != nil {
		return nil, e
	}
	embedding, e := r.newEmb(ctx, embModel)
	if e != nil {
		return nil, e
	}
	xv, e := regularFile(x.XVectorTransform.Path, 64<<20)
	if e != nil {
		return nil, e
	}
	plda, e := regularFile(x.PLDA.Path, 64<<20)
	if e != nil {
		xv.Close()
		return nil, e
	}
	xs, _ := xv.Stat()
	ps, _ := plda.Stat()
	prepared, e := r.loadPLDA(ctx, xv, xs.Size(), plda, ps.Size(), communityPLDAConfig(x.PLDAConfig))
	e = errors.Join(e, xv.Close(), plda.Close())
	if e != nil {
		return nil, e
	}
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
