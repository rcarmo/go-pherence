//go:build linux && amd64

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	nem "github.com/rcarmo/go-pherence/model/nemotrondiarization"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

type nemotronStageOwner struct {
	stage  speechjob.Stage
	mu     sync.Mutex
	closed bool
	run    func(context.Context, *speechjob.Input, io.Writer) error
}

func (o *nemotronStageOwner) Stage() speechjob.Stage { return o.stage }
func (o *nemotronStageOwner) Close(context.Context) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.closed = true
	o.run = nil
	return nil
}

func prepareNemotron(ctx context.Context, c ServerConfig) (stageOwner, error) {
	profiles, err := c.configuredProfiles()
	if err != nil {
		return nil, err
	}
	var x *NemotronSettings
	for _, p := range profiles {
		if p.Nemotron != nil {
			x = p.Nemotron
			break
		}
	}
	if x == nil {
		return nil, nil
	}
	if err = inspectNemotronMetadata(ctx, c); err != nil {
		return nil, err
	}
	f, err := safetensors.Open(x.Model.Path)
	if err != nil {
		return nil, err
	}
	model, err := nem.LoadPCMStreamingRequest(f)
	err = errors.Join(err, f.Close())
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	stage, err := speechjob.NewNemotronDiarizationStage(model, speechjob.NemotronDiarizationConfig{AllowExperimental: x.AllowExperimental, ModelSHA256: x.Model.SHA256, RuntimeSHA256: c.RuntimeSHA256, ModelRevision: x.ModelRevision, MaxResultBytes: x.MaxResultBytes})
	if err != nil {
		return nil, err
	}
	return newNemotronStageOwner(stage), nil
}

func newNemotronStageOwner(stage speechjob.Stage) *nemotronStageOwner {
	owner := &nemotronStageOwner{run: stage.Run}
	stage.Run = func(ctx context.Context, in *speechjob.Input, out io.Writer) error {
		owner.mu.Lock()
		defer owner.mu.Unlock()
		if owner.closed {
			return speechjob.ErrClosed
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		return owner.run(ctx, in, out)
	}
	owner.stage = stage
	return owner
}

func inspectNemotronMetadata(ctx context.Context, c ServerConfig) error {
	profiles, err := c.configuredProfiles()
	if err != nil {
		return err
	}
	for _, p := range profiles {
		if p.Nemotron == nil {
			continue
		}
		x := p.Nemotron
		fileLimit := min(int64(2<<30), c.Limits.WeightBytes)
		if err = verifyAsset(ctx, x.Model, fileLimit, false); err != nil {
			return err
		}
		if err = preflightSafetensors(ctx, x.Model, fileLimit, c.Limits.OwnedWeightBytes); err != nil {
			return err
		}
		f, err := safetensors.Open(x.Model.Path)
		if err != nil {
			return err
		}
		infos := f.TensorInfos()
		// Representative released boundary shapes; the loader validates every
		// required tensor, value and shape before publishing a loaded owner.
		if len(infos) < 10 || len(infos) > 4096 {
			f.Close()
			return fmt.Errorf("Nemotron tensor inventory rejected")
		}
		for _, check := range []struct {
			name  string
			shape []int
		}{{"model.audio_tower.input_layer_norm.weight", []int{512}}, {"model.audio_tower.layers.30.layer_norm1.weight", []int{512}}, {"silence_embeds", []int{512}}, {"classifier.out_proj.bias", []int{8}}} {
			info, ok := infos[check.name]
			if !ok || len(info.Shape) != len(check.shape) {
				f.Close()
				return fmt.Errorf("Nemotron checkpoint geometry rejected")
			}
			for i, d := range check.shape {
				if info.Shape[i] != d {
					f.Close()
					return fmt.Errorf("Nemotron checkpoint geometry rejected")
				}
			}
		}
		return f.Close()
	}
	return nil
}
