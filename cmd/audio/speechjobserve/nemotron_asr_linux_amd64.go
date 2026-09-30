//go:build linux && amd64

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	asr "github.com/rcarmo/go-pherence/model/nemotronasr"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
	"github.com/rcarmo/go-pherence/runtime/speechjob/httpapi"
)

func prepareNemotronASRMetadata(ctx context.Context, c ServerConfig) (*tokenizer.Tokenizer, error) {
	if err := validateRuntime(c); err != nil {
		return nil, err
	}
	for _, p := range []struct {
		a   Asset
		cap int64
	}{{c.ModelConfig, 1 << 20}, {c.Tokenizer, 4 << 20}, {c.Generation, 16 << 10}} {
		if err := verifyAsset(ctx, p.a, p.cap, false); err != nil {
			return nil, err
		}
	}
	raw, err := boundedFile(ctx, c.ModelConfig.Path, 1<<20)
	if err != nil {
		return nil, err
	}
	var geometry struct {
		ModelType     string `json:"model_type"`
		DefaultPrompt int    `json:"default_prompt_id"`
		Vocab         int    `json:"vocab_size"`
		Encoder       struct {
			Layers int `json:"num_hidden_layers"`
			Width  int `json:"hidden_size"`
			Mels   int `json:"num_mel_bins"`
		} `json:"encoder_config"`
	}
	if json.Unmarshal(raw, &geometry) != nil || geometry.ModelType != "nemotron3_5_asr" || geometry.DefaultPrompt != 101 || geometry.Vocab != 13088 || geometry.Encoder.Layers != 24 || geometry.Encoder.Width != 1024 || geometry.Encoder.Mels != 128 {
		return nil, fmt.Errorf("Nemotron ASR geometry rejected")
	}
	if err = verifyAsset(ctx, c.Weights, c.Limits.WeightBytes, false); err != nil {
		return nil, err
	}
	if err = preflightSafetensors(ctx, c.Weights, c.Limits.WeightBytes, c.Limits.OwnedWeightBytes); err != nil {
		return nil, err
	}
	raw, err = boundedFile(ctx, c.Tokenizer.Path, 4<<20)
	if err != nil {
		return nil, err
	}
	if err = uniqueJSON(raw, 16, false, true); err != nil {
		return nil, err
	}
	vocab, err := tokenizer.Load(c.Tokenizer.Path)
	if err != nil {
		return nil, err
	}
	if len(vocab.InvVocab) != 13089 {
		return nil, fmt.Errorf("Nemotron ASR vocabulary rejected")
	}
	for id := 0; id < 13087; id++ {
		if _, ok := vocab.InvVocab[id]; !ok {
			return nil, fmt.Errorf("Nemotron ASR vocabulary ID missing")
		}
	}
	if c.Profile.MediaBackend == "ffmpeg" {
		for _, a := range []Asset{c.FFmpeg, c.FFprobe} {
			if err = verifyAsset(ctx, a, 1<<30, true); err != nil {
				return nil, err
			}
		}
	}
	if _, err = newDecodeStage(c); err != nil {
		return nil, err
	}
	return vocab, ctx.Err()
}

func buildNemotronASRProfiles(ctx context.Context, c ServerConfig, load bool, communityRuntime communityRuntime) (*builtProfiles, error) {
	vocab, err := prepareNemotronASRMetadata(ctx, c)
	if err != nil {
		return nil, err
	}
	if !load {
		return &builtProfiles{}, nil
	}
	f, err := safetensors.Open(c.Weights.Path)
	if err != nil {
		return nil, err
	}
	model, loadErr := asr.LoadPCMGenerationModel(f)
	if err = errors.Join(loadErr, f.Close()); err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	profiles, err := c.configuredProfiles()
	if err != nil {
		return nil, err
	}
	result := &builtProfiles{}
	diarOwner, err := prepareNemotron(ctx, c)
	if err != nil {
		return nil, err
	}
	if diarOwner != nil {
		result.owners = append(result.owners, diarOwner)
	}
	var communityOwner stageOwner
	if configuredCommunity(profiles) != nil {
		communityOwner, err = prepareCommunity(ctx, c, communityRuntime)
		if err != nil {
			closeBuiltProfiles(result)
			return nil, err
		}
		result.owners = append(result.owners, communityOwner)
	}
	prompts := map[string]int{"auto": 101, "en": 0, "pt": 13, "fr": 8, "es": 2, "it": 15}
	for _, opts := range profiles {
		decode, err := newDecodeStageFor(c, opts)
		if err != nil {
			closeBuiltProfiles(result)
			return nil, err
		}
		text, err := speechjob.NewNemotronASRStage(model, vocab, speechjob.NemotronASRConfig{ModelSHA256: c.Weights.SHA256, TokenizerSHA256: c.Tokenizer.SHA256, RuntimeSHA256: c.RuntimeSHA256, ModelRevision: c.NemotronASR.ModelRevision, Language: opts.Language, PromptID: prompts[opts.Language], MaxResultBytes: opts.ResultBytes})
		if err != nil {
			closeBuiltProfiles(result)
			return nil, err
		}
		// Reuse the serial drained CPU owner; every profile owns only its closure,
		// while all closures share immutable ASR weights and separate request state.
		owner := newNemotronStageOwner(text)
		result.owners = append(result.owners, owner)
		text = owner.Stage()
		stages := []speechjob.Stage{decode, text, speechjob.NewVTTStage()}
		if opts.Nemotron != nil {
			if diarOwner == nil {
				closeBuiltProfiles(result)
				return nil, fmt.Errorf("Nemotron diarization owner missing")
			}
			diar := diarOwner.Stage()
			labels, err := speechjob.NewNemotronSpeakerTranscriptStage(speechjob.SpeakerTranscriptConfig{AllowExperimental: true, TranscriptVersion: text.Version, DiarizationVersion: diar.Version})
			if err != nil {
				closeBuiltProfiles(result)
				return nil, err
			}
			stages = append(stages, diar, labels, speechjob.NewSpeakerVTTStage())
		}
		if opts.Community != nil {
			if communityOwner == nil {
				closeBuiltProfiles(result)
				return nil, fmt.Errorf("Community owner missing")
			}
			diar := communityOwner.Stage()
			labels, err := speechjob.NewSpeakerTranscriptStage(speechjob.SpeakerTranscriptConfig{AllowExperimental: true, TranscriptVersion: text.Version, DiarizationVersion: diar.Version})
			if err != nil {
				closeBuiltProfiles(result)
				return nil, err
			}
			stages = append(stages, diar, labels, speechjob.NewSpeakerVTTStage())
		}
		type stageIdentity struct{ Name, Version string }
		versions := make([]stageIdentity, len(stages))
		for i, s := range stages {
			versions[i] = stageIdentity{s.Name, s.Version}
		}
		identity, err := json.Marshal(struct {
			Schema                                string
			Runtime                               string
			Weights, Model, Tokenizer, Generation Asset
			Profile                               ProfileSettings
			Threads                               int
			Stages                                []stageIdentity
		}{"nemotron-asr-cpu-profiles-v1", c.RuntimeSHA256, c.Weights, c.ModelConfig, c.Tokenizer, c.Generation, opts, c.Threads, versions})
		if err != nil {
			closeBuiltProfiles(result)
			return nil, err
		}
		result.Profiles = append(result.Profiles, httpapi.Profile{ID: opts.ID, Configuration: identity, Stages: stages})
	}
	return result, nil
}
