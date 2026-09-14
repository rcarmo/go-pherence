//go:build linux && amd64

package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"time"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/models/whisper"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
	"github.com/rcarmo/go-pherence/runtime/speechjob/httpapi"
)

// Runtime flags are read at package init by legacy Whisper. They MUST be set in
// the launching environment, not changed after startup. In particular Whisper
// import probes NVIDIA at init; DISABLE_NVIDIA must already be 1 before exec.
func validateRuntime(c ServerConfig) error {
	if os.Getenv("GO_PHERENCE_DISABLE_NVIDIA") != "1" || os.Getenv("GOMAXPROCS") != fmt.Sprint(c.Threads) || runtime.GOMAXPROCS(0) != c.Threads {
		return fmt.Errorf("launch with GO_PHERENCE_DISABLE_NVIDIA=1 and GOMAXPROCS equal to configured threads")
	}
	for _, env := range os.Environ() {
		key, v, _ := strings.Cut(env, "=")
		if (strings.HasPrefix(key, "WHISPER_") || strings.HasPrefix(key, "GO_PHERENCE_WHISPER_") || key == "GO_PHERENCE_A100_NOINIT") && v != "" && v != "0" {
			return fmt.Errorf("non-default Whisper runtime flags are unsupported by this profile")
		}
	}
	return nil
}

type preparedModel struct {
	cfg                   whisper.Config
	modelJSON, generation []byte
	tokenizer             *whisper.Tokenizer
	source                *safetensors.File
}

func prepareModel(ctx context.Context, c ServerConfig) (*preparedModel, error) {
	if e := validateRuntime(c); e != nil {
		return nil, e
	}
	// Preflight small pinned metadata before tensor payload hashing/opening.
	for _, x := range []struct {
		a    Asset
		cap  int64
		exec bool
	}{{c.ModelConfig, 1 << 20, false}, {c.Generation, 16 << 10, false}, {c.Tokenizer, 32 << 20, false}} {
		if e := verifyAsset(ctx, x.a, x.cap, x.exec); e != nil {
			return nil, e
		}
	}
	modelJSON, e := boundedFile(ctx, c.ModelConfig.Path, 1<<20)
	if e != nil {
		return nil, e
	}
	cfg, e := whisper.ParseModelConfigChecked(modelJSON)
	if e != nil {
		return nil, fmt.Errorf("model configuration rejected")
	}
	generation, e := boundedFile(ctx, c.Generation.Path, 16<<10)
	if e != nil {
		return nil, e
	}
	tokenizerBytes, e := boundedFile(ctx, c.Tokenizer.Path, 32<<20)
	if e != nil {
		return nil, e
	}
	if e = uniqueJSON(tokenizerBytes, 16, false, true); e != nil {
		return nil, fmt.Errorf("tokenizer JSON rejected")
	}
	// Pinned, bounded, immutable local tokenizer: loader and checked generation
	// validate full multilingual vocabulary before any tensor payload is loaded.
	tokenizer, e := whisper.LoadTokenizer(c.Tokenizer.Path)
	if e != nil {
		return nil, fmt.Errorf("tokenizer load rejected")
	}
	if _, e = whisper.ParseGenerationConfigChecked(generation, cfg, tokenizer); e != nil {
		return nil, fmt.Errorf("generation policy rejected")
	}
	var generationBounds struct {
		MaxLength int `json:"max_length"`
	}
	if e = json.Unmarshal(generation, &generationBounds); e != nil {
		return nil, fmt.Errorf("generation policy rejected")
	}
	profiles, e := c.configuredProfiles()
	if e != nil {
		return nil, e
	}
	c.Profile = profiles[0]
	for _, p := range profiles {
		if p.MaxInitialTimestampIndex != 0 || p.MaxNewTokens > generationBounds.MaxLength-3 {
			return nil, fmt.Errorf("generation document owns initial timestamp and token bounds")
		}
		if _, e = whisper.NewWindowPlan(0, int64(cfg.MaxLength)*160, p.OverlapSamples); e != nil || p.OverlapSamples > int64(cfg.MaxLength)*80 || p.MaxNewTokens > cfg.MaxDecoderLength-3 {
			return nil, fmt.Errorf("profile exceeds model geometry")
		}
		if p.Language == "auto" {
			continue
		}
		languageFound := false
		for _, s := range tokenizer.Vocab {
			if s == "<|"+p.Language+"|>" {
				languageFound = true
			}
		}
		if !languageFound {
			return nil, fmt.Errorf("profile language not in tokenizer")
		}
	}
	if c.Profile.MediaBackend == "ffmpeg" {
		for _, x := range []Asset{c.FFmpeg, c.FFprobe} {
			if e = verifyAsset(ctx, x, 1<<30, true); e != nil {
				return nil, e
			}
		}
	}
	if _, e = newDecodeStage(c); e != nil {
		return nil, fmt.Errorf("decode configuration rejected")
	}
	if e = preflightSafetensors(ctx, c.Weights, c.Limits.WeightBytes, c.Limits.OwnedWeightBytes); e != nil {
		return nil, e
	}
	if e = verifyAsset(ctx, c.Weights, c.Limits.WeightBytes, false); e != nil {
		return nil, e
	}
	source, e := safetensors.Open(c.Weights.Path)
	if e != nil {
		return nil, fmt.Errorf("safetensors metadata rejected")
	}
	return &preparedModel{cfg, modelJSON, generation, tokenizer, source}, nil
}

// Preflight the bounded header before safetensors.Open unmarshals it. Widened
// tensor storage + one source materialisation are admitted conservatively as
// 2*sum(F32 bytes). This is a loading allocation estimate, not a process RSS cap.
func preflightSafetensors(ctx context.Context, a Asset, fileLimit, ownedLimit int64) error {
	f, e := regularFile(a.Path, fileLimit)
	if e != nil {
		return e
	}
	defer f.Close()
	st, e := f.Stat()
	if e != nil {
		return e
	}
	var length [8]byte
	if _, e = io.ReadFull(f, length[:]); e != nil {
		return fmt.Errorf("short safetensors header")
	}
	n := binary.LittleEndian.Uint64(length[:])
	if n < 2 || n > 4<<20 || n > uint64(st.Size()-8) {
		return fmt.Errorf("safetensors header exceeds bounds")
	}
	b := make([]byte, int(n))
	if _, e = io.ReadFull(f, b); e != nil {
		return fmt.Errorf("short safetensors header")
	}
	if e = ctx.Err(); e != nil {
		return e
	}
	if e = uniqueJSON(b, 8, false, false); e != nil {
		return fmt.Errorf("ambiguous safetensors header")
	}
	var fields map[string]json.RawMessage
	if e = json.Unmarshal(b, &fields); e != nil || len(fields) < 1 || len(fields) > 2048 {
		return fmt.Errorf("invalid safetensors inventory")
	}
	var owned int64
	for name, raw := range fields {
		if e = ctx.Err(); e != nil {
			return e
		}
		if name == "__metadata__" {
			continue
		}
		var info safetensors.TensorInfo
		if e = json.Unmarshal(raw, &info); e != nil {
			return fmt.Errorf("invalid tensor metadata")
		}
		width := int64(0)
		switch info.DType {
		case "F32":
			width = 4
		case "F16", "BF16":
			width = 2
		default:
			return fmt.Errorf("unsupported weight dtype")
		}
		elements := int64(1)
		if len(info.Shape) < 1 || len(info.Shape) > 4 {
			return fmt.Errorf("invalid tensor rank")
		}
		for _, d := range info.Shape {
			if d < 1 || int64(d) > ownedLimit/4/elements {
				return fmt.Errorf("tensor allocation exceeds cap")
			}
			elements *= int64(d)
		}
		start, end := int64(info.DataOffsets[0]), int64(info.DataOffsets[1])
		if start < 0 || end < start || end > st.Size()-8-int64(n) || end-start != elements*width {
			return fmt.Errorf("invalid tensor extent")
		}
		if elements*8 > ownedLimit-owned {
			return fmt.Errorf("widened model allocation exceeds configured cap")
		}
		owned += elements * 8
	}
	return ctx.Err()
}
func newDecodeStage(c ServerConfig) (speechjob.Stage, error) { return newDecodeStageFor(c, c.Profile) }
func newDecodeStageFor(c ServerConfig, profile ProfileSettings) (speechjob.Stage, error) {
	switch profile.MediaBackend {
	case "go264":
		return speechjob.NewGo264DecodeStage(speechjob.Go264DecodeConfig{InputExtension: profile.Extension, MaxInputBytes: c.Limits.UploadBytes, MaxOutputBytes: profile.DecodeBytes, MaxDuration: duration(profile.MaxDurationSeconds)})
	case "ffmpeg":
		return speechjob.NewFFmpegDecodeStage(speechjob.FFmpegDecodeConfig{FFmpegPath: c.FFmpeg.Path, FFprobePath: c.FFprobe.Path, FFmpegSHA256: c.FFmpeg.SHA256, FFprobeSHA256: c.FFprobe.SHA256, InputExtension: profile.Extension, MaxInputBytes: c.Limits.UploadBytes, MaxOutputBytes: profile.DecodeBytes, MaxDuration: duration(profile.MaxDurationSeconds)})
	default:
		return speechjob.Stage{}, fmt.Errorf("invalid media backend")
	}
}

type profileRuntimes struct {
	Vulkan    vulkanProfileRuntime
	Community communityRuntime
}
type vulkanProfileRuntime struct {
	init       func() bool
	deviceName func() string
	newEncoder func(context.Context, *whisper.Encoder, int) (*whisper.VulkanEncoder, error)
	newStage   func(*whisper.Whisper, *whisper.Tokenizer, *whisper.VulkanEncoder, speechjob.VulkanWhisperStageConfig) (stageOwner, error)
	newStages  func(*whisper.Whisper, *whisper.Tokenizer, *whisper.VulkanEncoder, []speechjob.VulkanWhisperStageConfig) (*speechjob.VulkanWhisperStage, error)
	drain      func(context.Context, time.Duration) error
}

func defaultVulkanProfileRuntime() vulkanProfileRuntime {
	return vulkanProfileRuntime{init: vk.VulkanInit, deviceName: vk.VulkanDeviceName, newEncoder: whisper.NewVulkanEncoder, newStage: func(m *whisper.Whisper, t *whisper.Tokenizer, e *whisper.VulkanEncoder, c speechjob.VulkanWhisperStageConfig) (stageOwner, error) {
		return speechjob.NewVulkanWhisperWindowStage(m, t, e, c)
	}, newStages: speechjob.NewVulkanWhisperWindowStages, drain: vk.VulkanDrain}
}
func buildProfile(ctx context.Context, c ServerConfig, load bool) ([]httpapi.Profile, error) {
	b, e := buildProfileOwned(ctx, c, load, defaultVulkanProfileRuntime())
	if e != nil {
		return nil, e
	}
	if len(b.owners) > 0 {
		return nil, fmt.Errorf("experimental Vulkan profile requires owned builder")
	}
	return b.Profiles, nil
}
func buildProfileOwned(ctx context.Context, c ServerConfig, load bool, runtime vulkanProfileRuntime) (*builtProfiles, error) {
	return buildProfileOwnedRuntimes(ctx, c, load, profileRuntimes{runtime, defaultCommunityRuntime()})
}
func buildProfileOwnedRuntimes(ctx context.Context, c ServerConfig, load bool, runtimes profileRuntimes) (*builtProfiles, error) {
	runtime := runtimes.Vulkan
	p, e := prepareModel(ctx, c)
	if e != nil {
		return nil, e
	}
	defer p.source.Close()
	if !load {
		return &builtProfiles{}, nil
	}
	model, _, e := whisper.LoadConfiguredModelSourceChecked(ctx, p.source, p.modelJSON, p.generation, p.tokenizer)
	if e != nil {
		return nil, fmt.Errorf("checked model loading failed")
	}
	if e = model.ValidatePCMHostOnly(); e != nil {
		return nil, e
	}
	profiles, e := c.configuredProfiles()
	if e != nil {
		return nil, e
	}
	decodeStages := make([]speechjob.Stage, len(profiles))
	whisperConfigs := make([]speechjob.WhisperStageConfig, len(profiles))
	asrStages := make([]speechjob.Stage, len(profiles))
	for i, opts := range profiles {
		decodeStages[i], e = newDecodeStageFor(c, opts)
		if e != nil {
			return nil, e
		}
		whisperConfigs[i] = speechjob.WhisperStageConfig{ModelSHA256: c.Weights.SHA256, RuntimeSHA256: c.RuntimeSHA256, Language: opts.Language, OverlapSamples: opts.OverlapSamples, MaxNewTokens: opts.MaxNewTokens, MaxInitialTimestampIndex: opts.MaxInitialTimestampIndex, SkipDigitalSilence: opts.SkipDigitalSilence, WordTimestamps: opts.WordTimestamps, GenerationJSON: p.generation, MaxWindowBytes: opts.WindowBytes, MaxResultBytes: opts.ResultBytes}
		asrStages[i], e = speechjob.NewWhisperWindowStage(model, p.tokenizer, whisperConfigs[i])
		if e != nil {
			return nil, e
		}
	}
	result := &builtProfiles{}
	if v := profiles[0].Vulkan; v != nil {
		if runtime.init == nil || runtime.deviceName == nil || runtime.newEncoder == nil || runtime.drain == nil || len(profiles) == 1 && runtime.newStage == nil || len(profiles) > 1 && runtime.newStages == nil || !runtime.init() {
			return nil, fmt.Errorf("experimental Vulkan initialisation failed")
		}
		if name := runtime.deviceName(); name == "" || !strings.Contains(name, v.DeviceContains) {
			return nil, fmt.Errorf("configured Vulkan device identity rejected")
		}
		encoder, e := runtime.newEncoder(ctx, model.Encoder, model.Config.MaxLength)
		if e != nil {
			if encoder != nil {
				closeVulkanEncoder(encoder, time.Duration(v.DrainMilliseconds)*time.Millisecond, runtime.drain)
			}
			return nil, e
		}
		model.Encoder.ReleaseHostWeights()
		model.Encoder = nil
		vulkanConfigs := make([]speechjob.VulkanWhisperStageConfig, len(profiles))
		for i := range profiles {
			vulkanConfigs[i] = speechjob.VulkanWhisperStageConfig{Whisper: whisperConfigs[i], AllowExperimental: true, BackendSHA256: v.BackendSHA256, DrainPoll: time.Duration(v.DrainMilliseconds) * time.Millisecond}
		}
		var owner stageOwner
		if len(profiles) == 1 {
			owner, e = runtime.newStage(model, p.tokenizer, encoder, vulkanConfigs[0])
		} else {
			multi, err := runtime.newStages(model, p.tokenizer, encoder, vulkanConfigs)
			e = err
			owner = multi
			if multi != nil {
				asrStages = multi.Stages()
			}
		}
		if e != nil || owner == nil {
			if owner != nil {
				closeBuiltProfiles(&builtProfiles{owners: []stageOwner{owner}})
			} else {
				closeVulkanEncoder(encoder, time.Duration(v.DrainMilliseconds)*time.Millisecond, runtime.drain)
			}
			if e != nil {
				return nil, e
			}
			return nil, fmt.Errorf("Vulkan stage constructor returned nil")
		}
		result.owners = append(result.owners, owner)
		if len(profiles) == 1 {
			asrStages[0] = owner.Stage()
		}
	}
	var communityOwner stageOwner
	if profiles[0].Community != nil {
		communityOwner, e = prepareCommunity(ctx, c, runtimes.Community)
		if e != nil || communityOwner == nil {
			closeBuiltProfiles(result)
			if e != nil {
				return nil, e
			}
			return nil, fmt.Errorf("Community-1 stage constructor returned nil")
		}
		result.owners = append(result.owners, communityOwner)
	}
	for i, opts := range profiles {
		asr := asrStages[i]
		text, err := speechjob.NewTranscriptStage(speechjob.TranscriptStageConfig{ASRVersion: asr.Version, Language: opts.Language, WindowSamples: int64(p.cfg.MaxLength) * 160, OverlapSamples: opts.OverlapSamples})
		if err != nil {
			closeBuiltProfiles(result)
			return nil, err
		}
		stages := []speechjob.Stage{decodeStages[i], asr, text, speechjob.NewVTTStage()}
		if communityOwner != nil {
			diar := communityOwner.Stage()
			speaker, err := speechjob.NewSpeakerTranscriptStage(speechjob.SpeakerTranscriptConfig{TranscriptVersion: text.Version, DiarizationVersion: diar.Version, AllowExperimental: true})
			if err != nil {
				closeBuiltProfiles(result)
				return nil, err
			}
			stages = append(stages, diar, speaker, speechjob.NewSpeakerVTTStage())
		}
		type stageIdentity struct{ Name, Version string }
		versions := make([]stageIdentity, len(stages))
		for j, stage := range stages {
			versions[j] = stageIdentity{stage.Name, stage.Version}
		}
		identity, err := json.Marshal(struct {
			Schema                                int
			Runtime                               string
			Weights, Model, Tokenizer, Generation Asset
			Profile                               ProfileSettings
			Threads                               int
			Stages                                []stageIdentity
		}{1, c.RuntimeSHA256, c.Weights, c.ModelConfig, c.Tokenizer, c.Generation, opts, c.Threads, versions})
		if err != nil {
			closeBuiltProfiles(result)
			return nil, err
		}
		result.Profiles = append(result.Profiles, httpapi.Profile{ID: opts.ID, Configuration: identity, Stages: stages})
	}
	return result, nil
}
