//go:build linux && amd64

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	c1 "github.com/rcarmo/go-pherence/models/speaker/community1"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testSafeTensor(t *testing.T, name string, shape []int, values []float32) []byte {
	t.Helper()
	var payload bytes.Buffer
	for _, v := range values {
		binary.Write(&payload, binary.LittleEndian, math.Float32bits(v))
	}
	header, _ := json.Marshal(map[string]safetensors.TensorInfo{name: {DType: "F32", Shape: shape, DataOffsets: [2]int{0, payload.Len()}}})
	var out bytes.Buffer
	binary.Write(&out, binary.LittleEndian, uint64(len(header)))
	out.Write(header)
	out.Write(payload.Bytes())
	return out.Bytes()
}
func communityConfig(t *testing.T) CommunitySettings {
	t.Helper()
	d := t.TempDir()
	put := func(n string, b []byte) Asset { return putAsset(t, d, n, b, 0600) }
	npz := make([]byte, 8)
	binary.LittleEndian.PutUint32(npz, 0x04034b50)
	return CommunitySettings{Enable: true, AllowExperimental: true, ModelRevision: "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee", Segmentation: put("segmentation.safetensors", testSafeTensor(t, "dummy", []int{1}, []float32{1})), Filters: put("filters.safetensors", testSafeTensor(t, "sincnet.filters", []int{80, 251}, make([]float32, 80*251))), Embedding: put("embedding.safetensors", testSafeTensor(t, "dummy", []int{1}, []float32{1})), XVectorTransform: put("xvec.npz", npz), PLDA: put("plda.npz", npz), SegmentationConfig: CommunitySegmentationSettings{SincNetStride: 10, LSTM: CommunityLSTMSettings{InputSize: 60, HiddenSize: 128, NumLayers: 4, Bidirectional: true}, Head: CommunityHeadSettings{InputSize: 256, HiddenSize: 128, NumLayers: 2, Speakers: 3, MaxActive: 2}}, EmbeddingConfig: CommunityEmbeddingSettings{32, 80, 256}, EmbeddingPrefix: "resnet", PLDAConfig: CommunityPLDASettings{256, 128, 128}, PCM: CommunityPCMSettings{WindowSamples: 160000, StepSamples: 16000, MinimumEmbeddingSamples: 400, ExcludeOverlap: true, MinSpeakers: 1, MaxSpeakers: 64, AHCThreshold: .6, Fa: .07, Fb: .8, Constrained: true, TiePolicy: "reject"}, Modes: CommunityModesSettings{SincNet: "simd", LSTM: "simd", Head: "simd", Embedding: "simd"}, MaxResultBytes: 16 << 20}
}
func TestCommunityConfigStrictSnakeCaseAndModes(t *testing.T) {
	c := baseConfig(t)
	r := ResourceSettings{CPUSlots: 2, MemoryBytes: 1 << 30, MaxWaiting: 4, LoadBytes: 512 << 20, ResidentBytes: 256 << 20, WorkBytes: 256 << 20}
	c.Resources = &r
	x := communityConfig(t)
	c.Profile.Community = &x
	if e := c.validate(); e != nil {
		t.Fatal(e)
	}
	cfg, e := speechCommunityConfig(x, c.RuntimeSHA256)
	if e != nil || cfg.PCM.TiePolicy != c1.RejectAmbiguousTies || cfg.EmbeddingMode != c1.WeSpeakerBlockSIMD || !validHash(cfg.ModelIdentitySHA256) {
		t.Fatal(cfg, e)
	}
	baseIdentity := cfg.ModelIdentitySHA256
	baseStageIdentity, _ := json.Marshal(cfg)
	x.Modes.OverlapBranches = true
	overlapped, e := speechCommunityConfig(x, c.RuntimeSHA256)
	overlappedStageIdentity, _ := json.Marshal(overlapped)
	if e != nil || !overlapped.OverlapBranches || bytes.Equal(baseStageIdentity, overlappedStageIdentity) || overlapped.ModelIdentitySHA256 != baseIdentity {
		t.Fatal("CPU overlap identity", overlapped, e)
	}
	x.Modes.OverlapBranches = false
	vulkanOverlap := x
	vulkanOverlap.Modes.LSTM = "vulkan"
	vulkanOverlap.Modes.Embedding = "vulkan"
	vulkanOverlap.Modes.OverlapBranches = true
	vulkanOverlap.Vulkan = &CommunityVulkanSettings{}
	if _, e := speechCommunityConfig(vulkanOverlap, c.RuntimeSHA256); e == nil {
		t.Fatal("Vulkan accepted CPU branch overlap")
	}
	x.PCM.TiePolicy = "lowest-index"
	x.Modes.Embedding = "gemm"
	cfg, e = speechCommunityConfig(x, c.RuntimeSHA256)
	if e != nil || cfg.PCM.TiePolicy != c1.LowestIndexTies || cfg.EmbeddingMode != c1.WeSpeakerBlockGEMM || cfg.ModelIdentitySHA256 != baseIdentity {
		t.Fatal(cfg, e)
	}
	changed := x
	changed.XVectorTransform.SHA256 = hashBytes([]byte("changed"))
	different, e := speechCommunityConfig(changed, c.RuntimeSHA256)
	if e != nil || different.ModelIdentitySHA256 == baseIdentity {
		t.Fatal("model identity unchanged", e)
	}
	for _, kind := range []string{"revision", "consent", "prefix", "mode", "tie", "geometry", "asset", "resources"} {
		bad := c
		y := communityConfig(t)
		switch kind {
		case "revision":
			y.ModelRevision = "bad"
		case "consent":
			y.AllowExperimental = false
		case "prefix":
			y.EmbeddingPrefix = ""
		case "mode":
			y.Modes.Head = "auto"
		case "tie":
			y.PCM.TiePolicy = "auto"
		case "geometry":
			y.PLDAConfig.InputDim = 2
		case "asset":
			y.PLDA.Path = "relative"
		case "resources":
			bad.Resources = nil
		}
		bad.Profile.Community = &y
		if e := bad.validate(); e == nil {
			t.Fatal(kind)
		}
	}
}
func TestInspectCommunityMetadataRejectsContainers(t *testing.T) {
	c := baseConfig(t)
	r := ResourceSettings{CPUSlots: 2, MemoryBytes: 1 << 30, MaxWaiting: 4, LoadBytes: 512 << 20, ResidentBytes: 256 << 20, WorkBytes: 256 << 20}
	c.Resources = &r
	x := communityConfig(t)
	c.Profile.Community = &x
	if e := inspectCommunityMetadata(context.Background(), c); e != nil {
		t.Fatal(e)
	}
	// NPZ signature is checked only after tensor sources; direct replacement of a
	// real path is covered by loader tests, while this verifies bounded hash input.
	x.Segmentation = putAsset(t, t.TempDir(), "empty", []byte("bad"), 0600)
	c.Profile.Community = &x
	if e := inspectCommunityMetadata(context.Background(), c); e == nil {
		t.Fatal("bad asset accepted")
	}
}

type fakeCommunityOwner struct {
	stage speechjob.Stage
	close int
	fail  bool
}

func (o *fakeCommunityOwner) Stage() speechjob.Stage { return o.stage }
func (o *fakeCommunityOwner) Close(context.Context) error {
	o.close++
	if o.fail && o.close == 1 {
		return io.ErrClosedPipe
	}
	return nil
}
func TestCommunityPrepareRuntimeFailureAndOwner(t *testing.T) {
	c := baseConfig(t)
	rset := ResourceSettings{CPUSlots: 2, MemoryBytes: 1 << 30, MaxWaiting: 4, LoadBytes: 512 << 20, ResidentBytes: 256 << 20, WorkBytes: 256 << 20}
	c.Resources = &rset
	x := communityConfig(t)
	c.Profile.Community = &x
	// Production source opening succeeds; missing constructors fail before inference.
	if _, e := prepareCommunity(context.Background(), c, communityRuntime{}); e == nil {
		t.Fatal("missing constructor")
	}
	// regularFile rejects symlink even when target hash is configured.
	link := filepath.Join(t.TempDir(), "link")
	if e := os.Symlink(x.Segmentation.Path, link); e == nil {
		x.Segmentation.Path = link
		c.Profile.Community = &x
		if _, e := prepareCommunity(context.Background(), c, communityRuntime{}); e == nil {
			t.Fatal("symlink")
		}
	}
}
func TestBuiltProfilesClosesOwnersReverseAndRetries(t *testing.T) {
	a := &fakeCommunityOwner{fail: true}
	b := &fakeCommunityOwner{}
	p := &builtProfiles{owners: []stageOwner{a, b}}
	closeBuiltProfiles(p)
	if a.close != 2 || b.close != 1 || p.owners[0] != nil || p.owners[1] != nil {
		t.Fatal(a.close, b.close, p.owners)
	}
	if e := p.Close(context.Background()); e != nil {
		t.Fatal(e)
	}
}
func TestCommunityServerJSONRejectsGoFieldNames(t *testing.T) {
	c := baseConfig(t)
	r := ResourceSettings{CPUSlots: 2, MemoryBytes: 1 << 30, MaxWaiting: 4, LoadBytes: 512 << 20, ResidentBytes: 256 << 20, WorkBytes: 256 << 20}
	c.Resources = &r
	x := communityConfig(t)
	c.Profile.Community = &x
	b, e := jsonMarshal(c)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = parseConfig(b); e != nil {
		t.Fatal(e)
	}
	bad := bytesReplace(b, []byte(`"input_size"`), []byte(`"InputSize"`))
	if _, e = parseConfig(bad); e == nil {
		t.Fatal("Go-name alias")
	}
}

// Tiny wrappers avoid adding broad imports to production tests.
func jsonMarshal(v any) ([]byte, error)  { return json.Marshal(v) }
func bytesReplace(b, a, c []byte) []byte { return bytes.Replace(b, a, c, 1) }

var _ = errors.Is

func TestPrepareCommunityPositiveProductionIO(t *testing.T) {
	c := baseConfig(t)
	rset := ResourceSettings{CPUSlots: 2, MemoryBytes: 1 << 30, MaxWaiting: 4, LoadBytes: 512 << 20, ResidentBytes: 256 << 20, WorkBytes: 256 << 20}
	c.Resources = &rset
	x := communityConfig(t)
	c.Profile.Community = &x
	owner := &fakeCommunityOwner{stage: speechjob.Stage{Name: "diarization", Version: hashBytes([]byte("diar")), Run: func(context.Context, *speechjob.Input, io.Writer) error { return nil }}}
	calls := []string{}
	runtime := communityRuntime{
		loadSeg: func(context.Context, c1.SegmentationTensorSource, c1.SegmentationLoadConfig) (*c1.SegmentationCheckpoint, error) {
			calls = append(calls, "load-seg")
			return &c1.SegmentationCheckpoint{}, nil
		},
		newSeg: func(context.Context, *c1.SegmentationCheckpoint, []float32) (*c1.ExperimentalSegmentation, error) {
			calls = append(calls, "new-seg")
			return &c1.ExperimentalSegmentation{}, nil
		},
		loadEmb: func(context.Context, c1.WeSpeakerTensorSource, c1.WeSpeakerResNetConfig, string) (*c1.WeSpeakerResNet34, error) {
			calls = append(calls, "load-emb")
			return &c1.WeSpeakerResNet34{}, nil
		},
		newEmb: func(context.Context, *c1.WeSpeakerResNet34) (*c1.ExperimentalEmbedding, error) {
			calls = append(calls, "new-emb")
			return &c1.ExperimentalEmbedding{}, nil
		},
		loadPLDA: func(context.Context, io.ReaderAt, int64, io.ReaderAt, int64, c1.PLDAConfig) (*c1.RawPLDAPreparation, error) {
			calls = append(calls, "load-plda")
			return &c1.RawPLDAPreparation{Model: &c1.PreparedPLDA{}}, nil
		},
		newModel: func(context.Context, *c1.ExperimentalSegmentation, *c1.ExperimentalEmbedding, *c1.PreparedPLDA) (*c1.ExperimentalDiarization, error) {
			calls = append(calls, "new-model")
			return &c1.ExperimentalDiarization{}, nil
		},
		newOwner: func(*c1.ExperimentalDiarization, speechjob.Community1StageConfig) (stageOwner, error) {
			calls = append(calls, "new-owner")
			return owner, nil
		},
	}
	got, e := prepareCommunity(context.Background(), c, runtime)
	if e != nil || got != owner || strings.Join(calls, ",") != "load-seg,new-seg,load-emb,new-emb,load-plda,new-model,new-owner" {
		t.Fatal(got, e, calls)
	}
	runtime.newOwner = func(*c1.ExperimentalDiarization, speechjob.Community1StageConfig) (stageOwner, error) {
		return nil, nil
	}
	if got, e = prepareCommunity(context.Background(), c, runtime); e == nil || got != nil {
		t.Fatal("nil CPU owner accepted", got, e)
	}
	partial := &fakeCommunityOwner{}
	runtime.newOwner = func(*c1.ExperimentalDiarization, speechjob.Community1StageConfig) (stageOwner, error) {
		return partial, io.ErrClosedPipe
	}
	if got, e = prepareCommunity(context.Background(), c, runtime); !errors.Is(e, io.ErrClosedPipe) || got != nil || partial.close != 1 {
		t.Fatal("partial CPU owner not closed", got, e, partial.close)
	}
}

func TestCombinedProfileStagesAndReverseOwners(t *testing.T) {
	c := toyAssets(t)
	rset := ResourceSettings{CPUSlots: 2, MemoryBytes: 1 << 30, MaxWaiting: 4, LoadBytes: 512 << 20, ResidentBytes: 256 << 20, WorkBytes: 256 << 20}
	c.Resources = &rset
	x := communityConfig(t)
	c.Profile.Community = &x
	communityOwner := &fakeCommunityOwner{stage: speechjob.Stage{Name: "diarization", Version: hashBytes([]byte("diar")), Run: func(context.Context, *speechjob.Input, io.Writer) error { return nil }}}
	cr := communityRuntime{loadSeg: func(context.Context, c1.SegmentationTensorSource, c1.SegmentationLoadConfig) (*c1.SegmentationCheckpoint, error) {
		return &c1.SegmentationCheckpoint{}, nil
	}, newSeg: func(context.Context, *c1.SegmentationCheckpoint, []float32) (*c1.ExperimentalSegmentation, error) {
		return &c1.ExperimentalSegmentation{}, nil
	}, loadEmb: func(context.Context, c1.WeSpeakerTensorSource, c1.WeSpeakerResNetConfig, string) (*c1.WeSpeakerResNet34, error) {
		return &c1.WeSpeakerResNet34{}, nil
	}, newEmb: func(context.Context, *c1.WeSpeakerResNet34) (*c1.ExperimentalEmbedding, error) {
		return &c1.ExperimentalEmbedding{}, nil
	}, loadPLDA: func(context.Context, io.ReaderAt, int64, io.ReaderAt, int64, c1.PLDAConfig) (*c1.RawPLDAPreparation, error) {
		return &c1.RawPLDAPreparation{Model: &c1.PreparedPLDA{}}, nil
	}, newModel: func(context.Context, *c1.ExperimentalSegmentation, *c1.ExperimentalEmbedding, *c1.PreparedPLDA) (*c1.ExperimentalDiarization, error) {
		return &c1.ExperimentalDiarization{}, nil
	}, newOwner: func(*c1.ExperimentalDiarization, speechjob.Community1StageConfig) (stageOwner, error) {
		return communityOwner, nil
	}}
	built, e := buildProfileOwnedRuntimes(context.Background(), c, true, profileRuntimes{defaultVulkanProfileRuntime(), cr})
	if e != nil {
		t.Fatal(e)
	}
	if len(built.Profiles) != 1 || len(built.Profiles[0].Stages) != 7 || len(built.owners) != 1 {
		t.Fatal(built)
	}
	want := []string{"decode", "asr-windows", "transcript", "vtt", "diarization", "speaker-transcript", "speaker-vtt"}
	for i, s := range built.Profiles[0].Stages {
		if s.Name != want[i] {
			t.Fatal(i, s.Name)
		}
	}
	if e = built.Close(context.Background()); e != nil || communityOwner.close != 1 {
		t.Fatal(e, communityOwner.close)
	}
}
func TestCommunityVulkanConfigAndInjectedConstruction(t *testing.T) {
	c := baseConfig(t)
	rset := ResourceSettings{CPUSlots: 2, MemoryBytes: 1 << 30, MaxWaiting: 4, LoadBytes: 512 << 20, ResidentBytes: 256 << 20, WorkBytes: 256 << 20}
	c.Resources = &rset
	x := communityConfig(t)
	v := CommunityVulkanSettings{Enable: true, AllowExperimental: true, DeviceContains: "fixture-device", BackendSHA256: hashBytes([]byte("community-vulkan")), DrainMilliseconds: 10}
	x.Vulkan = &v
	x.Modes.LSTM = "vulkan"
	x.Modes.Embedding = "vulkan"
	c.Profile.Community = &x
	if e := c.validate(); e != nil {
		t.Fatal(e)
	}
	for _, kind := range []string{"enable", "consent", "device", "hash", "poll", "lstm-mode", "embedding-mode", "shared-device"} {
		bad := c
		y := x
		z := v
		switch kind {
		case "enable":
			z.Enable = false
		case "consent":
			z.AllowExperimental = false
		case "device":
			z.DeviceContains = ""
		case "hash":
			z.BackendSHA256 = "bad"
		case "poll":
			z.DrainMilliseconds = 0
		case "lstm-mode":
			y.Modes.LSTM = "simd"
		case "embedding-mode":
			y.Modes.Embedding = "gemm"
		case "shared-device":
			bad.Profile.Vulkan = &VulkanSettings{Enable: true, AllowExperimental: true, DeviceContains: "fixture", BackendSHA256: hashBytes([]byte("whisper-vulkan")), DrainMilliseconds: 10}
		}
		y.Vulkan = &z
		bad.Profile.Community = &y
		if e := bad.validate(); e == nil {
			t.Fatal(kind)
		}
	}
	owner := &fakeCommunityOwner{stage: speechjob.Stage{Name: "diarization", Version: hashBytes([]byte("vk-diar")), Run: func(context.Context, *speechjob.Input, io.Writer) error { return nil }}}
	calls := []string{}
	runtime := communityRuntime{
		loadSeg: func(context.Context, c1.SegmentationTensorSource, c1.SegmentationLoadConfig) (*c1.SegmentationCheckpoint, error) {
			calls = append(calls, "load-seg")
			return &c1.SegmentationCheckpoint{}, nil
		},
		loadEmb: func(context.Context, c1.WeSpeakerTensorSource, c1.WeSpeakerResNetConfig, string) (*c1.WeSpeakerResNet34, error) {
			calls = append(calls, "load-emb")
			return &c1.WeSpeakerResNet34{}, nil
		},
		loadPLDA: func(context.Context, io.ReaderAt, int64, io.ReaderAt, int64, c1.PLDAConfig) (*c1.RawPLDAPreparation, error) {
			calls = append(calls, "load-plda")
			return &c1.RawPLDAPreparation{Model: &c1.PreparedPLDA{}}, nil
		},
		initVulkan: func() bool { calls = append(calls, "init"); return true },
		deviceName: func() string { calls = append(calls, "device"); return "fixture-device-1" },
		newVulkanModel: func(context.Context, *c1.SegmentationCheckpoint, []float32, *c1.WeSpeakerResNet34, *c1.PreparedPLDA, int) (*c1.VulkanDiarization, error) {
			calls = append(calls, "new-vulkan-model")
			return &c1.VulkanDiarization{}, nil
		},
		newVulkanOwner: func(*c1.VulkanDiarization, speechjob.VulkanCommunity1StageConfig) (stageOwner, error) {
			calls = append(calls, "new-vulkan-owner")
			return owner, nil
		},
	}
	got, e := prepareCommunity(context.Background(), c, runtime)
	if e != nil || got != owner || strings.Join(calls, ",") != "load-seg,load-emb,load-plda,init,device,new-vulkan-model,new-vulkan-owner" {
		t.Fatal(got, e, calls)
	}
	runtime.newVulkanOwner = func(*c1.VulkanDiarization, speechjob.VulkanCommunity1StageConfig) (stageOwner, error) {
		return nil, nil
	}
	if got, e = prepareCommunity(context.Background(), c, runtime); e == nil || got != nil {
		t.Fatal("nil Vulkan owner accepted", got, e)
	}
	partial := &fakeCommunityOwner{}
	runtime.newVulkanOwner = func(*c1.VulkanDiarization, speechjob.VulkanCommunity1StageConfig) (stageOwner, error) {
		return partial, io.ErrClosedPipe
	}
	if got, e = prepareCommunity(context.Background(), c, runtime); !errors.Is(e, io.ErrClosedPipe) || got != nil || partial.close != 1 {
		t.Fatal("partial Vulkan owner not closed", got, e, partial.close)
	}
}

func TestCommunityMetadataCheckDoesNotConstructModels(t *testing.T) {
	c := toyAssets(t)
	r := ResourceSettings{CPUSlots: 2, MemoryBytes: 1 << 30, MaxWaiting: 4, LoadBytes: 512 << 20, ResidentBytes: 256 << 20, WorkBytes: 256 << 20}
	c.Resources = &r
	x := communityConfig(t)
	c.Profile.Community = &x
	if e := inspectCommunityMetadata(context.Background(), c); e != nil {
		t.Fatal(e)
	}
}
