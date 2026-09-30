//go:build linux && amd64

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

func nemotronFixture(t *testing.T) (ServerConfig, []byte) {
	t.Helper()
	c := baseConfig(t)
	c.Resources = &ResourceSettings{CPUSlots: 2, MemoryBytes: 1 << 30, MaxWaiting: 4, LoadBytes: 512 << 20, ResidentBytes: 256 << 20, WorkBytes: 256 << 20}
	header := make(map[string]safetensors.TensorInfo)
	end := 0
	for _, name := range []string{"model.audio_tower.input_layer_norm.weight", "model.audio_tower.layers.30.layer_norm1.weight", "silence_embeds", "classifier.out_proj.bias", "test0", "test1", "test2", "test3", "test4", "test5"} {
		width := 1
		if name == "classifier.out_proj.bias" {
			width = 8
		} else if name == "silence_embeds" || name == "model.audio_tower.input_layer_norm.weight" || name == "model.audio_tower.layers.30.layer_norm1.weight" {
			width = 512
		}
		header[name] = safetensors.TensorInfo{DType: "F32", Shape: []int{width}, DataOffsets: [2]int{end, end + width*4}}
		end += width * 4
	}
	h, _ := json.Marshal(header)
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint64(len(h)))
	buf.Write(h)
	buf.Write(make([]byte, end))
	data := buf.Bytes()
	c.Profile.Nemotron = &NemotronSettings{Enable: true, AllowExperimental: true, ModelRevision: speechjob.NemotronDiarizationRevision, Model: putAsset(t, t.TempDir(), "model.safetensors", data, 0600), MaxResultBytes: 16 << 20}
	return c, data
}

func TestNemotronConfigAndMetadataModelFree(t *testing.T) {
	c, _ := nemotronFixture(t)
	data, _ := json.Marshal(c)
	if _, err := parseConfig(data); err != nil {
		t.Fatal(err)
	}
	if err := inspectNemotronMetadata(context.Background(), c); err != nil {
		t.Fatal(err)
	}
	// Representative boundaries alone cannot pass complete model loading.
	if _, err := prepareNemotron(context.Background(), c); err == nil {
		t.Fatal("incomplete model loaded")
	}
	for _, change := range []func(*ServerConfig){
		func(c *ServerConfig) { c.Profile.Nemotron.ModelRevision = "wrong" },
		func(c *ServerConfig) { c.Profile.Nemotron.AllowExperimental = false },
		func(c *ServerConfig) { c.Profile.Nemotron.Enable = false },
		func(c *ServerConfig) { c.Profile.Nemotron.MaxResultBytes = 16<<20 + 1 },
		func(c *ServerConfig) { c.Profile.Nemotron.Model.Path = "relative" },
		func(c *ServerConfig) { c.Profile.Nemotron.Model.SHA256 = "wrong" },
		func(c *ServerConfig) { c.Profile.Community = &CommunitySettings{} },
		func(c *ServerConfig) { c.Resources = nil },
	} {
		bad := c
		x := *c.Profile.Nemotron
		bad.Profile.Nemotron = &x
		change(&bad)
		data, _ = json.Marshal(bad)
		if _, err := parseConfig(data); err == nil {
			t.Fatal("invalid configuration accepted")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := inspectNemotronMetadata(ctx, c); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	bad := c
	bad.Limits.OwnedWeightBytes = 16
	if err := inspectNemotronMetadata(context.Background(), bad); err == nil {
		t.Fatal("unbounded allocation accepted")
	}
	bad = c
	bad.Limits.WeightBytes = 8
	if err := inspectNemotronMetadata(context.Background(), bad); err == nil {
		t.Fatal("unbounded asset accepted")
	}
}

func TestNemotronOwnerSerializesDrainsAndReleasesRun(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	owner := newNemotronStageOwner(speechjob.Stage{Name: "diarization", Run: func(context.Context, *speechjob.Input, io.Writer) error {
		calls.Add(1)
		close(entered)
		<-release
		return nil
	}})
	stage := owner.Stage()
	done := make(chan error, 1)
	go func() { done <- stage.Run(context.Background(), nil, io.Discard) }()
	<-entered
	closed := make(chan error, 1)
	go func() { closed <- owner.Close(context.Background()) }()
	select {
	case <-closed:
		t.Fatal("owner closed before run drained")
	case <-time.After(10 * time.Millisecond):
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if err := <-closed; err != nil {
		t.Fatal(err)
	}
	if owner.run != nil || calls.Load() != 1 {
		t.Fatal("model run retained")
	}
	if err := stage.Run(context.Background(), nil, io.Discard); !errors.Is(err, speechjob.ErrClosed) {
		t.Fatal(err)
	}
	if err := owner.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestNemotronPinnedMetadata(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_METADATA") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_METADATA=1")
	}
	c := baseConfig(t)
	c.Limits.WeightBytes, c.Limits.OwnedWeightBytes = 2<<30, 2<<30
	c.Profile.Nemotron = &NemotronSettings{Enable: true, AllowExperimental: true, ModelRevision: speechjob.NemotronDiarizationRevision, Model: Asset{Path: os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL"), SHA256: "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"}, MaxResultBytes: 16 << 20}
	if err := inspectNemotronMetadata(context.Background(), c); err != nil {
		t.Fatal(err)
	}
	t.Log("pinned asset hash and bounded tensor geometry verified; model not loaded")
}

func TestNemotronCandidateDeploymentConfig(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_CANDIDATE_CONFIG")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_CANDIDATE_CONFIG")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	c, err := parseConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	profiles, err := c.configuredProfiles()
	if err != nil || len(profiles) != 36 {
		t.Fatal(len(profiles), err)
	}
	var asr, community, nemotron int
	for _, p := range profiles {
		switch {
		case p.Nemotron != nil:
			nemotron++
		case p.Community != nil:
			community++
		default:
			asr++
		}
	}
	if asr != 12 || community != 12 || nemotron != 12 {
		t.Fatal(asr, community, nemotron)
	}
	t.Logf("candidate parse passed: bytes=%d profiles=%d; runtime SHA still requires final build", len(data), len(profiles))
}

func TestNemotronProfileSetKeepsLegacyIdentity(t *testing.T) {
	c, _ := nemotronFixture(t)
	nemotron := c.Profile
	nemotron.ID = "nem-pt-wav"
	asr := c.Profile
	asr.Nemotron = nil
	asr.ID = "asr-pt-wav"
	c.Profile = ProfileSettings{}
	c.Profiles = []ProfileSettings{asr, nemotron}
	profiles, err := c.configuredProfiles()
	if err != nil || len(profiles) != 2 || profiles[0] != asr {
		t.Fatal(profiles, err)
	}
	extra := nemotron
	x := *extra.Nemotron
	x.Model.SHA256 = hashBytes([]byte("different"))
	extra.Nemotron = &x
	extra.ID = "nem-en-wav"
	extra.Language = "en"
	c.Profiles = append(c.Profiles, extra)
	if _, err := c.configuredProfiles(); err == nil {
		t.Fatal("different provider models accepted")
	}
}
