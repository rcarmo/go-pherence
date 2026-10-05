//go:build linux && amd64

package main

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

func TestNemotronASRConfigRejectsImplicitAlignmentAndGPU(t *testing.T) {
	c := baseConfig(t)
	c.Resources = &ResourceSettings{CPUSlots: 2, MemoryBytes: 1 << 30, LoadBytes: 512 << 20, ResidentBytes: 256 << 20, WorkBytes: 256 << 20}
	c.NemotronASR = &NemotronASRSettings{Enable: true, AllowExperimental: true, ModelRevision: speechjob.NemotronASRRevision}
	c.Profile.WordTimestamps = false
	data, _ := json.Marshal(c)
	if _, err := parseConfig(data); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*ServerConfig){func(c *ServerConfig) { c.Profile.WordTimestamps = true }, func(c *ServerConfig) { c.Profile.Vulkan = &VulkanSettings{} }, func(c *ServerConfig) { c.Profile.Language = "xx" }, func(c *ServerConfig) { c.NemotronASR.ModelRevision = "wrong" }, func(c *ServerConfig) { c.NemotronASR.AllowExperimental = false }, func(c *ServerConfig) { c.Resources = nil }} {
		bad := c
		copy := *c.NemotronASR
		bad.NemotronASR = &copy
		change(&bad)
		data, _ = json.Marshal(bad)
		if _, err := parseConfig(data); err == nil {
			t.Fatal("invalid Nemotron ASR config accepted")
		}
	}
}

// Opt-in pinned metadata check, no tensor payload loading or server listener.
func TestNemotronASRCandidateMetadata(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_CANDIDATE_CONFIG")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_CANDIDATE_CONFIG")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	c, err := parseConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	if c.NemotronASR == nil {
		t.Fatal("ASR missing")
	}
	if _, err = buildProfileOwnedRuntimes(context.Background(), c, false, profileRuntimes{}); err != nil {
		t.Fatal(err)
	}
	t.Log("Nemotron ASR pinned metadata checked without model load")
}

func TestNemotronASRSharedProjectionConfiguration(t *testing.T) {
	c := baseConfig(t)
	c.NemotronASR = &NemotronASRSettings{Enable: true, AllowExperimental: true, ModelRevision: speechjob.NemotronASRRevision}
	c.Resources = &ResourceSettings{CPUSlots: 2, MemoryBytes: 1 << 30, LoadBytes: 512 << 20, ResidentBytes: 256 << 20, WorkBytes: 256 << 20}
	c.Profile.WordTimestamps = false
	c.NemotronASR.Projection = &NemotronASRProjectionSettings{Backend: "vulkan-shared", AllowExperimental: true, DeviceContains: "Intel", BackendSHA256: hashBytes([]byte("backend"))}
	if err := c.validate(); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*NemotronASRProjectionSettings){
		func(p *NemotronASRProjectionSettings) { p.Backend = "vulkan" },
		func(p *NemotronASRProjectionSettings) { p.AllowExperimental = false },
		func(p *NemotronASRProjectionSettings) { p.DeviceContains = " " },
		func(p *NemotronASRProjectionSettings) { p.BackendSHA256 = "bad" },
	} {
		x := *c.NemotronASR.Projection
		mutate(&x)
		test := c
		settings := *c.NemotronASR
		settings.Projection = &x
		test.NemotronASR = &settings
		if test.validate() == nil {
			t.Fatal("accepted invalid shared backend")
		}
	}
}
