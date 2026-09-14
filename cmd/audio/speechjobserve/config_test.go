package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func hashBytes(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func baseConfig(t *testing.T) ServerConfig {
	t.Helper()
	root := t.TempDir()
	a := Asset{Path: filepath.Join(root, "asset"), SHA256: hashBytes([]byte("asset"))}
	return ServerConfig{Schema: 2, Store: filepath.Join(root, "store"), RuntimeSHA256: hashBytes([]byte("runtime")), Threads: 2, Limits: Limits{Jobs: 8, UploadBytes: 1 << 20, ArtifactBytes: 1 << 20, StoreBytes: 8 << 20, WeightBytes: 8 << 20, OwnedWeightBytes: 16 << 20}, HTTP: HTTPSettings{Listen: "127.0.0.1:8099", Hosts: []string{"127.0.0.1:8099"}, AllowLoopbackHTTP: true, RequestSeconds: 10, HeaderSeconds: 1, IdleSeconds: 1, ShutdownSeconds: 1, MaxRequests: 2, MaxConnections: 4}, Weights: a, ModelConfig: a, Tokenizer: a, Generation: a, Profile: ProfileSettings{ID: "asr-pt", Language: "pt", Extension: ".wav", MediaBackend: "go264", MaxDurationSeconds: 1, DecodeBytes: 90000, WindowBytes: 4096, ResultBytes: 128 << 10}}
}
func TestConfigStrictAndSafeListener(t *testing.T) {
	good := baseConfig(t)
	b, _ := json.Marshal(good)
	parsed, e := parseConfig(b)
	if e != nil || parsed.Profile.WordTimestamps {
		t.Fatal(e)
	}
	good.Profile.WordTimestamps = true
	b, _ = json.Marshal(good)
	parsed, e = parseConfig(b)
	if e != nil || !parsed.Profile.WordTimestamps {
		t.Fatal("word timestamp profile option", e)
	}
	good.Profile.WordTimestamps = false
	b, _ = json.Marshal(good)
	for _, raw := range [][]byte{[]byte("null"), append(b, []byte(" {}")...), bytes.Replace(b, []byte(`"schema":2`), []byte(`"schema":2,"schema":2`), 1), bytes.Replace(b, []byte(`"schema":2`), []byte(`"Schema":2`), 1), bytes.Replace(b, []byte(`"threads":2`), []byte(`"threads":null`), 1), bytes.Replace(b, []byte(`"profile":{`), []byte(`"profile":{"vulkan":{"encoder_weights":"q8-kv-mlp"},`), 1), append([]byte(strings.Repeat(" ", 64<<10)), b...)} {
		if _, e := parseConfig(raw); e == nil {
			t.Fatal("ambiguous config")
		}
	}
	for _, kind := range []string{"schema", "store", "hash", "thread", "weight", "owned", "upload", "jobs", "listen", "public-http", "http-optin", "tls-pair", "hosts", "origin", "request", "header", "connections", "language", "extension", "media-backend", "media-backend-missing", "ffmpeg-missing", "ffmpeg-assets-with-go264", "result", "overlap"} {
		c := good
		switch kind {
		case "schema":
			c.Schema = 1
		case "store":
			c.Store = "relative"
		case "hash":
			c.RuntimeSHA256 = "bad"
		case "thread":
			c.Threads = 0
		case "weight":
			c.Limits.WeightBytes = 1
		case "owned":
			c.Limits.OwnedWeightBytes = 0
		case "upload":
			c.Limits.UploadBytes = 1 << 30
		case "jobs":
			c.Limits.Jobs = 1001
		case "listen":
			c.HTTP.Listen = "localhost:8099"
		case "public-http":
			c.HTTP.Listen = "0.0.0.0:8099"
		case "http-optin":
			c.HTTP.AllowLoopbackHTTP = false
		case "tls-pair":
			c.HTTP.TLSCert = "/cert"
		case "hosts":
			c.HTTP.Hosts = []string{"user@evil"}
		case "origin":
			c.HTTP.Origin = "https://example/a"
		case "request":
			c.HTTP.RequestSeconds = 0
		case "header":
			c.HTTP.HeaderSeconds = 11
		case "connections":
			c.HTTP.MaxConnections = 2
		case "language":
			c.Profile.Language = "P!"
		case "extension":
			c.Profile.Extension = ".mp3"
		case "media-backend":
			c.Profile.MediaBackend = "unknown"
		case "media-backend-missing":
			c.Profile.MediaBackend = ""
		case "ffmpeg-missing":
			c.Profile.MediaBackend = "ffmpeg"
		case "ffmpeg-assets-with-go264":
			c.FFmpeg = Asset{Path: "/ffmpeg", SHA256: hashBytes([]byte("ffmpeg"))}
		case "result":
			c.Profile.ResultBytes = 1
		case "overlap":
			c.Profile.OverlapSamples = -1
		}
		if e := c.validate(); e == nil {
			t.Fatal(kind)
		}
	}
	// Tokenizer keys are case-sensitive and HF optional fields may be null. Exact
	// duplicate keys are still forbidden at every nested level.
	if e := uniqueJSON([]byte(`{"A":1,"a":2,"nullable":null}`), 8, false, true); e != nil {
		t.Fatal(e)
	}
	if e := uniqueJSON([]byte(`{"A":1,"\u0041":2}`), 8, false, true); e == nil {
		t.Fatal("escaped duplicate")
	}
}
func TestConfigProfileSetCompatibilityAndBounds(t *testing.T) {
	c := baseConfig(t)
	base := c.Profile
	c.Profile = ProfileSettings{}
	for _, kind := range []string{"asr", "diar"} {
		for _, language := range []string{"auto", "en", "pt", "fr"} {
			for _, extension := range []string{".m4a", ".wav"} {
				profile := base
				profile.ID = kind + "-" + language + "-" + strings.TrimPrefix(extension, ".")
				profile.Language = language
				profile.Extension = extension
				c.Profiles = append(c.Profiles, profile)
			}
		}
	}
	if err := c.validate(); err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(c)
	parsed, err := parseConfig(data)
	if err != nil || len(parsed.Profiles) != 16 || parsed.Profile.ID != c.Profiles[0].ID {
		t.Fatal(len(parsed.Profiles), parsed.Profile.ID, err)
	}
	if _, err = parsed.configuredProfiles(); err != nil {
		t.Fatal("parsed profile set cannot be reused", err)
	}
	for _, mutate := range []func(*ServerConfig){
		func(c *ServerConfig) { c.Profile = base },
		func(c *ServerConfig) { c.Profiles[1].ID = c.Profiles[0].ID },
		func(c *ServerConfig) { c.Profiles[1].OverlapSamples++ },
		func(c *ServerConfig) { c.Profiles[15].Community = &CommunitySettings{} },
		func(c *ServerConfig) { c.Profiles = append(c.Profiles, base) },
	} {
		bad := c
		bad.Profiles = append([]ProfileSettings(nil), c.Profiles...)
		mutate(&bad)
		if err := bad.validate(); err == nil {
			t.Fatal("accepted invalid profile set")
		}
	}
}

func TestAssetHashBoundsAndCancellation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "asset")
	data := []byte("bounded fixture")
	if e := os.WriteFile(path, data, 0600); e != nil {
		t.Fatal(e)
	}
	a := Asset{path, hashBytes(data)}
	if e := verifyAsset(context.Background(), a, 64, false); e != nil {
		t.Fatal(e)
	}
	if e := verifyAsset(context.Background(), a, 1, false); e == nil {
		t.Fatal("size")
	}
	if e := verifyAsset(context.Background(), a, 64, true); e == nil {
		t.Fatal("not executable")
	}
	wrong := a
	wrong.SHA256 = hashBytes([]byte("other"))
	if e := verifyAsset(context.Background(), wrong, 64, false); e == nil {
		t.Fatal("hash")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if e := verifyAsset(ctx, a, 64, false); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	link := path + "-link"
	if e := os.Symlink(path, link); e == nil {
		if _, e := regularFile(link, 64); e == nil {
			t.Fatal("symlink")
		}
	}
	var out bytes.Buffer
	if e := command(context.Background(), []string{"--help"}, &out); e != nil || !strings.Contains(out.String(), "--check") {
		t.Fatal(out.String(), e)
	}
	if e := command(context.Background(), []string{"--token", "secret"}, io.Discard); e == nil || strings.Contains(e.Error(), "secret") {
		t.Fatal(e)
	}
}

func TestUIRequiresExactServerOrigin(t *testing.T) {
	c := baseConfig(t)
	c.HTTP.EnableUI = true
	if e := c.validate(); e == nil {
		t.Fatal("UI enabled without origin")
	}
	c.HTTP.Origin = "http://" + c.HTTP.Listen
	if e := c.validate(); e != nil {
		t.Fatal(e)
	}
	c.HTTP.Origin = "https://" + c.HTTP.Listen
	if e := c.validate(); e == nil {
		t.Fatal("TLS origin on plain listener")
	}
	c.HTTP.Origin = "http://" + c.HTTP.Listen
	c.HTTP.MaxRequests = 1
	if e := c.validate(); e == nil {
		t.Fatal("no status slot while run active")
	}
}

func TestQueueWorkerRequiresSeparateConsent(t *testing.T) {
	c := baseConfig(t)
	c.Queue = QueueSettings{Enable: true, Directory: filepath.Join(t.TempDir(), "queue"), MaxEntries: 8, MaxBytes: 1 << 20, JobSeconds: 30}
	if e := c.validate(); e != nil {
		t.Fatal(e)
	}
	c.Queue.StartWorker = true
	if e := c.validate(); e != nil {
		t.Fatal(e)
	}
	c.Queue.Enable = false
	if e := c.validate(); e == nil {
		t.Fatal("worker without enabled queue")
	}
	c.Queue.Enable = true
	c.HTTP.EnableUI = true
	c.HTTP.Origin = "http://" + c.HTTP.Listen
	if e := c.validate(); e == nil {
		t.Fatal("synchronous UI in queue mode")
	}
}

func TestResourceConfigExplicitEstimates(t *testing.T) {
	c := baseConfig(t)
	good := ResourceSettings{CPUSlots: 2, MemoryBytes: 64 << 20, MaxWaiting: 4, LoadBytes: 32 << 20, ResidentBytes: 16 << 20, WorkBytes: 16 << 20}
	c.Resources = &good
	b, _ := json.Marshal(c)
	if _, e := parseConfig(b); e != nil {
		t.Fatal(e)
	}
	for _, kind := range []string{"cpu", "zero-memory", "load", "resident", "work", "waiting", "negative"} {
		r := good
		switch kind {
		case "cpu":
			r.CPUSlots = 1
		case "zero-memory":
			r.MemoryBytes = 0
		case "load":
			r.LoadBytes = 8 << 20
		case "resident":
			r.ResidentBytes = 8 << 20
		case "work":
			r.WorkBytes = 50 << 20
		case "waiting":
			r.MaxWaiting = 129
		case "negative":
			r.WorkBytes = -1
		}
		c.Resources = &r
		if e := c.validate(); e == nil {
			t.Fatal(kind)
		}
	}
}

func TestVulkanConfigRequiresExplicitConsentAndResources(t *testing.T) {
	c := baseConfig(t)
	r := ResourceSettings{CPUSlots: 2, MemoryBytes: 64 << 20, MaxWaiting: 4, LoadBytes: 32 << 20, ResidentBytes: 16 << 20, WorkBytes: 16 << 20}
	c.Resources = &r
	good := VulkanSettings{Enable: true, AllowExperimental: true, DeviceContains: "Intel fixture", BackendSHA256: hashBytes([]byte("backend")), DrainMilliseconds: 10}
	c.Profile.Vulkan = &good
	if e := c.validate(); e != nil {
		t.Fatal(e)
	}
	b, _ := json.Marshal(c)
	parsed, e := parseConfig(b)
	if e != nil || parsed.Profile.Vulkan == nil || *parsed.Profile.Vulkan != good {
		t.Fatal(parsed, e)
	}
	for _, kind := range []string{"disabled", "consent", "device", "device-control", "hash", "poll-zero", "poll-large", "resources"} {
		v := good
		bad := c
		switch kind {
		case "disabled":
			v.Enable = false
		case "consent":
			v.AllowExperimental = false
		case "device":
			v.DeviceContains = ""
		case "device-control":
			v.DeviceContains = "x\n"
		case "hash":
			v.BackendSHA256 = "bad"
		case "poll-zero":
			v.DrainMilliseconds = 0
		case "poll-large":
			v.DrainMilliseconds = 30001
		case "resources":
			bad.Resources = nil
		}
		bad.Profile.Vulkan = &v
		if e := bad.validate(); e == nil {
			t.Fatal(kind)
		}
	}
}
