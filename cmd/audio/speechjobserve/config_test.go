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
	return ServerConfig{Schema: 1, Store: filepath.Join(root, "store"), RuntimeSHA256: hashBytes([]byte("runtime")), Threads: 2, Limits: Limits{Jobs: 8, UploadBytes: 1 << 20, ArtifactBytes: 1 << 20, StoreBytes: 8 << 20, WeightBytes: 8 << 20, OwnedWeightBytes: 16 << 20}, HTTP: HTTPSettings{Listen: "127.0.0.1:8099", Hosts: []string{"127.0.0.1:8099"}, AllowLoopbackHTTP: true, RequestSeconds: 10, HeaderSeconds: 1, IdleSeconds: 1, ShutdownSeconds: 1, MaxRequests: 2, MaxConnections: 4}, Weights: a, ModelConfig: a, Tokenizer: a, Generation: a, FFmpeg: a, FFprobe: a, Profile: ProfileSettings{ID: "asr-pt", Language: "pt", Extension: ".wav", MaxDurationSeconds: 1, DecodeBytes: 90000, WindowBytes: 4096, ResultBytes: 128 << 10}}
}
func TestConfigStrictAndSafeListener(t *testing.T) {
	good := baseConfig(t)
	b, _ := json.Marshal(good)
	if _, e := parseConfig(b); e != nil {
		t.Fatal(e)
	}
	for _, raw := range [][]byte{[]byte("null"), append(b, []byte(" {}")...), bytes.Replace(b, []byte(`"schema":1`), []byte(`"schema":1,"schema":1`), 1), bytes.Replace(b, []byte(`"schema":1`), []byte(`"Schema":1`), 1), bytes.Replace(b, []byte(`"threads":2`), []byte(`"threads":null`), 1), append([]byte(strings.Repeat(" ", 64<<10)), b...)} {
		if _, e := parseConfig(raw); e == nil {
			t.Fatal("ambiguous config")
		}
	}
	for _, kind := range []string{"schema", "store", "hash", "thread", "weight", "owned", "upload", "jobs", "listen", "public-http", "http-optin", "tls-pair", "hosts", "origin", "request", "header", "connections", "language", "extension", "result", "overlap"} {
		c := good
		switch kind {
		case "schema":
			c.Schema = 2
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
