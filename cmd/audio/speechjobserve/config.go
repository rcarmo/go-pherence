package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

type Asset struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}
type Limits struct {
	Jobs             int   `json:"jobs"`
	UploadBytes      int64 `json:"upload_bytes"`
	ArtifactBytes    int64 `json:"artifact_bytes"`
	StoreBytes       int64 `json:"store_bytes"`
	WeightBytes      int64 `json:"weight_bytes"`
	OwnedWeightBytes int64 `json:"owned_weight_bytes"`
}
type HTTPSettings struct {
	EnableUI          bool     `json:"enable_ui"`
	Listen            string   `json:"listen"`
	Hosts             []string `json:"hosts"`
	Origin            string   `json:"origin"`
	TLSCert           string   `json:"tls_cert"`
	TLSKey            string   `json:"tls_key"`
	AllowLoopbackHTTP bool     `json:"allow_loopback_http"`
	RequestSeconds    int      `json:"request_seconds"`
	HeaderSeconds     int      `json:"header_seconds"`
	IdleSeconds       int      `json:"idle_seconds"`
	ShutdownSeconds   int      `json:"shutdown_seconds"`
	MaxRequests       int      `json:"max_requests"`
	MaxConnections    int      `json:"max_connections"`
}
type ProfileSettings struct {
	ID                       string `json:"id"`
	Language                 string `json:"language"`
	Extension                string `json:"extension"`
	MaxDurationSeconds       int    `json:"max_duration_seconds"`
	DecodeBytes              int64  `json:"decode_bytes"`
	OverlapSamples           int64  `json:"overlap_samples"`
	MaxNewTokens             int    `json:"max_new_tokens"`
	MaxInitialTimestampIndex int    `json:"max_initial_timestamp_index"`
	SkipDigitalSilence       bool   `json:"skip_digital_silence"`
	WindowBytes              int64  `json:"window_bytes"`
	ResultBytes              int64  `json:"result_bytes"`
}

// QueueSettings opt in separately to durable intent and worker execution.
// An unstarted queue accepts intents but never runs them. UI currently requires
// synchronous mode and is rejected with queue mode instead of mislabelling runs.
type QueueSettings struct {
	Enable      bool   `json:"enable"`
	StartWorker bool   `json:"start_worker"`
	Directory   string `json:"directory"`
	MaxEntries  int    `json:"max_entries"`
	MaxBytes    int64  `json:"max_bytes"`
	JobSeconds  int    `json:"job_seconds"`
}

// ResourceSettings are explicit operator estimates, not measured RSS limits.
// ResidentBytes stays reserved through server drain. LoadBytes covers startup
// peak; WorkBytes is additional transient work while the model remains resident.
type ResourceSettings struct {
	CPUSlots      int   `json:"cpu_slots"`
	MemoryBytes   int64 `json:"memory_bytes"`
	MaxWaiting    int   `json:"max_waiting"`
	LoadBytes     int64 `json:"load_bytes"`
	ResidentBytes int64 `json:"resident_bytes"`
	WorkBytes     int64 `json:"work_bytes"`
}
type ServerConfig struct {
	Resources      *ResourceSettings `json:"resources,omitempty"`
	Queue          QueueSettings     `json:"queue"`
	Schema         int               `json:"schema"`
	AllowExecution bool              `json:"allow_execution"`
	Store          string            `json:"store"`
	RuntimeSHA256  string            `json:"runtime_sha256"`
	Threads        int               `json:"threads"`
	Limits         Limits            `json:"limits"`
	HTTP           HTTPSettings      `json:"http"`
	Weights        Asset             `json:"weights"`
	ModelConfig    Asset             `json:"model_config"`
	Tokenizer      Asset             `json:"tokenizer"`
	Generation     Asset             `json:"generation"`
	FFmpeg         Asset             `json:"ffmpeg"`
	FFprobe        Asset             `json:"ffprobe"`
	Profile        ProfileSettings   `json:"profile"`
}

func validHash(s string) bool {
	b, e := hex.DecodeString(s)
	return e == nil && len(b) == 32 && s == strings.ToLower(s)
}
func slug(s string) bool {
	if len(s) < 1 || len(s) > 48 {
		return false
	}
	for _, r := range s {
		if !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-') {
			return false
		}
	}
	return true
}
func parseConfig(data []byte) (ServerConfig, error) {
	var cfg ServerConfig
	if len(data) > 64<<10 || !utf8.Valid(data) {
		return cfg, fmt.Errorf("server configuration size/encoding rejected")
	}
	if e := uniqueJSON(data, 12, true, false); e != nil {
		return cfg, e
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if e := d.Decode(&cfg); e != nil {
		return cfg, fmt.Errorf("invalid server configuration fields")
	}
	if e := cfg.validate(); e != nil {
		return cfg, e
	}
	return cfg, nil
}

// Token-walk bounds depth and rejects duplicates/escaped aliases/case aliases
// and nulls. Struct decoding then checks types and unknown fields. Configuration
// remains administrator-authored; no neural graph defaults are inferred here.
func uniqueJSON(data []byte, maxDepth int, foldKeys, allowNull bool) error {
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	var value func(int) error
	value = func(depth int) error {
		if depth > maxDepth {
			return fmt.Errorf("JSON nesting exceeds bound")
		}
		t, e := d.Token()
		if e != nil {
			return fmt.Errorf("invalid JSON")
		}
		if t == nil && !allowNull {
			return fmt.Errorf("null JSON value")
		}
		delim, ok := t.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := map[string]bool{}
			for d.More() {
				key, e := d.Token()
				if e != nil {
					return fmt.Errorf("invalid JSON key")
				}
				s, ok := key.(string)
				keyID := s
				if foldKeys {
					keyID = strings.ToLower(s)
					if keyID != s {
						return fmt.Errorf("noncanonical configuration key")
					}
				}
				if !ok || seen[keyID] {
					return fmt.Errorf("duplicate JSON key")
				}
				seen[keyID] = true
				if e = value(depth + 1); e != nil {
					return e
				}
			}
			end, e := d.Token()
			if e != nil || end != json.Delim('}') {
				return fmt.Errorf("invalid JSON object")
			}
		case '[':
			for d.More() {
				if e = value(depth + 1); e != nil {
					return e
				}
			}
			end, e := d.Token()
			if e != nil || end != json.Delim(']') {
				return fmt.Errorf("invalid JSON array")
			}
		default:
			return fmt.Errorf("invalid JSON delimiter")
		}
		return nil
	}
	if e := value(0); e != nil {
		return e
	}
	if _, e := d.Token(); e != io.EOF {
		return fmt.Errorf("trailing JSON")
	}
	return nil
}
func (c ServerConfig) validate() error {
	if c.Schema != 1 || !filepath.IsAbs(c.Store) || !validHash(c.RuntimeSHA256) || c.Threads < 1 || c.Threads > 16 {
		return fmt.Errorf("invalid server identity/store/threads")
	}
	l := c.Limits
	if l.Jobs < 1 || l.Jobs > 1000 || l.UploadBytes < 1 || l.UploadBytes > 512<<20 || l.ArtifactBytes < 1 || l.ArtifactBytes > 1<<30 || l.StoreBytes < 128<<10 || l.StoreBytes > 64<<30 || l.UploadBytes > l.StoreBytes || l.ArtifactBytes > l.StoreBytes || l.WeightBytes < 8 || l.WeightBytes > 8<<30 || l.OwnedWeightBytes < 1 || l.OwnedWeightBytes > 16<<30 {
		return fmt.Errorf("invalid configured resource caps")
	}
	if r := c.Resources; r != nil {
		if r.CPUSlots < c.Threads || r.CPUSlots > 65536 || r.MemoryBytes < 1 || r.MaxWaiting < 0 || r.MaxWaiting > 128 || r.LoadBytes < r.ResidentBytes || r.LoadBytes > r.MemoryBytes || r.ResidentBytes < l.OwnedWeightBytes || r.WorkBytes < 1 || r.ResidentBytes > r.MemoryBytes || r.WorkBytes > r.MemoryBytes-r.ResidentBytes {
			return fmt.Errorf("invalid declared CPU/memory resource budget")
		}
	}
	q := c.Queue
	if q.Enable {
		if !filepath.IsAbs(q.Directory) || q.Directory == c.Store || q.MaxEntries < 1 || q.MaxEntries > 128 || q.MaxBytes < 256<<10 || q.MaxBytes > 4<<20 || q.JobSeconds < 1 || q.JobSeconds > 8*3600 || c.HTTP.EnableUI {
			return fmt.Errorf("invalid queue configuration or unsupported queued browser UI")
		}
	} else if q.StartWorker || q.Directory != "" || q.MaxEntries != 0 || q.MaxBytes != 0 || q.JobSeconds != 0 {
		return fmt.Errorf("queue options require enable")
	}
	h := c.HTTP
	host, port, e := net.SplitHostPort(h.Listen)
	p, pe := strconv.Atoi(port)
	ip := net.ParseIP(host)
	if e != nil || pe != nil || p < 1 || p > 65535 || ip == nil || h.RequestSeconds < 1 || h.RequestSeconds > 8*3600 || h.HeaderSeconds < 1 || h.HeaderSeconds > 30 || h.HeaderSeconds > h.RequestSeconds || h.IdleSeconds < 1 || h.IdleSeconds > 300 || h.ShutdownSeconds < 1 || h.ShutdownSeconds > 300 || h.MaxRequests < 1 || h.MaxRequests > 32 || h.MaxConnections < 2 || h.MaxConnections > 128 || h.MaxConnections < h.MaxRequests+1 {
		return fmt.Errorf("invalid HTTP listener/deadlines/caps")
	}
	hasTLS := h.TLSCert != "" && h.TLSKey != ""
	if (h.TLSCert == "") != (h.TLSKey == "") || hasTLS && (!filepath.IsAbs(h.TLSCert) || !filepath.IsAbs(h.TLSKey)) || !hasTLS && (!h.AllowLoopbackHTTP || !ip.IsLoopback()) {
		return fmt.Errorf("TLS required except explicit literal loopback HTTP")
	}
	if len(h.Hosts) < 1 || len(h.Hosts) > 16 {
		return fmt.Errorf("HTTP Host allowlist required")
	}
	seen := map[string]bool{}
	for _, host := range h.Hosts {
		u, e := url.Parse("http://" + host)
		if e != nil || host == "" || strings.ContainsAny(host, " /\\?#@\t\r\n") || u.Host != host || u.Hostname() == "" || seen[host] {
			return fmt.Errorf("invalid Host allowlist")
		}
		seen[host] = true
	}
	if h.Origin != "" {
		u, e := url.Parse(h.Origin)
		if e != nil || u == nil || u.User != nil || u.Path != "" || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") {
			return fmt.Errorf("invalid origin")
		}
	}
	if h.EnableUI {
		u, e := url.Parse(h.Origin)
		if e != nil || u == nil || h.MaxRequests < 2 || !seen[u.Host] || (u.Scheme == "https") != hasTLS {
			return fmt.Errorf("browser UI requires matching TLS/origin/Host and two request slots")
		}
	}
	for _, a := range []Asset{c.Weights, c.ModelConfig, c.Tokenizer, c.Generation, c.FFmpeg, c.FFprobe} {
		if !filepath.IsAbs(a.Path) || !validHash(a.SHA256) {
			return fmt.Errorf("all assets require absolute paths and SHA256")
		}
	}
	f := c.Profile
	if !slug(f.ID) || len(f.Language) < 2 || len(f.Language) > 3 || f.MaxDurationSeconds < 1 || f.MaxDurationSeconds > 14400 || f.DecodeBytes < 46 || f.DecodeBytes > l.ArtifactBytes || f.OverlapSamples < 0 || f.OverlapSamples > 240000 || f.MaxNewTokens < 0 || f.MaxNewTokens > 445 || f.MaxInitialTimestampIndex < 0 || f.MaxInitialTimestampIndex > 1500 || f.WindowBytes < 1 || f.WindowBytes > 1<<20 || f.ResultBytes < f.WindowBytes || f.ResultBytes > 64<<20 || f.ResultBytes > l.ArtifactBytes {
		return fmt.Errorf("invalid ASR profile limits")
	}
	for _, r := range f.Language {
		if r < 'a' || r > 'z' {
			return fmt.Errorf("invalid language code")
		}
	}
	switch f.Extension {
	case ".wav", ".m4a", ".mp4", ".mov":
	default:
		return fmt.Errorf("invalid input extension")
	}
	return nil
}
func boundedFile(ctx context.Context, path string, cap int64) ([]byte, error) {
	f, e := regularFile(path, cap)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	var b bytes.Buffer
	buf := make([]byte, 32<<10)
	for {
		if e = ctx.Err(); e != nil {
			return nil, e
		}
		n, re := f.Read(buf)
		if n > 0 {
			if int64(b.Len()+n) > cap {
				return nil, fmt.Errorf("file exceeds cap")
			}
			b.Write(buf[:n])
		}
		if re == io.EOF {
			break
		}
		if re != nil {
			return nil, fmt.Errorf("file read failed")
		}
	}
	return b.Bytes(), ctx.Err()
}
func regularFile(path string, cap int64) (*os.File, error) {
	st, e := os.Lstat(path)
	if e != nil || !st.Mode().IsRegular() || st.Size() < 1 || st.Size() > cap {
		return nil, fmt.Errorf("asset is not a bounded regular file")
	}
	f, e := os.Open(path)
	if e != nil {
		return nil, fmt.Errorf("asset open failed")
	}
	got, e := f.Stat()
	if e != nil || !os.SameFile(st, got) || !got.Mode().IsRegular() {
		f.Close()
		return nil, fmt.Errorf("asset changed during open")
	}
	return f, nil
}
func verifyAsset(ctx context.Context, a Asset, cap int64, executable bool) error {
	f, e := regularFile(a.Path, cap)
	if e != nil {
		return e
	}
	defer f.Close()
	st, e := f.Stat()
	if e != nil {
		return e
	}
	if executable && st.Mode().Perm()&0111 == 0 {
		return fmt.Errorf("media executable lacks execute permission")
	}
	h := sha256.New()
	buf := make([]byte, 32<<10)
	var bytes int64
	for {
		if e = ctx.Err(); e != nil {
			return e
		}
		n, re := f.Read(buf)
		bytes += int64(n)
		if bytes > cap {
			return fmt.Errorf("asset exceeds cap")
		}
		if n > 0 {
			h.Write(buf[:n])
		}
		if re == io.EOF {
			break
		}
		if re != nil {
			return fmt.Errorf("asset hash read failed")
		}
	}
	if hex.EncodeToString(h.Sum(nil)) != a.SHA256 {
		return fmt.Errorf("asset SHA256 mismatch")
	}
	return ctx.Err()
}
func duration(seconds int) time.Duration { return time.Duration(seconds) * time.Second }
