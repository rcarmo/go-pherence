package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode"
	"unicode/utf16"
	"unicode/utf8"
)

const maxJSON = 4 << 20
const maxArtifact = 16 << 20

var artifactTypes = map[string]string{"transcript": "application/json", "vtt": "text/vtt", "speaker-transcript": "application/json", "speaker-vtt": "text/vtt"}

type client struct {
	base      string
	token     string
	http      *http.Client
	transport *http.Transport
}

func newClient(endpoint, token string, allowHTTP bool) (*client, error) {
	u, e := url.Parse(endpoint)
	if e != nil || u == nil || u.Host == "" || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || u.RawPath != "" || (u.Path != "" && u.Path != "/") || strings.ContainsAny(u.Host, "\\ \t\r\n") {
		return nil, fmt.Errorf("invalid endpoint: use an HTTPS origin with no credentials, query or path")
	}
	if u.Scheme != "https" {
		ip := net.ParseIP(u.Hostname())
		if u.Scheme != "http" || !allowHTTP || ip == nil || !ip.IsLoopback() {
			return nil, fmt.Errorf("plain HTTP requires --allow-loopback-http and a literal loopback IP")
		}
	}
	if len(token) < 32 || len(token) > 256 {
		return nil, fmt.Errorf("SPEECHJOB_TOKEN must contain 32..256 printable non-space ASCII bytes")
	}
	for _, b := range []byte(token) {
		if b < 33 || b > 126 {
			return nil, fmt.Errorf("invalid SPEECHJOB_TOKEN encoding")
		}
	}
	tr := &http.Transport{Proxy: nil, DialContext: (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext, TLSHandshakeTimeout: 10 * time.Second, MaxResponseHeaderBytes: 64 << 10, MaxConnsPerHost: 2, MaxIdleConnsPerHost: 2, IdleConnTimeout: 30 * time.Second, DisableCompression: true}
	// No redirects, even same-host redirects: mutations are not replayed and the
	// bearer token cannot follow an endpoint-controlled redirect to another host.
	c := &http.Client{Transport: tr, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	return &client{base: u.Scheme + "://" + u.Host, token: token, http: c, transport: tr}, nil
}
func (c *client) close() { c.transport.CloseIdleConnections() }
func validID(id string) bool {
	b, e := hex.DecodeString(id)
	return e == nil && len(b) == 16 && strings.ToLower(id) == id
}
func validHash(hash string) bool {
	b, e := hex.DecodeString(hash)
	return e == nil && len(b) == 32 && strings.ToLower(hash) == hash
}
func slug(s string) bool {
	if len(s) < 1 || len(s) > 48 {
		return false
	}
	for _, b := range s {
		if !(b >= 'a' && b <= 'z' || b >= '0' && b <= '9' || b == '-') {
			return false
		}
	}
	return true
}
func (c *client) request(ctx context.Context, method, path string, body io.Reader, size int64) (*http.Response, error) {
	r, e := http.NewRequestWithContext(ctx, method, c.base+path, body)
	if e != nil {
		return nil, fmt.Errorf("construct request failed")
	}
	r.Header.Set("Authorization", "Bearer "+c.token)
	r.Header.Set("Accept-Encoding", "identity")
	if body != nil {
		r.ContentLength = size
		r.Header.Set("Content-Type", "application/octet-stream")
	}
	// Do not install GetBody or idempotency keys. net/http may retry idempotent
	// reads internally on a stale connection; upload/run/delete are never marked
	// replayable and no application-level retries are performed.
	r.GetBody = nil
	response, e := c.http.Do(r)
	if e != nil {
		if ctx.Err() != nil {
			return nil, fmt.Errorf("request cancelled or timed out; inspect job status before retrying: %w", ctx.Err())
		}
		return nil, fmt.Errorf("request failed; inspect job status before retrying")
	}
	return response, nil
}
func readBounded(ctx context.Context, r io.Reader, limit int) ([]byte, error) {
	b := make([]byte, 0, min(limit, 32<<10))
	buf := make([]byte, 32<<10)
	for {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		n, e := r.Read(buf)
		if n > 0 {
			if len(b)+n > limit {
				return nil, fmt.Errorf("response exceeds size limit")
			}
			b = append(b, buf[:n]...)
		}
		if e == io.EOF {
			break
		}
		if e != nil {
			return nil, fmt.Errorf("response stream failed")
		}
		if n == 0 {
			return nil, io.ErrNoProgress
		}
	}
	// EOF completes the response. Cancellation racing after EOF must not turn a
	// fully received acknowledgement into an apparent failed mutation.
	return b, nil
}

// APIError exposes only the HTTP status and a fixed known code. Unknown response
// bodies, proxy errors, Location values and private callback text are not echoed.
type APIError struct {
	Status int
	Code   string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("HTTP %d: %s; inspect status/inventory before retrying", e.Status, e.Code)
}

var knownErrors = map[string]bool{"admission_rejected": true, "queue_enabled": true, "queue_state_conflict": true, "host_rejected": true, "unauthorized": true, "origin_rejected": true, "cross_site_rejected": true, "unavailable": true, "cancelled": true, "not_found": true, "invalid_query": true, "unexpected_body": true, "unknown_profile": true, "invalid_name": true, "unsupported_content_type": true, "empty_upload": true, "upload_too_large": true, "busy": true, "profile_changed": true, "not_running": true, "range_unsupported": true, "artifact_too_large": true, "method_not_allowed": true, "persistence_uncertain": true, "limit_exceeded": true, "integrity_failure": true, "operation_failed": true}

func responseError(ctx context.Context, r *http.Response) error {
	code := "request_rejected"
	if b, e := readBounded(ctx, r.Body, 64<<10); e == nil {
		var result struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(b, &result) == nil && knownErrors[result.Error] {
			code = result.Error
		}
	}
	return &APIError{r.StatusCode, code}
}
func (c *client) jsonResponse(ctx context.Context, method, path string, body io.Reader, size int64, status int) ([]byte, error) {
	r, e := c.request(ctx, method, path, body, size)
	if e != nil {
		return nil, e
	}
	defer r.Body.Close()
	if r.StatusCode != status {
		return nil, responseError(ctx, r)
	}
	if r.Header.Get("Content-Encoding") != "" {
		return nil, fmt.Errorf("encoded response rejected")
	}
	if status == 204 {
		b, e := readBounded(ctx, r.Body, 0)
		if e != nil || len(b) != 0 {
			return nil, fmt.Errorf("unexpected delete response body")
		}
		return []byte("{\"deleted\":true}"), nil
	}
	kind, _, e := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if e != nil || kind != "application/json" {
		return nil, fmt.Errorf("expected JSON response")
	}
	b, e := readBounded(ctx, r.Body, maxJSON)
	if e != nil {
		return nil, fmt.Errorf("response incomplete; inspect status/inventory before retrying: %w", e)
	}
	if !utf8.Valid(b) || !json.Valid(b) {
		return nil, fmt.Errorf("invalid JSON response")
	}
	return b, nil
}
func (c *client) jsonRequest(ctx context.Context, method, path string, body io.Reader, size int64, status int, out io.Writer) error {
	b, e := c.jsonResponse(ctx, method, path, body, size, status)
	if e != nil {
		return e
	}
	// Re-encode JSON instead of printing response text directly to a terminal.
	// String control characters and HTML are escaped. Preserve numeric precision.
	d := json.NewDecoder(strings.NewReader(string(b)))
	d.UseNumber()
	var result any
	if e = d.Decode(&result); e != nil {
		return fmt.Errorf("invalid JSON response")
	}
	if e = writeSafeJSON(out, result); e != nil {
		return fmt.Errorf("response received but output failed; inspect status before retrying")
	}
	return nil
}

// encoding/json escapes C0 controls but leaves DEL, C1 and Unicode formatting
// controls literal. Uploaded names can contain those too: escape them before
// terminal output without altering the decoded JSON value.
func writeSafeJSON(out io.Writer, value any) error {
	b, e := json.MarshalIndent(value, "", "  ")
	if e != nil {
		return e
	}
	var text strings.Builder
	for _, r := range string(b) {
		// MarshalIndent's literal LF is JSON whitespace; input LF is already escaped.
		if r != '\n' && (unicode.IsControl(r) || unicode.Is(unicode.Cf, r)) {
			if r <= 0xffff {
				fmt.Fprintf(&text, "\\u%04x", r)
			} else {
				hi, lo := utf16.EncodeRune(r)
				fmt.Fprintf(&text, "\\u%04x\\u%04x", hi, lo)
			}
		} else {
			text.WriteRune(r)
		}
	}
	text.WriteByte('\n')
	n, e := io.WriteString(out, text.String())
	if e == nil && n != text.Len() {
		return io.ErrShortWrite
	}
	return e
}

func (c *client) upload(ctx context.Context, profile, path, name string, out io.Writer) error {
	if !slug(profile) || path == "" {
		return fmt.Errorf("upload requires --profile ID and --file FILE")
	}
	st, e := os.Lstat(path)
	if e != nil || !st.Mode().IsRegular() {
		return fmt.Errorf("upload requires an existing regular non-symlink file")
	}
	f, e := os.Open(path)
	if e != nil {
		return fmt.Errorf("cannot open upload")
	}
	defer f.Close()
	opened, e := f.Stat()
	if e != nil || !opened.Mode().IsRegular() || !os.SameFile(st, opened) || opened.Size() < 1 || opened.Size() > 512<<20 {
		return fmt.Errorf("upload file changed or exceeds 1..512MiB bounds")
	}
	// Caller owns a stable local file for the request. Size/hash changes during the
	// upload are not a snapshot facility; no unqualified resumable-upload claim.
	if name == "" {
		name = filepath.Base(path)
	}
	if len(name) < 1 || len(name) > 256 || !utf8.ValidString(name) || strings.ContainsAny(name, "\r\n\x00") {
		return fmt.Errorf("invalid upload name")
	}
	q := url.Values{"profile": {profile}, "name": {name}}
	return c.jsonRequest(ctx, "POST", "/v1/jobs?"+q.Encode(), io.LimitReader(f, opened.Size()), opened.Size(), 201, out)
}

type artifact struct {
	Name   string `json:"name"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

func (c *client) download(ctx context.Context, id, name, destination string, out io.Writer) error {
	expectedType, ok := artifactTypes[name]
	if !validID(id) || !ok || destination == "" {
		return fmt.Errorf("download requires a valid --job, allowlisted --artifact and --out")
	}
	absolute, e := filepath.Abs(destination)
	if e != nil {
		return fmt.Errorf("invalid output path")
	}
	if _, e = os.Lstat(absolute); e == nil {
		return fmt.Errorf("output already exists")
	} else if !errors.Is(e, os.ErrNotExist) {
		return fmt.Errorf("cannot inspect output path")
	}
	// Bind download to a preceding authenticated status snapshot, not merely to
	// self-reported headers on an unrelated response. No content-address integrity
	// can protect against a malicious authenticated server; trust TLS and endpoint.
	data, e := c.jsonResponse(ctx, "GET", "/v1/jobs/"+id, nil, 0, 200)
	if e != nil {
		return e
	}
	var status struct {
		ID        string     `json:"id"`
		Artifacts []artifact `json:"artifacts"`
	}
	if e = json.Unmarshal(data, &status); e != nil || status.ID != id || len(status.Artifacts) > 4 {
		return fmt.Errorf("invalid job artifact metadata")
	}
	var a artifact
	count := 0
	for _, candidate := range status.Artifacts {
		if candidate.Name == name {
			a = candidate
			count++
		}
	}
	if count != 1 || a.Bytes < 0 || a.Bytes > maxArtifact || !validHash(a.SHA256) {
		return fmt.Errorf("missing or invalid artifact metadata")
	}
	r, e := c.request(ctx, "GET", "/v1/jobs/"+id+"/artifacts/"+name, nil, 0)
	if e != nil {
		return e
	}
	defer r.Body.Close()
	if r.StatusCode != 200 {
		return responseError(ctx, r)
	}
	kind, _, e := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if e != nil || kind != expectedType || r.Header.Get("Content-Encoding") != "" || r.ContentLength != a.Bytes || len(r.Header.Values("ETag")) != 1 || r.Header.Get("ETag") != `"sha256-`+a.SHA256+`"` {
		return fmt.Errorf("artifact response metadata mismatch")
	}
	// No server filename/path is used. A 0600 same-directory temporary file is
	// published with a no-clobber hard link only after complete size/hash checks.
	dir := filepath.Dir(absolute)
	f, e := os.CreateTemp(dir, ".speechjob-download-*")
	if e != nil {
		return fmt.Errorf("cannot create private download temporary file")
	}
	temp := f.Name()
	defer func() { f.Close(); os.Remove(temp) }()
	digest := sha256.New()
	buf := make([]byte, 32<<10)
	remaining := a.Bytes
	for remaining > 0 {
		if e = ctx.Err(); e != nil {
			return e
		}
		n, re := io.ReadFull(r.Body, buf[:min(int64(len(buf)), remaining)])
		if re != nil {
			return fmt.Errorf("incomplete artifact response")
		}
		if e = ctx.Err(); e != nil {
			return e
		}
		nw, we := f.Write(buf[:n])
		if we != nil || nw != n {
			return fmt.Errorf("download write failed")
		}
		digest.Write(buf[:n])
		remaining -= int64(n)
	}
	var extra [1]byte
	n, re := r.Body.Read(extra[:])
	if n != 0 || re != io.EOF {
		return fmt.Errorf("artifact response has trailing or invalid data")
	}
	if hex.EncodeToString(digest.Sum(nil)) != a.SHA256 {
		return fmt.Errorf("artifact SHA256 mismatch")
	}
	if e = ctx.Err(); e != nil {
		return e
	}
	if e = f.Sync(); e != nil {
		return fmt.Errorf("download sync failed")
	}
	if e = f.Close(); e != nil {
		return fmt.Errorf("download close failed")
	}
	if e = ctx.Err(); e != nil {
		return e
	}
	if e = os.Link(temp, absolute); e != nil {
		return fmt.Errorf("cannot publish output without replacing an existing file")
	}
	// After linking, cancellation cannot retract a published artifact. Report sync
	// uncertainty, preserving the output; never overwrite or blindly retry.
	parent, e := os.Open(dir)
	if e != nil {
		return fmt.Errorf("output published but directory durability is uncertain")
	}
	e = errors.Join(parent.Sync(), parent.Close())
	if e != nil {
		return fmt.Errorf("output published but directory durability is uncertain")
	}
	if e = writeSafeJSON(out, struct {
		Saved  string `json:"saved"`
		Bytes  int64  `json:"bytes"`
		SHA256 string `json:"sha256"`
	}{absolute, a.Bytes, a.SHA256}); e != nil {
		return fmt.Errorf("output file published and synced but status output failed; inspect destination before retrying")
	}
	return nil
}
