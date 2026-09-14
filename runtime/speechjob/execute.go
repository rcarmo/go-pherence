package speechjob

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	stdhash "hash"
	"io"
	"os"
	"strings"
	"time"
	"unicode/utf8"
)

type boundedWriter struct {
	ctx          context.Context
	file         *os.File
	digest       stdhash.Hash
	count, limit int64
	failed       error
}

func (w *boundedWriter) Write(b []byte) (int, error) {
	if w.failed != nil {
		return 0, w.failed
	}
	if e := w.ctx.Err(); e != nil {
		w.failed = e
		return 0, e
	}
	if int64(len(b)) > w.limit-w.count {
		w.failed = ErrLimit
		return 0, w.failed
	}
	n, e := w.file.Write(b)
	w.digest.Write(b[:n])
	w.count += int64(n)
	if e == nil && n != len(b) {
		e = io.ErrShortWrite
	}
	w.failed = e
	return n, e
}
func (s *Store) writeBlob(ctx context.Context, id, name string, limit int64, fn func(io.Writer) error) (Blob, error) {
	var zero Blob
	tok, e := token()
	if e != nil {
		return zero, e
	}
	temp := id + "/.payload-" + tok
	f, e := s.root.OpenFile(temp, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return zero, e
	}
	defer func() { f.Close(); s.root.Remove(temp) }()
	w := &boundedWriter{ctx: ctx, file: f, digest: sha256.New(), limit: limit}
	e = callStage(func() error { return fn(w) })
	if e == nil {
		e = w.failed
	}
	if e == nil {
		e = ctx.Err()
	}
	if e != nil {
		return zero, e
	}
	if e = f.Sync(); e != nil {
		return zero, e
	}
	if e = f.Close(); e != nil {
		return zero, e
	}
	if e = s.hit("payload-synced"); e != nil {
		return zero, e
	}
	blob := Blob{File: name, SHA256: hex.EncodeToString(w.digest.Sum(nil)), Bytes: w.count}
	if e = s.root.Link(temp, id+"/"+name); e != nil {
		if !errors.Is(e, os.ErrExist) {
			return zero, e
		}
		// A crash after payload publication may leave an unreferenced stage. Reuse
		// only identical bytes; changed data under the same identity is corruption.
		existing, ve := s.openBlob(ctx, id, blob)
		if ve != nil {
			return zero, ve
		}
		existing.Close()
	}
	if e = s.hit("payload-published"); e != nil {
		return zero, e
	}
	if e = s.root.Remove(temp); e != nil {
		return zero, e
	}
	if e = s.syncDir(id); e != nil {
		return zero, e
	}
	return blob, nil
}
func callStage(fn func() error) (err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("speech stage panic: %v", p)
		}
	}()
	return fn()
}
func checkpointKey(m Manifest, stage Stage) string {
	// All preceding stage identities+payloads are dependencies, in order.
	parents := append([]Checkpoint{}, m.Checkpoints...)
	data, _ := json.Marshal(struct {
		Schema                        int
		Input, Config, Stage, Version string
		Parents                       []Checkpoint
	}{1, m.Input.SHA256, m.ConfigurationSHA256, stage.Name, stage.Version, parents})
	return hash(data)
}

// Run serially resumes verified checkpoints, or appends newly committed ones.
// Existing checkpoints MUST be an exact prefix of the supplied stage names and
// versions. Different config/order/version is rejected, never silently reused.
// Retry is explicit: queued/failed/cancelled jobs can run; a complete job is a
// verified no-op only for the same full stage list. One store-wide admission slot
// prevents oversubscription; this API does not start goroutines or services.
// A nonnil error may accompany an in-memory proposed Manifest whose latest
// transition was NOT durably published. ErrPersistence identifies that case;
// inspect Get/reopen to determine authoritative stored state before acting.
func (s *Store) Run(ctx context.Context, id string, configuration []byte, stages []Stage, progress Progress) (result Manifest, err error) {
	if e := ctx.Err(); e != nil {
		return Manifest{}, e
	}
	if !validConfiguration(configuration) || len(stages) < 1 || len(stages) > 64 {
		return Manifest{}, ErrConfiguration
	}
	seen := map[string]bool{}
	for _, stage := range stages {
		if !validStage(stage.Name) || !validHash(stage.Version) || stage.Run == nil || seen[stage.Name] {
			return Manifest{}, ErrConfiguration
		}
		seen[stage.Name] = true
	}
	s.mu.Lock()
	if e := s.ready(); e != nil {
		s.mu.Unlock()
		return Manifest{}, e
	}
	if s.busy {
		s.mu.Unlock()
		return Manifest{}, ErrBusy
	}
	m, e := s.load(id)
	if e != nil {
		s.mu.Unlock()
		return Manifest{}, e
	}
	if hash(configuration) != m.ConfigurationSHA256 || len(m.Checkpoints) > len(stages) {
		s.mu.Unlock()
		return Manifest{}, ErrConfiguration
	}
	for i, cp := range m.Checkpoints {
		if cp.Stage != stages[i].Name || cp.Version != stages[i].Version {
			s.mu.Unlock()
			return Manifest{}, ErrConfiguration
		}
	}
	if m.Status == Complete && len(m.Checkpoints) != len(stages) {
		s.mu.Unlock()
		return Manifest{}, ErrConfiguration
	}
	runCtx, cancel := context.WithCancel(ctx)
	s.busy = true
	s.runningID = id
	s.cancel = cancel
	s.mu.Unlock()
	defer func() {
		cancel()
		s.mu.Lock()
		s.busy = false
		s.runningID = ""
		s.cancel = nil
		s.mu.Unlock()
	}()
	// No state mutation until immutable dependencies have been verified.
	source, e := s.openBlob(runCtx, id, m.Input)
	if e != nil {
		return clone(m), e
	}
	source.Close()
	prefix := m
	prefix.Checkpoints = nil
	for i, cp := range m.Checkpoints {
		if cp.Key != checkpointKey(prefix, stages[i]) {
			return clone(m), ErrCorrupt
		}
		reader, e := s.openBlob(runCtx, id, cp.Blob)
		if e != nil {
			return clone(m), e
		}
		reader.Close()
		prefix.Checkpoints = append(prefix.Checkpoints, cp)
	}
	if m.Status == Complete {
		return clone(m), nil
	}
	publish := func() error {
		s.mu.Lock()
		defer s.mu.Unlock()
		if e := s.save(m); e != nil {
			return errors.Join(ErrPersistence, e)
		}
		return nil
	}
	notify := func() error {
		if progress == nil {
			return runCtx.Err()
		}
		_ = callStage(func() error { progress(clone(m)); return nil })
		return runCtx.Err()
	}
	m.Status = Running
	m.Attempts++
	m.Error = ""
	m.Updated = time.Now().UTC()
	if e = publish(); e != nil {
		return clone(m), e
	}
	terminal := func(cause error) (Manifest, error) {
		m.Status = Failed
		if errors.Is(cause, context.Canceled) || errors.Is(cause, context.DeadlineExceeded) {
			m.Status = Cancelled
		}
		m.ActiveStage = ""
		m.Error = strings.ToValidUTF8(cause.Error(), "�")
		if len(m.Error) > 1024 {
			m.Error = m.Error[:1024]
			for !utf8.ValidString(m.Error) {
				m.Error = m.Error[:len(m.Error)-1]
			}
		}
		m.Updated = time.Now().UTC()
		pe := publish()
		return clone(m), errors.Join(cause, pe)
	}
	if e = notify(); e != nil {
		return terminal(e)
	}
	for i := len(m.Checkpoints); i < len(stages); i++ {
		if e = runCtx.Err(); e != nil {
			return terminal(e)
		}
		stage := stages[i]
		m.ActiveStage = stage.Name
		m.Updated = time.Now().UTC()
		if e = publish(); e != nil {
			return terminal(e)
		}
		used, _, e := s.usage()
		if e != nil {
			return terminal(e)
		}
		limit := min(s.limits.MaxArtifactBytes, s.limits.MaxBytes-used-maxManifest)
		if limit < 0 {
			return terminal(ErrLimit)
		}
		key := checkpointKey(m, stage)
		blob, e := s.writeBlob(runCtx, id, "stage-"+key, limit, func(out io.Writer) error { return stage.Run(runCtx, &Input{store: s, job: clone(m)}, out) })
		if e != nil {
			return terminal(e)
		}
		m.Checkpoints = append(m.Checkpoints, Checkpoint{Stage: stage.Name, Version: stage.Version, Key: key, Blob: blob})
		m.ActiveStage = ""
		m.Updated = time.Now().UTC()
		if e = publish(); e != nil {
			return terminal(e)
		}
		if e = notify(); e != nil {
			return terminal(e)
		}
	}
	if e = runCtx.Err(); e != nil {
		return terminal(e)
	}
	m.Status = Complete
	m.ActiveStage = ""
	m.Updated = time.Now().UTC()
	if e = publish(); e != nil {
		return terminal(e)
	}
	// Final durable success is not undone by a reporting callback panic/cancel.
	if progress != nil {
		_ = callStage(func() error { progress(clone(m)); return nil })
	}
	return clone(m), nil
}

// Cancel signals only the currently admitted job. It does not report successful
// cancellation until Run's callback has returned and terminal state is durable.
func (s *Store) Cancel(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || !s.busy || s.runningID != id {
		return false
	}
	s.cancel()
	return true
}
