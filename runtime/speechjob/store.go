package speechjob

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const maxManifest = 128 << 10
const maxConfig = 16 << 10

type Store struct {
	mu           sync.Mutex
	root         *os.Root
	lock         *os.File
	limits       Limits
	closed, busy bool
	runningID    string
	cancel       context.CancelFunc
	// Fault-injection hook, never exposed in the production API.
	fault func(string) error
}

// Open creates or opens a private store under an existing caller-owned parent
// directory on a local filesystem with
// atomic rename/hard-link, file fsync and directory fsync support. It holds an
// advisory process lock; symlink entries are rejected. Competing processes using
// this API are excluded. It is not safe against a malicious same-UID directory
// writer, distributed storage, or hardware that ignores flushes. Opening marks
// interrupted running jobs failed; it never resumes or removes payloads.
func Open(directory string, limits Limits) (*Store, error) {
	// Freeze the root's display path for path-based media adapters. The rooted
	// descriptor still owns filesystem access; callers must not rename the root.
	absolute, err := filepath.Abs(directory)
	if err != nil {
		return nil, err
	}
	directory = absolute
	if limits.MaxJobs < 1 || limits.MaxJobs > 10000 || limits.MaxUploadBytes < 1 || limits.MaxArtifactBytes < 1 || limits.MaxBytes < maxManifest || limits.MaxUploadBytes > limits.MaxBytes || limits.MaxArtifactBytes > limits.MaxBytes {
		return nil, fmt.Errorf("invalid speech job limits")
	}
	// Provision only the final component: the caller owns an existing parent.
	// Sync that parent too so a newly acknowledged store cannot lose its root
	// directory entry despite its children having been synced.
	if err := os.Mkdir(directory, 0700); err != nil && !errors.Is(err, os.ErrExist) {
		return nil, err
	}
	st, err := os.Lstat(directory)
	if err != nil {
		return nil, err
	}
	if !st.IsDir() || st.Mode().Perm()&0077 != 0 {
		return nil, fmt.Errorf("speech store requires private directory")
	}
	parent, err := os.Open(filepath.Dir(filepath.Clean(directory)))
	if err != nil {
		return nil, err
	}
	err = errors.Join(parent.Sync(), parent.Close())
	if err != nil {
		return nil, err
	}
	root, err := os.OpenRoot(directory)
	if err != nil {
		return nil, err
	}
	fail := func(e error) (*Store, error) { root.Close(); return nil, e }
	if st, e := root.Lstat(".lock"); e == nil {
		if !st.Mode().IsRegular() {
			return fail(ErrCorrupt)
		}
	} else if !errors.Is(e, os.ErrNotExist) {
		return fail(e)
	}
	lock, err := root.OpenFile(".lock", os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return fail(err)
	}
	if err = lockStore(lock); err != nil {
		lock.Close()
		return fail(err)
	}
	s := &Store{root: root, lock: lock, limits: limits}
	if err = s.recover(); err != nil {
		s.Close()
		return nil, err
	}
	return s, nil
}
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	if s.busy {
		return ErrBusy
	}
	s.closed = true
	return errors.Join(s.lock.Close(), s.root.Close())
}
func (s *Store) ready() error {
	if s.closed {
		return ErrClosed
	}
	return nil
}
func hash(data []byte) string { h := sha256.Sum256(data); return hex.EncodeToString(h[:]) }
func validHash(v string) bool {
	if len(v) != 64 {
		return false
	}
	_, e := hex.DecodeString(v)
	return e == nil && v == strings.ToLower(v)
}
func validID(v string) bool {
	if len(v) != 32 {
		return false
	}
	_, e := hex.DecodeString(v)
	return e == nil && v == strings.ToLower(v)
}
func validStage(v string) bool {
	if len(v) < 1 || len(v) > 48 {
		return false
	}
	for _, c := range v {
		if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '-') {
			return false
		}
	}
	return true
}
func token() (string, error) {
	var b [16]byte
	if _, e := rand.Read(b[:]); e != nil {
		return "", e
	}
	return hex.EncodeToString(b[:]), nil
}
func clone(m Manifest) Manifest {
	m.Checkpoints = append([]Checkpoint(nil), m.Checkpoints...)
	return m
}
func (s *Store) hit(name string) error {
	if s.fault != nil {
		return s.fault(name)
	}
	return nil
}
func (s *Store) syncDir(dir string) error {
	f, e := s.root.Open(dir)
	if e != nil {
		return e
	}
	return errors.Join(f.Sync(), f.Close())
}
func (s *Store) usage() (int64, int, error) {
	var size int64
	jobs := 0
	e := fs.WalkDir(s.root.FS(), ".", func(p string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if d.Type()&os.ModeSymlink != 0 {
			return ErrCorrupt
		}
		if d.IsDir() {
			if len(p) == 32 && validID(p) {
				jobs++
			}
			return nil
		}
		i, e := d.Info()
		if e != nil {
			return e
		}
		if !i.Mode().IsRegular() || i.Size() < 0 {
			return ErrCorrupt
		}
		if i.Size() > s.limits.MaxBytes-size {
			return ErrLimit
		}
		size += i.Size()
		return nil
	})
	return size, jobs, e
}
func (s *Store) recover() error {
	_, jobs, e := s.usage()
	if e != nil {
		return e
	}
	if jobs > s.limits.MaxJobs {
		return ErrLimit
	}
	entries, e := fs.ReadDir(s.root.FS(), ".")
	if e != nil {
		return e
	}
	for _, entry := range entries {
		if !entry.IsDir() || !validID(entry.Name()) {
			continue
		}
		m, e := s.load(entry.Name())
		if errors.Is(e, os.ErrNotExist) {
			continue
		}
		if e != nil {
			return e
		}
		if m.Status == Running {
			m.Status = Failed
			m.Error = "interrupted before terminal checkpoint"
			m.ActiveStage = ""
			m.Updated = time.Now().UTC()
			if e = s.save(m); e != nil {
				return e
			}
		}
	}
	return nil
}
func (s *Store) load(id string) (Manifest, error) {
	var m Manifest
	if !validID(id) {
		return m, fmt.Errorf("invalid job id")
	}
	f, e := s.root.Open(id + "/manifest.json")
	if e != nil {
		return m, e
	}
	b, e := io.ReadAll(io.LimitReader(f, maxManifest+1))
	ce := f.Close()
	if e != nil || ce != nil {
		return m, errors.Join(e, ce)
	}
	if len(b) > maxManifest {
		return m, ErrCorrupt
	}
	decoder := json.NewDecoder(bytes.NewReader(b))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&m) != nil {
		return m, ErrCorrupt
	}
	if decoder.Decode(new(any)) != io.EOF {
		return m, ErrCorrupt
	}
	if m.Schema != 1 || m.ID != id || m.Attempts < 0 || m.Created.IsZero() || m.Updated.Before(m.Created) || !validTitle(m.Title) || !validConfiguration([]byte(m.Configuration)) || hash([]byte(m.Configuration)) != m.ConfigurationSHA256 || len(m.Checkpoints) > 64 || len(m.Error) > 1024 || m.Input.File != "input" || m.Input.Bytes < 1 || m.Input.Bytes > s.limits.MaxUploadBytes || !validHash(m.Input.SHA256) {
		return m, ErrCorrupt
	}
	switch m.Status {
	case Queued, Running, Failed, Cancelled, Complete:
	default:
		return m, ErrCorrupt
	}
	if m.ActiveStage != "" && !validStage(m.ActiveStage) {
		return m, ErrCorrupt
	}
	if m.Status != Running && m.ActiveStage != "" {
		return m, ErrCorrupt
	}
	seen := map[string]bool{}
	for _, cp := range m.Checkpoints {
		if !validStage(cp.Stage) || seen[cp.Stage] || !validHash(cp.Version) || !validHash(cp.Key) || !validHash(cp.Blob.SHA256) || cp.Blob.File != "stage-"+cp.Key || cp.Blob.Bytes < 0 || cp.Blob.Bytes > s.limits.MaxArtifactBytes {
			return m, ErrCorrupt
		}
		seen[cp.Stage] = true
	}
	if m.MediaReleased && m.Status != Complete && m.Status != Cancelled {
		return m, ErrCorrupt
	}
	return m, nil
}
func validTitle(title string) bool {
	return len(title) <= 160 && utf8.ValidString(title) && !strings.ContainsAny(title, "\r\n\x00") && (title == "" || strings.TrimSpace(title) == title)
}

func validConfiguration(b []byte) bool {
	return len(b) > 1 && len(b) <= maxConfig && utf8.Valid(b) && json.Valid(b) && strings.HasPrefix(strings.TrimSpace(string(b)), "{")
}

// save writes a complete manifest beside its old version, fsyncs it, replaces it
// atomically and syncs the job directory. Failure after rename is ambiguous:
// callers must inspect persisted state; old and new manifests are both valid.
func (s *Store) save(m Manifest) error {
	b, e := json.Marshal(m)
	if e != nil {
		return e
	}
	if len(b) > maxManifest {
		return ErrLimit
	}
	used, _, e := s.usage()
	if e != nil {
		return e
	}
	if int64(len(b)) > s.limits.MaxBytes-used {
		return ErrLimit
	}
	tok, e := token()
	if e != nil {
		return e
	}
	tmp := m.ID + "/.manifest-" + tok
	f, e := s.root.OpenFile(tmp, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return e
	}
	defer s.root.Remove(tmp)
	_, e = f.Write(b)
	if e == nil {
		e = f.Sync()
	}
	ce := f.Close()
	if e != nil || ce != nil {
		return errors.Join(e, ce)
	}
	if e = s.hit("manifest-synced"); e != nil {
		return e
	}
	if e = s.root.Rename(tmp, m.ID+"/manifest.json"); e != nil {
		return e
	}
	if e = s.hit("manifest-renamed"); e != nil {
		return e
	}
	return s.syncDir(m.ID)
}

// Create acknowledges only after the original upload and queued manifest have
// been synced and published. displayName is inert metadata, never a path. A
// generic blocking reader must support cancellation externally if needed.
// Publication errors do not acknowledge a job; Inventory exposes retained
// unacknowledged files so the caller can inspect or explicitly delete them.
func (s *Store) Create(ctx context.Context, displayName string, configuration []byte, src io.Reader) (Manifest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var zero Manifest
	if e := s.ready(); e != nil {
		return zero, e
	}
	if s.busy {
		return zero, ErrBusy
	}
	if e := ctx.Err(); e != nil {
		return zero, e
	}
	if src == nil || !validConfiguration(configuration) || len(displayName) > 256 || !utf8.ValidString(displayName) || strings.ContainsAny(displayName, "\r\n\x00") {
		return zero, fmt.Errorf("invalid upload metadata")
	}
	used, jobs, e := s.usage()
	if e != nil {
		return zero, e
	}
	if jobs >= s.limits.MaxJobs {
		return zero, ErrLimit
	}
	id, e := token()
	if e != nil {
		return zero, e
	}
	if e = s.root.Mkdir(id, 0700); e != nil {
		return zero, e
	}
	if e = s.syncDir("."); e != nil {
		return zero, e
	}
	// Failed/unacknowledged uploads are retained too. List exposes only manifests;
	// Usage includes incomplete directories. No automatic deletion is performed.
	cap := min(s.limits.MaxUploadBytes, s.limits.MaxBytes-used-maxManifest)
	if cap < 1 {
		return zero, ErrLimit
	}
	blob, e := s.writeBlob(ctx, id, "input", cap, func(dst io.Writer) error { _, err := io.CopyBuffer(dst, src, make([]byte, 32<<10)); return err })
	if e != nil {
		return zero, e
	}
	if blob.Bytes == 0 {
		return zero, fmt.Errorf("empty upload")
	}
	now := time.Now().UTC()
	m := Manifest{Schema: 1, ID: id, Name: displayName, Configuration: string(configuration), ConfigurationSHA256: hash(configuration), Input: blob, Status: Queued, Created: now, Updated: now, Checkpoints: []Checkpoint{}}
	if e = ctx.Err(); e != nil {
		return zero, e
	}
	if e = s.save(m); e != nil {
		return zero, errors.Join(ErrPersistence, e)
	}
	return clone(m), nil
}

// Rename changes only user-facing title metadata. The original upload name,
// input identity, configuration, status and checkpoints remain immutable.
func (s *Store) Rename(ctx context.Context, id, title string) (Manifest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var zero Manifest
	if e := s.ready(); e != nil {
		return zero, e
	}
	if s.busy {
		return zero, ErrBusy
	}
	if e := ctx.Err(); e != nil {
		return zero, e
	}
	if !validID(id) || !validTitle(title) || title == "" {
		return zero, fmt.Errorf("invalid recording title")
	}
	m, e := s.load(id)
	if e != nil {
		return zero, e
	}
	if m.Title == title {
		return clone(m), nil
	}
	m.Title = title
	m.Updated = time.Now().UTC()
	if e = s.save(m); e != nil {
		return zero, errors.Join(ErrPersistence, e)
	}
	return clone(m), nil
}

func (s *Store) Get(id string) (Manifest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.ready(); e != nil {
		return Manifest{}, e
	}
	m, e := s.load(id)
	return clone(m), e
}
func (s *Store) List() ([]Manifest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.ready(); e != nil {
		return nil, e
	}
	entries, e := fs.ReadDir(s.root.FS(), ".")
	if e != nil {
		return nil, e
	}
	out := []Manifest{}
	for _, entry := range entries {
		if !entry.IsDir() || !validID(entry.Name()) {
			continue
		}
		m, e := s.load(entry.Name())
		if errors.Is(e, os.ErrNotExist) {
			continue
		}
		if e != nil {
			return nil, e
		}
		out = append(out, clone(m))
	}
	return out, nil
}
func (s *Store) Usage() (int64, int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.ready(); e != nil {
		return 0, 0, e
	}
	if s.busy {
		return 0, 0, ErrBusy
	}
	return s.usage()
}

func (s *Store) openBlob(ctx context.Context, id string, b Blob) (io.ReadCloser, error) {
	if !validID(id) || path.Base(b.File) != b.File || b.Bytes < 0 || !validHash(b.SHA256) {
		return nil, ErrCorrupt
	}
	f, e := s.root.Open(id + "/" + b.File)
	if e != nil {
		return nil, fmt.Errorf("%w: %v", ErrCorrupt, e)
	}
	fail := func(e error) (io.ReadCloser, error) { f.Close(); return nil, e }
	st, e := f.Stat()
	if e != nil || !st.Mode().IsRegular() || st.Size() != b.Bytes {
		return fail(ErrCorrupt)
	}
	h := sha256.New()
	buf := make([]byte, 32<<10)
	for {
		if e = ctx.Err(); e != nil {
			return fail(e)
		}
		n, re := f.Read(buf)
		if n > 0 {
			h.Write(buf[:n])
		}
		if re == io.EOF {
			break
		}
		if re != nil {
			return fail(re)
		}
	}
	if hex.EncodeToString(h.Sum(nil)) != b.SHA256 {
		return fail(ErrCorrupt)
	}
	if _, e = f.Seek(0, io.SeekStart); e != nil {
		return fail(e)
	}
	return f, nil
}

// OpenCheckpoint verifies bytes before exposing a read-only payload, even when
// the job failed in a later stage. No source/audio download API is provided.
func (s *Store) OpenCheckpoint(ctx context.Context, id, stage string) (io.ReadCloser, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.ready(); e != nil {
		return nil, e
	}
	m, e := s.load(id)
	if e != nil {
		return nil, e
	}
	for _, cp := range m.Checkpoints {
		if cp.Stage == stage {
			return s.openBlob(ctx, id, cp.Blob)
		}
	}
	return nil, os.ErrNotExist
}
