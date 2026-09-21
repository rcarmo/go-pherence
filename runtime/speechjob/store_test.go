package speechjob

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func limits() Limits {
	return Limits{MaxJobs: 10, MaxUploadBytes: 1 << 20, MaxArtifactBytes: 1 << 20, MaxBytes: 8 << 20}
}
func openTest(t *testing.T) (*Store, string) {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "private-store")
	s, e := Open(dir, limits())
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { s.Close() })
	return s, dir
}

var config = []byte(`{"model":"sha256:example","backend":"ffmpeg","task":"transcribe","language":"pt","schema":1}`)

func stage(name string, run func(context.Context, *Input, io.Writer) error) Stage {
	return Stage{Name: name, Version: hash([]byte(name + "v1")), Run: run}
}
func textStage(name, text string) Stage {
	return stage(name, func(_ context.Context, _ *Input, w io.Writer) error { _, e := io.WriteString(w, text); return e })
}
func createTest(t *testing.T, s *Store) Manifest {
	t.Helper()
	m, e := s.Create(context.Background(), "../../inert-name.m4a", config, strings.NewReader("source audio"))
	if e != nil {
		t.Fatal(e)
	}
	return m
}
func readAll(t *testing.T, r io.ReadCloser, e error) string {
	t.Helper()
	if e != nil {
		t.Fatal(e)
	}
	b, e := io.ReadAll(r)
	ce := r.Close()
	if e != nil || ce != nil {
		t.Fatal(e, ce)
	}
	return string(b)
}
func TestStoreUploadRunResumeAndDownloads(t *testing.T) {
	s, dir := openTest(t)
	m := createTest(t, s)
	if m.Status != Queued || m.Input.Bytes != 12 || m.Input.SHA256 != hash([]byte("source audio")) {
		t.Fatal(m)
	}
	runs := 0
	sentinel := errors.New("speaker failure")
	asr := stage("asr", func(ctx context.Context, in *Input, w io.Writer) error {
		runs++
		r, e := in.OpenSource(ctx)
		if e != nil {
			return e
		}
		defer r.Close()
		if b, _ := io.ReadAll(r); string(b) != "source audio" {
			t.Fatal("input")
		}
		_, e = io.WriteString(w, "WEBVTT\n\n")
		return e
	})
	speaker := stage("speaker", func(context.Context, *Input, io.Writer) error { return sentinel })
	m, e := s.Run(context.Background(), m.ID, config, []Stage{asr, speaker}, nil)
	if !errors.Is(e, sentinel) || m.Status != Failed || len(m.Checkpoints) != 1 {
		t.Fatal(m, e)
	}
	r, e := s.OpenCheckpoint(context.Background(), m.ID, "asr")
	if readAll(t, r, e) != "WEBVTT\n\n" {
		t.Fatal("partial transcript")
	}
	if e = s.Close(); e != nil {
		t.Fatal(e)
	}
	s, e = Open(dir, limits())
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	speaker.Run = func(ctx context.Context, in *Input, w io.Writer) error {
		r, e := in.OpenCheckpoint(ctx, "asr")
		if e != nil {
			return e
		}
		defer r.Close()
		_, e = io.Copy(w, r)
		return e
	}
	m, e = s.Run(context.Background(), m.ID, config, []Stage{asr, speaker}, nil)
	if e != nil || m.Status != Complete || runs != 1 || m.Attempts != 2 || len(m.Checkpoints) != 2 {
		t.Fatal(m, e, runs)
	}
	// Completed run validates data and executes nothing.
	m, e = s.Run(context.Background(), m.ID, config, []Stage{asr, speaker}, nil)
	if e != nil || m.Attempts != 2 {
		t.Fatal(e)
	}
	snap, _ := s.Get(m.ID)
	snap.Checkpoints[0].Blob.SHA256 = "bad"
	again, _ := s.Get(m.ID)
	if again.Checkpoints[0].Blob.SHA256 == "bad" {
		t.Fatal("manifest alias")
	}
	all, e := s.List()
	if e != nil || len(all) != 1 {
		t.Fatal(e, all)
	}
	used, jobs, e := s.Usage()
	if e != nil || used <= m.Input.Bytes || jobs != 1 {
		t.Fatal(used, jobs, e)
	}
}
func TestStoreResumeRejectsChangedIdentityAndCorruption(t *testing.T) {
	for _, kind := range []string{"config", "stage", "version", "order", "input", "payload", "manifest", "extra"} {
		t.Run(kind, func(t *testing.T) {
			s, dir := openTest(t)
			m := createTest(t, s)
			stages := []Stage{textStage("decode", "pcm"), textStage("asr", "vtt")}
			m, e := s.Run(context.Background(), m.ID, config, stages, nil)
			if e != nil {
				t.Fatal(e)
			}
			cfg := append([]byte(nil), config...)
			switch kind {
			case "config":
				cfg = []byte(`{"backend":"other"}`)
			case "stage":
				stages[1].Name = "changed"
			case "version":
				stages[0].Version = hash([]byte("v2"))
			case "order":
				stages[0], stages[1] = stages[1], stages[0]
			case "input":
				os.WriteFile(filepath.Join(dir, m.ID, "input"), []byte("source AUDIO"), 0600)
			case "payload":
				os.WriteFile(filepath.Join(dir, m.ID, m.Checkpoints[0].Blob.File), []byte("bad"), 0600)
			case "manifest":
				os.WriteFile(filepath.Join(dir, m.ID, "manifest.json"), []byte(`{"schema":9}`), 0600)
			case "extra":
				stages = append(stages, textStage("speaker", "json"))
			}
			for i := range stages {
				stages[i].Run = func(context.Context, *Input, io.Writer) error { t.Fatal("ran incompatible job"); return nil }
			}
			if _, e = s.Run(context.Background(), m.ID, cfg, stages, nil); e == nil {
				t.Fatal("accepted", kind)
			}
		})
	}
}
func TestStoreLimitsAndMalformed(t *testing.T) {
	s, dir := openTest(t)
	for _, cfg := range [][]byte{nil, []byte(`[]`), []byte(`null`), []byte(`{"x":`), bytes.Repeat([]byte("a"), maxConfig+1)} {
		if _, e := s.Create(context.Background(), "x", cfg, strings.NewReader("audio")); e == nil {
			t.Fatal("config accepted")
		}
	}
	for _, name := range []string{"bad\nname", string([]byte{255}), strings.Repeat("x", 257)} {
		if _, e := s.Create(context.Background(), name, config, strings.NewReader("audio")); e == nil {
			t.Fatal("name accepted")
		}
	}
	for _, id := range []string{"../x", "", strings.Repeat("g", 32)} {
		if _, e := s.Get(id); e == nil {
			t.Fatal("id")
		}
	}
	if _, e := Open(dir, limits()); !errors.Is(e, ErrBusy) {
		t.Fatal("process lock", e)
	}
	s.Close()
	small := limits()
	small.MaxUploadBytes = 3
	s, e := Open(dir, small)
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	if _, e = s.Create(context.Background(), "big", config, strings.NewReader("1234")); !errors.Is(e, ErrLimit) {
		t.Fatal(e)
	}
	s.Close()
	small = limits()
	small.MaxArtifactBytes = 3
	s, e = Open(dir, small)
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	m := createTest(t, s)
	// A callback must not bypass limits by ignoring Writer errors.
	m, e = s.Run(context.Background(), m.ID, config, []Stage{stage("asr", func(_ context.Context, _ *Input, w io.Writer) error { _, _ = w.Write([]byte("1234")); return nil })}, nil)
	if !errors.Is(e, ErrLimit) || m.Status != Failed || len(m.Checkpoints) != 0 {
		t.Fatal(m, e)
	}
}
func TestStoreCancelExclusionPanicAndRecovery(t *testing.T) {
	s, _ := openTest(t)
	m := createTest(t, s)
	entered := make(chan struct{})
	done := make(chan error, 1)
	st := stage("asr", func(ctx context.Context, _ *Input, w io.Writer) error {
		close(entered)
		<-ctx.Done()
		_, _ = w.Write([]byte("partial"))
		return ctx.Err()
	})
	go func() { _, e := s.Run(context.Background(), m.ID, config, []Stage{st}, nil); done <- e }()
	<-entered
	if e := s.Close(); !errors.Is(e, ErrBusy) {
		t.Fatal(e)
	}
	if _, e := s.Run(context.Background(), m.ID, config, []Stage{textStage("asr", "x")}, nil); !errors.Is(e, ErrBusy) {
		t.Fatal(e)
	}
	if _, e := s.Create(context.Background(), "new", config, strings.NewReader("x")); !errors.Is(e, ErrBusy) {
		t.Fatal(e)
	}
	if s.Cancel(strings.Repeat("0", 32)) || !s.Cancel(m.ID) {
		t.Fatal("cancel scope")
	}
	if e := <-done; !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	m, _ = s.Get(m.ID)
	if m.Status != Cancelled || len(m.Checkpoints) != 0 {
		t.Fatal(m)
	}
	panicStage := stage("asr", func(context.Context, *Input, io.Writer) error { panic("owned worker failed") })
	m, e := s.Run(context.Background(), m.ID, config, []Stage{panicStage}, nil)
	if e == nil || m.Status != Failed {
		t.Fatal(m, e)
	}
	m, e = s.Run(context.Background(), m.ID, config, []Stage{textStage("asr", "good")}, nil)
	if e != nil || m.Status != Complete {
		t.Fatal(e)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, e := s.OpenCheckpoint(context.Background(), m.ID, "asr")
			if e != nil {
				t.Error(e)
				return
			}
			b, e := io.ReadAll(r)
			r.Close()
			if e != nil || string(b) != "good" {
				t.Error("concurrent read", e)
			}
		}()
	}
	wg.Wait()
}
func TestRenamePersistsTitleAndPreservesOriginalIdentity(t *testing.T) {
	s, dir := openTest(t)
	job := createTest(t, s)
	originalName, originalInput, originalConfig := job.Name, job.Input, job.ConfigurationSHA256
	renamed, err := s.Rename(context.Background(), job.ID, "Customer Interview")
	if err != nil || renamed.Title != "Customer Interview" || renamed.Name != originalName || renamed.Input != originalInput || renamed.ConfigurationSHA256 != originalConfig || !renamed.Updated.After(job.Updated) {
		t.Fatal(renamed, err)
	}
	if again, err := s.Rename(context.Background(), job.ID, "Customer Interview"); err != nil || again.Title != renamed.Title || !again.Updated.Equal(renamed.Updated) {
		t.Fatal("idempotent rename", again, err)
	}
	for _, title := range []string{"", " padded ", "bad\nname", strings.Repeat("x", 161)} {
		if _, err := s.Rename(context.Background(), job.ID, title); err == nil {
			t.Fatal("accepted title", title)
		}
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	s, err = Open(dir, limits())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	loaded, err := s.Get(job.ID)
	if err != nil || loaded.Title != "Customer Interview" || loaded.Name != originalName || loaded.Input != originalInput {
		t.Fatal(loaded, err)
	}
}

func TestStoreAtomicFailureAndOrphanConflict(t *testing.T) {
	for _, point := range []string{"payload-synced", "payload-published", "manifest-synced", "manifest-renamed"} {
		t.Run(point, func(t *testing.T) {
			s, dir := openTest(t)
			m := createTest(t, s)
			fired := false
			s.fault = func(p string) error {
				if p == point && !fired {
					fired = true
					return io.ErrUnexpectedEOF
				}
				return nil
			}
			_, _ = s.Run(context.Background(), m.ID, config, []Stage{textStage("asr", "transcript")}, nil)
			s.fault = nil
			s.Close()
			s, e := Open(dir, limits())
			if e != nil {
				t.Fatal(e)
			}
			defer s.Close()
			m, e = s.Run(context.Background(), m.ID, config, []Stage{textStage("asr", "transcript")}, nil)
			if e != nil || m.Status != Complete || !fired {
				t.Fatal(m, e)
			}
		})
	}
	s, _ := openTest(t)
	m := createTest(t, s)
	s.fault = func(p string) error {
		if p == "payload-published" {
			return io.ErrUnexpectedEOF
		}
		return nil
	}
	_, _ = s.Run(context.Background(), m.ID, config, []Stage{textStage("asr", "old")}, nil)
	s.fault = nil
	if _, e := s.Run(context.Background(), m.ID, config, []Stage{textStage("asr", "new")}, nil); !errors.Is(e, ErrCorrupt) {
		t.Fatal("orphan overwritten", e)
	}
}
func TestStoreSymlinkAndQuotaRecovery(t *testing.T) {
	s, dir := openTest(t)
	m := createTest(t, s)
	s.Close()
	if e := os.Symlink("/etc/passwd", filepath.Join(dir, m.ID, "evil")); e != nil {
		t.Fatal(e)
	}
	if _, e := Open(dir, limits()); !errors.Is(e, ErrCorrupt) {
		t.Fatal("symlink admitted", e)
	}
	os.Remove(filepath.Join(dir, m.ID, "evil"))
	small := limits()
	small.MaxBytes = maxManifest
	small.MaxUploadBytes = maxManifest
	small.MaxArtifactBytes = maxManifest
	os.WriteFile(filepath.Join(dir, "orphan"), make([]byte, maxManifest), 0600)
	if _, e := Open(dir, small); !errors.Is(e, ErrLimit) {
		t.Fatal("orphans bypass quota", e)
	}
}

func TestStoreRetentionInventoryAndDeleteRecovery(t *testing.T) {
	s, dir := openTest(t)
	m := createTest(t, s)
	if _, e := s.Run(context.Background(), m.ID, config, []Stage{textStage("asr", "vtt")}, nil); e != nil {
		t.Fatal(e)
	}
	inv, e := s.Inventory()
	if e != nil || len(inv) != 1 || !inv[0].ManifestPresent || inv[0].Corrupt || inv[0].Bytes < 12 {
		t.Fatal(inv, e)
	}
	cc, cancel := context.WithCancel(context.Background())
	cancel()
	if e := s.Delete(cc, m.ID); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	s.fault = func(p string) error {
		if p == "delete-renamed" {
			return io.ErrUnexpectedEOF
		}
		return nil
	}
	if e := s.Delete(context.Background(), m.ID); !errors.Is(e, io.ErrUnexpectedEOF) {
		t.Fatal(e)
	}
	s.fault = nil
	if _, e := s.Get(m.ID); !errors.Is(e, os.ErrNotExist) {
		t.Fatal("deleted job still visible", e)
	}
	inv, e = s.Inventory()
	if e != nil || len(inv) != 1 || !inv[0].Deleting {
		t.Fatal(inv, e)
	}
	s.Close()
	s, e = Open(dir, limits())
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	if e = s.Delete(context.Background(), m.ID); e != nil {
		t.Fatal(e)
	}
	if e = s.Delete(context.Background(), m.ID); e != nil {
		t.Fatal("idempotent", e)
	}
	inv, e = s.Inventory()
	if e != nil || len(inv) != 0 {
		t.Fatal(inv, e)
	}
	// Incomplete uploads remain visible and explicitly removable.
	if _, e = s.Create(context.Background(), "empty", config, strings.NewReader("")); e == nil {
		t.Fatal("empty")
	}
	inv, e = s.Inventory()
	if e != nil || len(inv) != 1 || inv[0].ManifestPresent {
		t.Fatal(inv, e)
	}
	if e = s.Delete(context.Background(), inv[0].ID); e != nil {
		t.Fatal(e)
	}
}

func TestStoreReportingAndPersistenceErrorContract(t *testing.T) {
	s, _ := openTest(t)
	m := createTest(t, s)
	calls := 0
	m, e := s.Run(context.Background(), m.ID, config, []Stage{textStage("asr", "ok")}, func(Manifest) { calls++; panic("reporter failed") })
	if e != nil || m.Status != Complete || calls < 2 {
		t.Fatal("observer changed job", m, e, calls)
	}
	next := createTest(t, s)
	s.fault = func(p string) error {
		if p == "manifest-synced" {
			return io.ErrClosedPipe
		}
		return nil
	}
	_, e = s.Run(context.Background(), next.ID, config, []Stage{textStage("asr", "ok")}, nil)
	if !errors.Is(e, ErrPersistence) {
		t.Fatal("missing persistence distinction", e)
	}
	s.fault = nil
	persisted, e := s.Get(next.ID)
	if e != nil || persisted.Status != Queued {
		t.Fatal("unpublished state treated as durable", persisted, e)
	}
}

func TestStoreUploadAckFailuresAndQuotaCaps(t *testing.T) {
	for _, point := range []string{"payload-synced", "payload-published", "manifest-synced", "manifest-renamed"} {
		t.Run(point, func(t *testing.T) {
			s, _ := openTest(t)
			s.fault = func(p string) error {
				if p == point {
					return io.ErrUnexpectedEOF
				}
				return nil
			}
			acknowledged, e := s.Create(context.Background(), "input", config, strings.NewReader("original bytes"))
			s.fault = nil
			if e == nil || acknowledged.ID != "" {
				t.Fatal("acknowledged failed upload", point, e)
			}
			inventory, e := s.Inventory()
			if e != nil || len(inventory) != 1 {
				t.Fatal(inventory, e)
			}
			id := inventory[0].ID
			if point == "manifest-renamed" {
				m, e := s.Get(id)
				if e != nil || m.Status != Queued {
					t.Fatal("ambiguous publication", e)
				}
				r, e := s.openBlob(context.Background(), id, m.Input)
				if readAll(t, r, e) != "original bytes" {
					t.Fatal("source lost")
				}
			}
			if e := s.Delete(context.Background(), id); e != nil {
				t.Fatal(e)
			}
		})
	}
	dir := filepath.Join(t.TempDir(), "private")
	lim := limits()
	lim.MaxJobs = 1
	s, e := Open(dir, lim)
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	m := createTest(t, s)
	if _, e = s.Create(context.Background(), "second", config, strings.NewReader("x")); !errors.Is(e, ErrLimit) {
		t.Fatal("job quota", e)
	}
	if e = s.Delete(context.Background(), m.ID); e != nil {
		t.Fatal(e)
	}
	if _, e = s.Create(context.Background(), "replacement", config, strings.NewReader("x")); e != nil {
		t.Fatal(e)
	}
	// Empty source and bad metadata never acknowledge success.
	cc, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e = s.Create(cc, "cancel", config, strings.NewReader("x")); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
}

func TestStoreUnicodeErrorSurvivesReopen(t *testing.T) {
	s, dir := openTest(t)
	m := createTest(t, s)
	long := errors.New(strings.Repeat("界", 500) + string([]byte{255}))
	m, e := s.Run(context.Background(), m.ID, config, []Stage{stage("asr", func(context.Context, *Input, io.Writer) error { return long })}, nil)
	if !errors.Is(e, long) || m.Status != Failed || len(m.Error) > 1024 {
		t.Fatal(m, e)
	}
	s.Close()
	s, e = Open(dir, limits())
	if e != nil {
		t.Fatal("error text corrupted manifest", e)
	}
	defer s.Close()
	loaded, e := s.Get(m.ID)
	if e != nil || loaded.Status != Failed || loaded.Error != m.Error {
		t.Fatal(e)
	}
}
