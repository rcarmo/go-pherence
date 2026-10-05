package httpapi

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

// An engine update changes a profile's configuration. An explicit retry of a
// failed recording rebinds it to the current configuration of the same profile
// ID, discarding stale checkpoints, instead of failing with profile_changed.
func TestRetryRebindsRecordingAfterEngineUpdate(t *testing.T) {
	root := t.TempDir()
	s, e := speechjob.Open(filepath.Join(root, "store"), speechjob.Limits{MaxJobs: 8, MaxUploadBytes: 1024, MaxArtifactBytes: 1 << 20, MaxBytes: 8 << 20})
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	handler := func(queue string, profiles ...Profile) *Handler {
		h, e := New(Config{Store: s, Profiles: profiles, Token: testToken, Hosts: []string{"speech.test"}, MaxUploadBytes: 1024, MaxConcurrentRequests: 4, Queue: &QueueOptions{Directory: filepath.Join(root, queue), MaxEntries: 8, MaxBytes: 1 << 20, JobTimeout: 5 * time.Second, Admission: speechjob.SerialAdmission()}})
		if e != nil {
			t.Fatal(e)
		}
		if e := h.StartQueue(context.Background()); e != nil {
			t.Fatal(e)
		}
		return h
	}
	shutdown := func(h *Handler) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if e := h.Shutdown(ctx); e != nil {
			t.Fatal(e)
		}
	}
	wait := func(id string, status speechjob.Status) speechjob.Manifest {
		for deadline := time.Now().Add(3 * time.Second); time.Now().Before(deadline); time.Sleep(time.Millisecond) {
			if m, e := s.Get(id); e == nil && m.Status == status {
				return m
			}
		}
		m, e := s.Get(id)
		t.Fatal("status timeout", status, m.Status, m.Error, e)
		return speechjob.Manifest{}
	}
	decode := testStage("decode", func(_ context.Context, _ *speechjob.Input, w io.Writer) error {
		_, e := io.WriteString(w, "pcm")
		return e
	})
	failing := testStage("transcript", func(context.Context, *speechjob.Input, io.Writer) error { return io.ErrUnexpectedEOF })
	old := handler("queue-old", Profile{ID: "test", Configuration: []byte(`{"engine":1}`), Stages: []speechjob.Stage{decode, failing}})
	j := upload(t, old)
	if w := request(old, "POST", "/v1/jobs/"+j.ID+"/enqueue", nil); w.Code != 202 {
		t.Fatal(w)
	}
	before := wait(j.ID, speechjob.Failed)
	if len(before.Checkpoints) != 1 {
		t.Fatal("decode checkpoint", before.Checkpoints)
	}
	stale := before.Checkpoints[0].Blob.File
	shutdown(old)

	updated := handler("queue-new", Profile{ID: "test", Configuration: []byte(`{"engine":2}`), Stages: []speechjob.Stage{decode, textStage("transcript", `{"plain":"ok"}`)}})
	defer shutdown(updated)
	if got := decodeJob(t, request(updated, "GET", "/v1/jobs/"+j.ID, nil), 200); got.ProfileAvailable || got.Profile != "" {
		t.Fatal("stale recording reported available", got)
	}
	if w := request(updated, "POST", "/v1/jobs/"+j.ID+"/enqueue", nil); w.Code != 409 {
		t.Fatal("plain enqueue must not rebind", w)
	}
	if w := request(updated, "POST", "/v1/jobs/"+j.ID+"/retry-queued", nil); w.Code != 202 {
		t.Fatal(w)
	}
	queuedState(t, updated, j.ID, speechjob.QueueSucceeded)
	after := wait(j.ID, speechjob.Complete)
	if after.ConfigurationSHA256 == before.ConfigurationSHA256 || len(after.Checkpoints) != 2 || after.Attempts != 2 {
		t.Fatal("rebind", after)
	}
	if got := decodeJob(t, request(updated, "GET", "/v1/jobs/"+j.ID, nil), 200); !got.ProfileAvailable || got.Profile != "test" {
		t.Fatal("rebound recording not available", got)
	}
	if after.Checkpoints[0].Blob.File != stale {
		if _, e := os.Stat(filepath.Join(root, "store", j.ID, stale)); !os.IsNotExist(e) {
			t.Fatal("stale checkpoint retained", e)
		}
	}
	// Complete recordings and unknown profile IDs are never rebound.
	if _, e := s.Rebind(context.Background(), j.ID, []byte(`{"other":1}`)); e == nil {
		t.Fatal("complete recording rebound")
	}
	other := upload(t, updated)
	if _, e := s.Rebind(context.Background(), other.ID, []byte(`{"other":1}`)); e == nil {
		t.Fatal("queued (never run) recording rebound")
	}
	if !strings.Contains(request(updated, "POST", "/v1/jobs/nonexistent0000000000000000000000/retry-queued", nil).Body.String(), "error") {
		t.Fatal("missing job")
	}
}
