package httpapi

import (
	"context"
	"encoding/json"
	"io"
	"path/filepath"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

func queuedHandler(t *testing.T, stages []speechjob.Stage) (*Handler, *speechjob.Store) {
	t.Helper()
	root := t.TempDir()
	s, e := speechjob.Open(filepath.Join(root, "store"), speechjob.Limits{MaxJobs: 8, MaxUploadBytes: 1024, MaxArtifactBytes: 1 << 20, MaxBytes: 8 << 20})
	if e != nil {
		t.Fatal(e)
	}
	h, e := New(Config{Store: s, Profiles: []Profile{{ID: "test", Configuration: []byte(`{}`), Stages: stages}}, Token: testToken, Hosts: []string{"speech.test"}, MaxUploadBytes: 1024, MaxConcurrentRequests: 4, Queue: &QueueOptions{Directory: filepath.Join(root, "queue"), MaxEntries: 8, MaxBytes: 1 << 20, JobTimeout: 5 * time.Second, Admission: speechjob.SerialAdmission()}})
	if e != nil {
		s.Close()
		t.Fatal(e)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if e := h.Shutdown(ctx); e != nil {
			t.Error(e)
		}
		if e := s.Close(); e != nil {
			t.Error(e)
		}
	})
	return h, s
}
func queuedState(t *testing.T, h *Handler, id string, status speechjob.QueueStatus) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		w := request(h, "GET", "/v1/queue", nil)
		if w.Code != 200 {
			t.Fatal(w)
		}
		var result struct {
			Entries []speechjob.QueueEntry `json:"entries"`
		}
		json.Unmarshal(w.Body.Bytes(), &result)
		for _, e := range result.Entries {
			if e.JobID == id && e.Status == status {
				return
			}
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("queue status timeout", status)
}
func TestHTTPDurableQueueExplicitRunAndCancel(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	h, _ := queuedHandler(t, []speechjob.Stage{testStage("transcript", func(ctx context.Context, _ *speechjob.Input, _ io.Writer) error {
		close(entered)
		<-ctx.Done()
		<-release
		return ctx.Err()
	})})
	j := upload(t, h)
	if w := request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil); w.Code != 409 {
		t.Fatal(w)
	}
	first := request(h, "POST", "/v1/jobs/"+j.ID+"/enqueue", nil)
	if first.Code != 202 {
		t.Fatal(first)
	}
	second := request(h, "POST", "/v1/jobs/"+j.ID+"/enqueue", nil)
	if second.Code != 202 || first.Body.String() != second.Body.String() {
		t.Fatal("enqueue not idempotent", second)
	}
	queuedState(t, h, j.ID, speechjob.QueuePending)
	if w := request(h, "DELETE", "/v1/jobs/"+j.ID, nil); w.Code != 409 {
		t.Fatal("deleted pending work", w)
	}
	if e := h.StartQueue(context.Background()); e != nil {
		t.Fatal(e)
	}
	<-entered
	if w := request(h, "POST", "/v1/jobs/"+j.ID+"/cancel", nil); w.Code != 202 {
		t.Fatal(w)
	}
	queuedState(t, h, j.ID, speechjob.QueueRunning)
	close(release)
	queuedState(t, h, j.ID, speechjob.QueueCancelled)
	if w := request(h, "DELETE", "/v1/jobs/"+j.ID+"/queue", nil); w.Code != 204 {
		t.Fatal(w)
	}
	if w := request(h, "DELETE", "/v1/jobs/"+j.ID, nil); w.Code != 204 {
		t.Fatal(w)
	}
}
func TestHTTPQueueSurvivesRequestContext(t *testing.T) {
	h, _ := queuedHandler(t, []speechjob.Stage{textStage("transcript", "background fixture")})
	j := upload(t, h)
	if w := request(h, "POST", "/v1/jobs/"+j.ID+"/enqueue", nil); w.Code != 202 {
		t.Fatal(w)
	}
	// Enqueue request has fully returned; only explicit worker lifetime controls run.
	if e := h.StartQueue(context.Background()); e != nil {
		t.Fatal(e)
	}
	queuedState(t, h, j.ID, speechjob.QueueSucceeded)
	w := request(h, "GET", "/v1/jobs/"+j.ID+"/artifacts/transcript", nil)
	if w.Code != 200 || w.Body.String() != "background fixture" {
		t.Fatal(w)
	}
}

func TestHTTPQueueAdmissionReleaseDrain(t *testing.T) {
	h, s := queuedHandler(t, []speechjob.Stage{textStage("transcript", "done")})
	// Replace only the test-owned shared callback via a separately constructed handler.
	h.Shutdown(context.Background())
	entered, release := make(chan struct{}), make(chan struct{})
	var e error
	h, e = New(Config{Store: s, Profiles: []Profile{{ID: "test", Configuration: []byte(`{}`), Stages: []speechjob.Stage{textStage("transcript", "done")}}}, Token: testToken, Hosts: []string{"speech.test"}, MaxUploadBytes: 1024, MaxConcurrentRequests: 2, Queue: &QueueOptions{Directory: filepath.Join(t.TempDir(), "queue"), MaxEntries: 8, MaxBytes: 1 << 20, JobTimeout: time.Second, Admission: func(context.Context) (func(), error) { return func() { close(entered); <-release }, nil }}})
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { h.Shutdown(context.Background()) })
	j := upload(t, h)
	request(h, "POST", "/v1/jobs/"+j.ID+"/enqueue", nil)
	h.StartQueue(context.Background())
	<-entered
	if w := request(h, "DELETE", "/v1/jobs/"+j.ID, nil); w.Code != 409 {
		t.Fatal("mutation before release drain", w)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if e = h.Shutdown(ctx); e == nil {
		t.Fatal("shutdown ignored release")
	}
	close(release)
	if e = h.Shutdown(context.Background()); e != nil {
		t.Fatal(e)
	}
}

func TestHTTPQueueReleasePanicFailsClosed(t *testing.T) {
	h, s := queuedHandler(t, []speechjob.Stage{textStage("transcript", "done")})
	h.Shutdown(context.Background())
	var e error
	h, e = New(Config{Store: s, Profiles: []Profile{{ID: "test", Configuration: []byte(`{}`), Stages: []speechjob.Stage{textStage("transcript", "done")}}}, Token: testToken, Hosts: []string{"speech.test"}, MaxUploadBytes: 1024, MaxConcurrentRequests: 2, Queue: &QueueOptions{Directory: filepath.Join(t.TempDir(), "queue"), MaxEntries: 8, MaxBytes: 1 << 20, JobTimeout: time.Second, Admission: func(context.Context) (func(), error) { return func() { panic("test uncertainty") }, nil }}})
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { h.Shutdown(context.Background()) })
	j := upload(t, h)
	request(h, "POST", "/v1/jobs/"+j.ID+"/enqueue", nil)
	h.StartQueue(context.Background())
	deadline := time.Now().Add(time.Second)
	for {
		_, e = h.queue.List()
		if e != nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("queue did not stop")
		}
		time.Sleep(time.Millisecond)
	}
	if w := request(h, "DELETE", "/v1/jobs/"+j.ID, nil); w.Code != 503 {
		t.Fatal("released uncertain mutation gate", w)
	}
	if w := request(h, "GET", "/v1/jobs/"+j.ID+"/artifacts/transcript", nil); w.Code != 200 {
		t.Fatal(w)
	}
	if e = h.Shutdown(context.Background()); e != nil {
		t.Fatal(e)
	}
}
func TestHTTPQueueRoutesRejectUnexpectedInput(t *testing.T) {
	h, _ := queuedHandler(t, []speechjob.Stage{textStage("transcript", "ok")})
	j := upload(t, h)
	for _, tc := range []struct {
		method, path string
		status       int
	}{{"POST", "/v1/queue", 405}, {"GET", "/v1/jobs/" + j.ID + "/enqueue", 405}, {"POST", "/v1/jobs/" + j.ID + "/enqueue?retry=true", 400}, {"POST", "/v1/jobs/" + j.ID + "/queue", 405}, {"GET", "/v1/queue?after=x", 400}} {
		w := request(h, tc.method, tc.path, nil)
		if w.Code != tc.status {
			t.Fatal(tc, w)
		}
	}
}
