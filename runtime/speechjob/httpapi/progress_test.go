package httpapi

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

func TestProgressSnapshotAndSSEBoundary(t *testing.T) {
	h, _, _ := fixture(t, []speechjob.Stage{textStage("decode", "PCM")}, 4)
	j := upload(t, h)
	path := "/v1/jobs/" + j.ID + "/progress"
	w := request(h, "GET", path, nil)
	if w.Code != http.StatusOK {
		t.Fatal(w.Code, w.Body.String())
	}
	var p ProgressSnapshot
	if err := json.Unmarshal(w.Body.Bytes(), &p); err != nil || p.JobID != j.ID || p.Status != speechjob.Queued || p.Work != nil {
		t.Fatal(p, err)
	}
	if w := request(h, "GET", "/v1/jobs/"+strings.Repeat("a", 32)+"/progress", nil); w.Code != http.StatusNotFound {
		t.Fatal("unknown job", w.Code)
	}
	if w := request(h, "POST", path, nil); w.Code != http.StatusMethodNotAllowed {
		t.Fatal("mutation allowed", w.Code)
	}
	unauthorized := httptest.NewRequest("GET", "https://speech.test"+path, nil)
	missingToken := httptest.NewRecorder()
	h.ServeHTTP(missingToken, unauthorized)
	if missingToken.Code != http.StatusUnauthorized {
		t.Fatal("missing token accepted", missingToken.Code)
	}
	h.recordWorkProgress(j.ID, speechjob.WorkProgress{Stage: "asr-windows", Completed: 3, Total: 4, Phase: "windows"})
	if w := request(h, "GET", path, nil); !strings.Contains(w.Body.String(), `"status":"queued"`) || strings.Contains(w.Body.String(), `"work"`) {
		t.Fatal("stale work leaked", w.Body.String())
	}
	r := httptest.NewRequest("GET", "https://speech.test"+path, nil)
	r.Header.Set("Authorization", "Bearer "+testToken)
	r.Header.Set("Accept", "text/event-stream")
	sse := httptest.NewRecorder()
	h.ServeHTTP(sse, r)
	if sse.Code != http.StatusOK || !strings.HasPrefix(sse.Header().Get("Content-Type"), "text/event-stream") || !strings.Contains(sse.Body.String(), "event: progress\ndata: ") || strings.Contains(sse.Body.String(), `"work"`) {
		t.Fatal("snapshot", sse.Code, sse.Body.String())
	}
}

func TestProgressSSEStreamsRunningAndTerminal(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	h, _, _ := fixture(t, []speechjob.Stage{testStage("diarization", func(ctx context.Context, _ *speechjob.Input, out io.Writer) error {
		speechjob.ReportDiarizationProgress(ctx, 1, 2, "windows")
		close(started)
		select {
		case <-release:
			_, err := io.WriteString(out, "done")
			return err
		case <-ctx.Done():
			return ctx.Err()
		}
	})}, 4)
	j := upload(t, h)
	runDone := make(chan *httptest.ResponseRecorder, 1)
	go func() { runDone <- request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil) }()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("run did not start")
	}
	server := httptest.NewServer(h)
	defer server.Close()
	get := func() *http.Response {
		req, err := http.NewRequest("GET", server.URL+"/v1/jobs/"+j.ID+"/progress", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Host = "speech.test"
		req.Header.Set("Authorization", "Bearer "+testToken)
		req.Header.Set("Accept", "text/event-stream")
		response, err := (&http.Client{Timeout: 8 * time.Second}).Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return response
	}
	stream := get()
	defer stream.Body.Close()
	if stream.StatusCode != http.StatusOK {
		t.Fatal("stream", stream.Status)
	}
	line := bufio.NewReader(stream.Body)
	if event, err := line.ReadString('\n'); err != nil || event != "event: progress\n" {
		t.Fatal("initial event", event, err)
	}
	if data, err := line.ReadString('\n'); err != nil || !strings.Contains(data, `"completed":1`) || !strings.Contains(data, `"total":2`) || !strings.Contains(data, `"stage":"diarization"`) {
		t.Fatal("running snapshot", data, err)
	}
	second := get()
	second.Body.Close()
	if second.StatusCode != http.StatusServiceUnavailable {
		t.Fatal("unbounded streams", second.Status)
	}
	once.Do(func() { close(release) })
	select {
	case result := <-runDone:
		if result.Code != http.StatusOK {
			t.Fatal("run", result.Code, result.Body.String())
		}
	case <-time.After(5 * time.Second):
		t.Fatal("run did not complete")
	}
	body, err := io.ReadAll(stream.Body)
	if err != nil || !strings.Contains(string(body), `"status":"complete"`) || strings.Contains(string(body), `"work"`) {
		t.Fatal("terminal snapshot", string(body), err)
	}
}

func TestProgressNemotronTranscriptSampleCounters(t *testing.T) {
	ready, release := make(chan struct{}), make(chan struct{})
	h, _, _ := fixture(t, []speechjob.Stage{testStage("transcript", func(ctx context.Context, _ *speechjob.Input, w io.Writer) error {
		speechjob.ReportASRProgress(ctx, 80000, 160000, "samples")
		close(ready)
		<-release
		_, err := io.WriteString(w, "{}")
		return err
	})}, 4)
	j := upload(t, h)
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil) }()
	<-ready
	w := request(h, "GET", "/v1/jobs/"+j.ID+"/progress", nil)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"stage":"transcript"`) || !strings.Contains(w.Body.String(), `"completed":80000`) {
		t.Fatal(w.Code, w.Body.String())
	}
	close(release)
	if w := <-done; w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
}

func TestProgressRejectsInvalidCountersAndBoundsMap(t *testing.T) {
	h, _, _ := fixture(t, []speechjob.Stage{textStage("decode", "PCM")}, 4)
	h.recordWorkProgress("first", speechjob.WorkProgress{Stage: "asr-windows", Completed: 2, Total: 3})
	h.recordWorkProgress("first", speechjob.WorkProgress{Stage: "asr-windows", Completed: 4, Total: 3})
	h.recordWorkProgress("first", speechjob.WorkProgress{Stage: "invalid", Completed: 1, Total: 3})
	if p := h.workProgress["first"]; p.Completed != 2 || len(h.workProgress) != 1 {
		t.Fatal(h.workProgress)
	}
	h.recordWorkProgress("second", speechjob.WorkProgress{Stage: "diarization", Completed: 1, Total: 2})
	if _, ok := h.workProgress["first"]; ok || len(h.workProgress) != 1 {
		t.Fatal("stale progress retained", h.workProgress)
	}
}
