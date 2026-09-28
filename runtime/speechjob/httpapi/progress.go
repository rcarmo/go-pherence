package httpapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

// ProgressSnapshot is transient status; only checkpointed stage transitions and
// ASR window acknowledgements are durable. No raw error, model path or tokens.
type ProgressSnapshot struct {
	JobID  string                  `json:"job_id"`
	Status speechjob.Status        `json:"status"`
	Stage  string                  `json:"stage,omitempty"`
	Work   *speechjob.WorkProgress `json:"work,omitempty"`
}

func (h *Handler) clearWorkProgress() {
	h.mu.Lock()
	clear(h.workProgress)
	h.mu.Unlock()
}

func (h *Handler) recordWorkProgress(id string, p speechjob.WorkProgress) {
	if p.Completed < 0 || p.Total < 1 || p.Completed > p.Total || (p.Stage != "asr-windows" && p.Stage != "diarization") {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	// One store run is admitted at a time. Avoid retaining stale job entries.
	for key := range h.workProgress {
		if key != id {
			delete(h.workProgress, key)
		}
	}
	h.workProgress[id] = p
}

func (h *Handler) progressSnapshot(id string) (ProgressSnapshot, error) {
	m, err := h.store.Get(id)
	if err != nil {
		return ProgressSnapshot{}, err
	}
	p := ProgressSnapshot{JobID: id, Status: m.Status, Stage: m.ActiveStage}
	if m.Status == speechjob.Running {
		h.mu.Lock()
		work, ok := h.workProgress[id]
		h.mu.Unlock()
		if ok && work.Stage == m.ActiveStage {
			p.Work = &work
		}
	}
	return p, nil
}

// SSE sends an immediate snapshot, changed snapshots and periodic heartbeats.
// It requires the normal bearer/origin/host checks and holds a bounded HTTP slot.
// Reconnecting clients get a fresh snapshot; no event history is retained.
func (h *Handler) serveProgress(w http.ResponseWriter, r *http.Request, id string) {
	if r.Header.Get("Accept") != "text/event-stream" {
		p, err := h.progressSnapshot(id)
		if err != nil {
			h.failure(w, err, nil)
			return
		}
		respond(w, http.StatusOK, p)
		return
	}
	select {
	case h.progressStreams <- struct{}{}:
		defer func() { <-h.progressStreams }()
	default:
		respondError(w, http.StatusServiceUnavailable, "unavailable", nil)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		respondError(w, http.StatusInternalServerError, "unavailable", nil)
		return
	}
	first, err := h.progressSnapshot(id)
	if err != nil {
		h.failure(w, err, nil)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("X-Accel-Buffering", "no")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	last := []byte(nil)
	send := func(p ProgressSnapshot) bool {
		b, err := json.Marshal(p)
		if err != nil {
			return false
		}
		if !bytes.Equal(b, last) {
			if _, err = fmt.Fprintf(w, "event: progress\ndata: %s\n\n", b); err != nil {
				return false
			}
			last = b
		} else if _, err = fmt.Fprint(w, ": keepalive\n\n"); err != nil {
			return false
		}
		flusher.Flush()
		return true
	}
	if !send(first) || first.Status != speechjob.Running {
		return
	}
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			p, err := h.progressSnapshot(id)
			if err != nil || !send(p) || p.Status != speechjob.Running {
				return
			}
		}
	}
}
