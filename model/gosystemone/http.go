package gosystemone

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"

	"github.com/rcarmo/go-pherence/internal/httpinput"
)

const MaxRequestBytes int64 = 1 << 20

var ErrBusy = errors.New("decision inference busy; retry later")

type Handler struct {
	Engine  *Engine
	ModelID string
	gate    sync.Mutex
}

// Busy reports whether a decision request currently owns the scorer.
func (h *Handler) Busy() bool {
	if h == nil || h.gate.TryLock() {
		if h != nil {
			h.gate.Unlock()
		}
		return false
	}
	return true
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
		return
	}
	if !h.gate.TryLock() {
		http.Error(w, ErrBusy.Error(), http.StatusTooManyRequests)
		return
	}
	defer h.gate.Unlock()
	if err := r.Context().Err(); err != nil {
		http.Error(w, err.Error(), http.StatusRequestTimeout)
		return
	}
	defer r.Body.Close()
	var request Request
	if err := httpinput.DecodeJSON(w, r, &request, MaxRequestBytes, true); err != nil {
		http.Error(w, "bad request: "+err.Error(), httpinput.ErrorStatus(err))
		return
	}
	if h.Engine == nil || h.ModelID == "" {
		http.Error(w, "decision engine is not configured", http.StatusServiceUnavailable)
		return
	}
	if request.Model != "" && request.Model != h.ModelID {
		http.Error(w, "unknown model", http.StatusBadRequest)
		return
	}
	request.Model = h.ModelID
	response, err := h.Engine.Decide(r.Context(), request)
	if err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			status = http.StatusRequestTimeout
		}
		http.Error(w, err.Error(), status)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		return
	}
}
