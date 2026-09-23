package gosystemone

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"

	"github.com/rcarmo/go-pherence/internal/httpinput"
	"github.com/rcarmo/go-pherence/model"
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
	h.serve(w, r, false)
}

// ServeSystemOne shares admission with ServeHTTP; the two routes never execute
// concurrently against the scorer's owned scratch and caches.
func (h *Handler) ServeSystemOne(w http.ResponseWriter, r *http.Request) {
	h.serve(w, r, true)
}

func (h *Handler) serve(w http.ResponseWriter, r *http.Request, systemOne bool) {
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
	var typed SystemOneRequest
	var target any = &request
	if systemOne {
		target = &typed
	}
	if err := httpinput.DecodeJSON(w, r, target, MaxRequestBytes, true); err != nil {
		http.Error(w, "bad request: "+err.Error(), httpinput.ErrorStatus(err))
		return
	}
	if h.Engine == nil || h.ModelID == "" {
		http.Error(w, "decision engine is not configured", http.StatusServiceUnavailable)
		return
	}
	modelID := request.Model
	if systemOne {
		modelID = typed.Model
	}
	if modelID != "" && modelID != h.ModelID {
		http.Error(w, "unknown model", http.StatusBadRequest)
		return
	}
	var response any
	var err error
	if systemOne {
		typed.Model = h.ModelID
		response, err = h.Engine.SystemOne(r.Context(), typed)
	} else {
		request.Model = h.ModelID
		response, err = h.Engine.Decide(r.Context(), request)
	}
	if err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, model.ErrGemma4ContextCapacity) {
			status = http.StatusUnprocessableEntity
		}
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
