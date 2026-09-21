package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/half"
)

// FrozenGPUDecisionStats uses synchronised wall-clock phase timings, not CUDA
// event timings. Profiling adds one upload synchronisation and is opt-in.
type FrozenGPUDecisionStats struct {
	QueueSeconds           float64 `json:"queue_seconds"`
	EmbeddingUploadSeconds float64 `json:"embedding_upload_seconds"`
	PrefillSeconds         float64 `json:"prefill_seconds"`
	DownloadSeconds        float64 `json:"download_seconds"`
	ProjectionSeconds      float64 `json:"projection_seconds"`
	UploadBytes            int     `json:"upload_bytes"`
	DownloadBytes          int     `json:"download_bytes"`
}

func (e *FrozenGPUEncoder) ProfileSelectedLogits(ids, candidates []int) ([]float32, FrozenGPUDecisionStats, error) {
	var stats FrozenGPUDecisionStats
	logits, err := e.prefillSelectedLogits(ids, candidates, &stats)
	return logits, stats, err
}

// PrefillSelectedLogits performs one independent prompt prefill and projects
// only selected output-head rows. The transformer is GPU-resident; the bounded
// selected-row projection is CPU float64 accumulation over the actual BF16
// checkpoint rows, explicitly avoiding a full vocabulary buffer or allocation.
func (e *FrozenGPUEncoder) PrefillSelectedLogits(ids, candidates []int) ([]float32, error) {
	return e.prefillSelectedLogits(ids, candidates, nil)
}

func (e *FrozenGPUEncoder) prefillSelectedLogits(ids, candidates []int, timing *FrozenGPUDecisionStats) ([]float32, error) {
	if e == nil {
		return nil, fmt.Errorf("nil frozen GPU encoder")
	}
	started := time.Now()
	e.mu.Lock()
	defer e.mu.Unlock()
	if timing != nil {
		timing.QueueSeconds = time.Since(started).Seconds()
	}
	if e.closed {
		return nil, fmt.Errorf("encoder closed")
	}
	if len(candidates) < 2 || len(candidates) > 32 {
		return nil, fmt.Errorf("candidate count must be 2..32")
	}
	seen := map[int]bool{}
	for _, id := range candidates {
		if id < 0 || id >= e.cfg.VocabSize || seen[id] {
			return nil, fmt.Errorf("invalid or repeated candidate token %d", id)
		}
		seen[id] = true
	}
	raw, bias, err := e.selectedHeadLocked(candidates)
	if err != nil {
		return nil, err
	}
	rows, err := e.encodeTokenHiddenStatesLocked(ids, false, timing)
	if err != nil {
		return nil, err
	}
	started = time.Now()
	logits, err := selectedBF16Logits(rows[0], raw, e.cfg.VocabSize, candidates, bias)
	if timing != nil {
		timing.ProjectionSeconds = time.Since(started).Seconds()
	}
	return logits, err
}

func (e *FrozenGPUEncoder) selectedHeadLocked(candidates []int) ([]byte, []float32, error) {
	if len(candidates) < 2 || len(candidates) > 32 {
		return nil, nil, fmt.Errorf("candidate count must be 2..32")
	}
	seen := map[int]bool{}
	for _, id := range candidates {
		if id < 0 || id >= e.cfg.VocabSize || seen[id] {
			return nil, nil, fmt.Errorf("invalid candidate token")
		}
		seen[id] = true
	}
	name := "lm_head.weight"
	if e.cfg.TieEmbeddings {
		name = "model.embed_tokens.weight"
	}
	raw, dtype, shape, err := e.source.GetRaw(name)
	if err != nil {
		return nil, nil, err
	}
	if dtype != "BF16" || len(shape) != 2 || shape[0] != e.cfg.VocabSize || shape[1] != e.cfg.HiddenSize || shape[0] > int(^uint(0)>>1)/2/shape[1] || len(raw) != shape[0]*shape[1]*2 {
		return nil, nil, fmt.Errorf("invalid selected head tensor %s", name)
	}
	// Bias is optional, but malformed existing tensors must not be ignored.
	var bias []float32
	if _, _, _, err := e.source.GetRaw("lm_head.bias"); err == nil {
		var shape []int
		bias, shape, err = e.source.GetFloat32("lm_head.bias")
		if err != nil {
			return nil, nil, err
		}
		if len(shape) != 1 || shape[0] != e.cfg.VocabSize || len(bias) != e.cfg.VocabSize {
			return nil, nil, fmt.Errorf("invalid output bias")
		}
	} else if !strings.HasSuffix(err.Error(), "not found") && !strings.HasSuffix(err.Error(), "not in weight map") {
		return nil, nil, err
	}
	return raw, bias, nil
}

func selectedBF16Logits(hidden []float32, weights []byte, vocab int, ids []int, bias []float32) ([]float32, error) {
	width := len(hidden)
	if width == 0 || vocab <= 0 || width > int(^uint(0)>>1)/2/vocab || len(weights) != vocab*width*2 {
		return nil, fmt.Errorf("head shape mismatch")
	}
	if len(bias) != 0 && len(bias) != vocab {
		return nil, fmt.Errorf("head bias mismatch")
	}
	for _, x := range hidden {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return nil, fmt.Errorf("nonfinite hidden row")
		}
	}
	out := make([]float32, len(ids))
	for i, id := range ids {
		if id < 0 || id >= vocab {
			return nil, fmt.Errorf("head token out of range")
		}
		row := weights[id*width*2 : (id+1)*width*2]
		var sum float64
		for j, x := range hidden {
			w := half.BF16ToF32(binary.LittleEndian.Uint16(row[j*2:]))
			sum += float64(x) * float64(w)
		}
		if len(bias) > 0 {
			sum += float64(bias[id])
		}
		out[i] = float32(sum)
		if math.IsNaN(sum) || math.IsInf(float64(out[i]), 0) {
			return nil, fmt.Errorf("nonfinite selected logit")
		}
	}
	return out, nil
}
