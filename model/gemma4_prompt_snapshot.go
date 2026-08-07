package model

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/rcarmo/go-pherence/runtime/promptcache"
)

var (
	gemma4PromptSnapshotBaseBytes  = int64(unsafe.Sizeof(Gemma4PromptSnapshot{}))
	gemma4PromptSnapshotSliceBytes = int64(unsafe.Sizeof([]float32(nil)))
)

// Gemma4PromptSnapshot is a prompt-cache snapshot for CPU Gemma4 decode.
// Shared-KV layers intentionally keep empty rows; only owning layers persist
// copied float32 K/V so restored sessions retain exclusive ownership.
type Gemma4PromptSnapshot struct {
	EndPos         int
	KVCacheK       [][]float32
	KVCacheV       [][]float32
	BoundaryLogits []float32
	BoundaryToken  int
}

func (s *Gemma4PromptSnapshot) Position() int {
	if s == nil {
		return 0
	}
	return s.EndPos
}

func (s *Gemma4PromptSnapshot) SizeBytes() (int64, error) {
	if s == nil {
		return 0, promptcache.ErrNilSnapshot
	}
	total := gemma4PromptSnapshotBaseBytes
	if err := gemma4PromptSnapshotValidateOuterLengths(s); err != nil {
		return 0, err
	}
	if add, ok := gemma4PromptSnapshotMulBytes(len(s.KVCacheK), gemma4PromptSnapshotSliceBytes); !ok {
		return 0, promptcache.ErrSizeOverflow
	} else if next, ok := gemma4PromptSnapshotAddBytes(total, add); !ok {
		return 0, promptcache.ErrSizeOverflow
	} else {
		total = next
	}
	if add, ok := gemma4PromptSnapshotMulBytes(len(s.KVCacheV), gemma4PromptSnapshotSliceBytes); !ok {
		return 0, promptcache.ErrSizeOverflow
	} else if next, ok := gemma4PromptSnapshotAddBytes(total, add); !ok {
		return 0, promptcache.ErrSizeOverflow
	} else {
		total = next
	}
	if add, ok := gemma4PromptSnapshotMulBytes(len(s.BoundaryLogits), 4); !ok {
		return 0, promptcache.ErrSizeOverflow
	} else if next, ok := gemma4PromptSnapshotAddBytes(total, add); !ok {
		return 0, promptcache.ErrSizeOverflow
	} else {
		total = next
	}
	for _, rows := range [][][]float32{s.KVCacheK, s.KVCacheV} {
		for _, row := range rows {
			add, ok := gemma4PromptSnapshotMulBytes(len(row), 4)
			if !ok {
				return 0, promptcache.ErrSizeOverflow
			}
			next, ok := gemma4PromptSnapshotAddBytes(total, add)
			if !ok {
				return 0, promptcache.ErrSizeOverflow
			}
			total = next
		}
	}
	return total, nil
}

func (s *Gemma4PromptSnapshot) Clone() promptcache.Snapshot {
	if s == nil {
		return (*Gemma4PromptSnapshot)(nil)
	}
	return &Gemma4PromptSnapshot{
		EndPos:         s.EndPos,
		KVCacheK:       cloneFloat32Matrix(s.KVCacheK),
		KVCacheV:       cloneFloat32Matrix(s.KVCacheV),
		BoundaryLogits: append([]float32(nil), s.BoundaryLogits...),
		BoundaryToken:  s.BoundaryToken,
	}
}

func newGemma4PromptSnapshotFromState(m *LlamaModel, st *cpuTokenState, pos int, logits []float32, token int) (*Gemma4PromptSnapshot, error) {
	if m == nil {
		return nil, fmt.Errorf("nil Gemma4 model")
	}
	if st == nil {
		return nil, fmt.Errorf("nil Gemma4 CPU token state")
	}
	if st.compressedKV != nil {
		return nil, fmt.Errorf("Gemma4 prompt snapshot requires float KV")
	}
	if pos < 0 {
		return nil, fmt.Errorf("Gemma4 prompt snapshot position=%d must be >= 0", pos)
	}
	cfg := m.Config
	if logits != nil && len(logits) != cfg.VocabSize {
		return nil, fmt.Errorf("Gemma4 prompt snapshot logits len=%d want %d", len(logits), cfg.VocabSize)
	}
	if logits != nil && (token < 0 || token >= cfg.VocabSize) {
		return nil, fmt.Errorf("Gemma4 prompt snapshot token=%d outside vocab=%d", token, cfg.VocabSize)
	}
	if len(st.kvCacheK) < cfg.NumLayers || len(st.kvCacheV) < cfg.NumLayers {
		return nil, fmt.Errorf("Gemma4 prompt snapshot KV layers=%d/%d want at least %d", len(st.kvCacheK), len(st.kvCacheV), cfg.NumLayers)
	}
	snap := &Gemma4PromptSnapshot{
		EndPos:         pos,
		KVCacheK:       make([][]float32, cfg.NumLayers),
		KVCacheV:       make([][]float32, cfg.NumLayers),
		BoundaryLogits: append([]float32(nil), logits...),
		BoundaryToken:  token,
	}
	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		if !layer.HasKV {
			continue
		}
		layerKVDim, err := m.LayerKVDim(l)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", l, err)
		}
		want, ok := checkedProduct(pos, layerKVDim)
		if !ok {
			return nil, fmt.Errorf("Gemma4 prompt snapshot KV size overflow layer=%d pos=%d kvDim=%d", l, pos, layerKVDim)
		}
		if len(st.kvCacheK[l]) != want || len(st.kvCacheV[l]) != want {
			return nil, fmt.Errorf("Gemma4 prompt snapshot layer %d KV len=%d/%d want %d", l, len(st.kvCacheK[l]), len(st.kvCacheV[l]), want)
		}
		snap.KVCacheK[l] = append([]float32(nil), st.kvCacheK[l]...)
		snap.KVCacheV[l] = append([]float32(nil), st.kvCacheV[l]...)
	}
	return snap, nil
}

func (s *Gemma4PromptSnapshot) restoreInto(m *LlamaModel, st *cpuTokenState) error {
	if s == nil {
		return fmt.Errorf("nil Gemma4 prompt snapshot")
	}
	if m == nil {
		return fmt.Errorf("nil Gemma4 model")
	}
	if st == nil {
		return fmt.Errorf("nil Gemma4 CPU token state")
	}
	if st.compressedKV != nil {
		return fmt.Errorf("Gemma4 prompt snapshot requires float KV")
	}
	cfg := m.Config
	if err := gemma4PromptSnapshotValidateOuterLengths(s); err != nil {
		return err
	}
	if len(s.BoundaryLogits) != 0 && len(s.BoundaryLogits) != cfg.VocabSize {
		return fmt.Errorf("Gemma4 prompt snapshot logits len=%d want 0 or %d", len(s.BoundaryLogits), cfg.VocabSize)
	}
	if len(st.kvCacheK) < cfg.NumLayers || len(st.kvCacheV) < cfg.NumLayers {
		return fmt.Errorf("Gemma4 prompt snapshot restore KV layers=%d/%d want at least %d", len(st.kvCacheK), len(st.kvCacheV), cfg.NumLayers)
	}
	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		if !layer.HasKV {
			if len(s.KVCacheK[l]) != 0 || len(s.KVCacheV[l]) != 0 {
				return fmt.Errorf("Gemma4 prompt snapshot shared layer %d must keep empty KV", l)
			}
			st.kvCacheK[l] = st.kvCacheK[l][:0]
			st.kvCacheV[l] = st.kvCacheV[l][:0]
			continue
		}
		layerKVDim, err := m.LayerKVDim(l)
		if err != nil {
			return fmt.Errorf("layer %d: %w", l, err)
		}
		want, ok := checkedProduct(s.EndPos, layerKVDim)
		if !ok {
			return fmt.Errorf("Gemma4 prompt snapshot restore KV size overflow layer=%d pos=%d kvDim=%d", l, s.EndPos, layerKVDim)
		}
		if len(s.KVCacheK[l]) != want || len(s.KVCacheV[l]) != want {
			return fmt.Errorf("Gemma4 prompt snapshot layer %d KV len=%d/%d want %d", l, len(s.KVCacheK[l]), len(s.KVCacheV[l]), want)
		}
		if cap(st.kvCacheK[l]) < want || cap(st.kvCacheV[l]) < want {
			return fmt.Errorf("Gemma4 prompt snapshot layer %d restore capacity=%d/%d want at least %d", l, cap(st.kvCacheK[l]), cap(st.kvCacheV[l]), want)
		}
		st.kvCacheK[l] = append(st.kvCacheK[l][:0], s.KVCacheK[l]...)
		st.kvCacheV[l] = append(st.kvCacheV[l][:0], s.KVCacheV[l]...)
	}
	st.position = s.EndPos
	return nil
}

func gemma4PromptSnapshotValidateOuterLengths(s *Gemma4PromptSnapshot) error {
	if s == nil {
		return promptcache.ErrNilSnapshot
	}
	if s.EndPos < 0 {
		return fmt.Errorf("Gemma4 prompt snapshot position=%d must be >= 0", s.EndPos)
	}
	if len(s.KVCacheK) != len(s.KVCacheV) {
		return fmt.Errorf("Gemma4 prompt snapshot KV layers=%d/%d mismatch", len(s.KVCacheK), len(s.KVCacheV))
	}
	return nil
}

func gemma4PromptSnapshotMulBytes(count int, elemBytes int64) (int64, bool) {
	if count < 0 || elemBytes < 0 {
		return 0, false
	}
	if count == 0 || elemBytes == 0 {
		return 0, true
	}
	c := int64(count)
	if c > math.MaxInt64/elemBytes {
		return 0, false
	}
	return c * elemBytes, true
}

func gemma4PromptSnapshotAddBytes(a, b int64) (int64, bool) {
	if a < 0 || b < 0 || a > math.MaxInt64-b {
		return 0, false
	}
	return a + b, true
}

func cloneFloat32Matrix(in [][]float32) [][]float32 {
	if in == nil {
		return nil
	}
	out := make([][]float32, len(in))
	for i, row := range in {
		if row == nil {
			continue
		}
		out[i] = append([]float32(nil), row...)
	}
	return out
}
