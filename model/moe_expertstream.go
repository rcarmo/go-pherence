package model

import (
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/mlx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/runtime/expertstream"
)

// expertStreamSource is the experimental out-of-core boundary. Production
// decode does not construct one, so the ordinary resident path is unchanged.
type expertStreamSource interface {
	Load(keys []uint64) ([]expertstream.LoadedExpert, error)
}

func mlxWeightFromStreamComponent(c expertstream.Component) (*mlx.QuantWeight, error) {
	if c.Quant == nil || c.DType != expertstream.DTypeMLXQuant {
		return nil, fmt.Errorf("component is not MLX affine quantized")
	}
	q := c.Quant
	if q.Bits != 4 {
		return nil, fmt.Errorf("MLX upload supports 4-bit packed weights, got %d", q.Bits)
	}
	w, s, b, err := q.SplitBytes(c.Bytes)
	if err != nil {
		return nil, err
	}
	if len(w)%4 != 0 || len(s)%4 != 0 || len(b)%4 != 0 || !aligned4(w) || !aligned4(s) || !aligned4(b) {
		return nil, fmt.Errorf("MLX affine subregions must be 4-byte aligned")
	}
	return &mlx.QuantWeight{
		Weight:    unsafe.Slice((*uint32)(unsafe.Pointer(&w[0])), len(w)/4),
		Scales:    unsafe.Slice((*float32)(unsafe.Pointer(&s[0])), len(s)/4),
		Biases:    unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4),
		InDim:     int(q.InDim),
		OutDim:    int(q.OutDim),
		Groups:    int(q.InDim / q.GroupSize),
		GroupSize: int(q.GroupSize),
		Bits:      q.Bits,
	}, nil
}

func aligned4(b []byte) bool { return len(b) == 0 || uintptr(unsafe.Pointer(&b[0]))%4 == 0 }

// uploadStreamExpertsToPool performs hit-first planning: resident entries are
// touched in requested order, then only misses are read in one bounded batch
// and uploaded in that same deterministic order. Pool eviction and placement
// accounting remain owned by ExpertPool. It is intentionally not called by the
// default decode path; callers must explicitly inject a source.
func uploadStreamExpertsToPool(pool *nvidia.ExpertPool, source expertStreamSource, keys []int) error {
	if pool == nil || source == nil || len(keys) == 0 {
		return nil
	}
	missing := make([]uint64, 0, len(keys))
	seen := make(map[int]struct{}, len(keys))
	for _, key := range keys {
		if key < 0 {
			return fmt.Errorf("negative expert key %d", key)
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		if pool.Get(key) == nil {
			missing = append(missing, uint64(key))
		}
	}
	loaded, err := source.Load(missing)
	if err != nil {
		return err
	}
	for _, expert := range loaded {
		gate, err := mlxWeightFromStreamComponent(expert.Gate)
		if err != nil {
			return fmt.Errorf("expert %d gate: %w", expert.Key, err)
		}
		up, err := mlxWeightFromStreamComponent(expert.Up)
		if err != nil {
			return fmt.Errorf("expert %d up: %w", expert.Key, err)
		}
		down, err := mlxWeightFromStreamComponent(expert.Down)
		if err != nil {
			return fmt.Errorf("expert %d down: %w", expert.Key, err)
		}
		gw, e1 := nvidia.UploadMLXWeightNative(gate.Weight, gate.Scales, gate.Biases, gate.InDim, gate.OutDim, gate.GroupSize)
		uw, e2 := nvidia.UploadMLXWeightNative(up.Weight, up.Scales, up.Biases, up.InDim, up.OutDim, up.GroupSize)
		dw, e3 := nvidia.UploadMLXWeightNative(down.Weight, down.Scales, down.Biases, down.InDim, down.OutDim, down.GroupSize)
		if e1 != nil || e2 != nil || e3 != nil {
			nvidia.FreeExpertEntry(&nvidia.ExpertEntry{GateW: gw, UpW: uw, DownW: dw})
			return fmt.Errorf("expert %d upload: gate=%v up=%v down=%v", expert.Key, e1, e2, e3)
		}
		entry := &nvidia.ExpertEntry{ExpertID: int(expert.Key), GateW: gw, UpW: uw, DownW: dw, SizeBytes: int64(len(expert.Slot.Bytes))}
		if evicted := pool.Put(entry); evicted != nil {
			nvidia.FreeExpertEntry(evicted)
		}
	}
	return nil
}
