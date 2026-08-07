package model

import "fmt"

// Gemma4KVLayerStats reports the logical and reserved float32 KV storage owned
// by one request-scoped decode session. Shared-KV consumer layers legitimately
// report zero bytes because their source layer owns the cache.
type Gemma4KVLayerStats struct {
	Layer         int
	UsedBytes     int64
	ReservedBytes int64
}

// Gemma4KVStats separates logical KV payload from Go slice capacity. The
// difference is allocator growth slack, not page-table fragmentation; it is the
// relevant baseline measurement before considering a block pool or paged KV.
type Gemma4KVStats struct {
	Layers              []Gemma4KVLayerStats
	UsedBytes           int64
	ReservedBytes       int64
	UnusedReservedBytes int64
}

// KVStats snapshots the current request-owned CPU KV footprint.
func (s *Gemma4DecodeSession) KVStats() (Gemma4KVStats, error) {
	if err := s.usable(); err != nil {
		return Gemma4KVStats{}, err
	}
	if s.state == nil {
		return Gemma4KVStats{}, nil
	}
	if len(s.state.kvCacheK) != len(s.state.kvCacheV) {
		return Gemma4KVStats{}, fmt.Errorf("Gemma4 session KV layer mismatch K=%d V=%d", len(s.state.kvCacheK), len(s.state.kvCacheV))
	}
	stats := Gemma4KVStats{Layers: make([]Gemma4KVLayerStats, len(s.state.kvCacheK))}
	for layer := range s.state.kvCacheK {
		used := int64(len(s.state.kvCacheK[layer])+len(s.state.kvCacheV[layer])) * 4
		reserved := int64(cap(s.state.kvCacheK[layer])+cap(s.state.kvCacheV[layer])) * 4
		stats.Layers[layer] = Gemma4KVLayerStats{Layer: layer, UsedBytes: used, ReservedBytes: reserved}
		stats.UsedBytes += used
		stats.ReservedBytes += reserved
	}
	stats.UnusedReservedBytes = stats.ReservedBytes - stats.UsedBytes
	return stats, nil
}
