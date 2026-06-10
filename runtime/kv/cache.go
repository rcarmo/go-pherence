package kv

import "github.com/rcarmo/go-pherence/internal/checked"

// CompressedKVCache wraps a per-layer KV cache with TurboQuant compression.
// Recent tokens (within the residual window) stay at full precision.
// Older tokens are compressed on demand.
type CompressedKVCache struct {
	// Full-precision storage for recent tokens
	FullK []float32 // [seqLen * kvDim] — full precision, appended per token
	FullV []float32

	// Compressed storage for older tokens
	CompressedK []compressedEntry
	CompressedV []compressedEntry

	// Reusable compression/decompression scratch buffers. GetK/GetV return
	// scratchK/scratchV when compressed entries exist, so callers must treat
	// returned slices as ephemeral.
	scratchK       []float32
	scratchV       []float32
	quantRotated   []float32
	quantIndices   []byte
	dequantRotated []float32
	dequantIndices []byte

	// Config
	kvDim          int
	numKVHeads     int
	headDim        int
	tq             *TurboQuantState
	isProtected    bool // if true, never compress this layer
	residualWindow int
	seqLen         int // total tokens stored (compressed + full)
}

type compressedEntry struct {
	Packed    []byte    // all heads' packed data concatenated
	HeadVMin  []float32 // per-head min values
	HeadScale []float32 // per-head scale values
}

// CompressedKVCacheStats summarizes a native TurboQuant cache layer.
type CompressedKVCacheStats struct {
	SeqLen          int   `json:"seq_len"`
	CompressedCount int   `json:"compressed"`
	FullCount       int   `json:"full"`
	StoredBytes     int64 `json:"stored_bytes"`
	ScratchBytes    int64 `json:"scratch_bytes"`
	TotalBytes      int64 `json:"total_bytes"`
}

// CompressedKVCacheAggregateStats summarizes a collection of native TurboQuant
// cache layers.
type CompressedKVCacheAggregateStats struct {
	Layers          int   `json:"layers"`
	SeqLen          int   `json:"seq_len"`
	CompressedCount int   `json:"compressed"`
	FullCount       int   `json:"full"`
	StoredBytes     int64 `json:"stored_bytes"`
	ScratchBytes    int64 `json:"scratch_bytes"`
	TotalBytes      int64 `json:"total_bytes"`
}

// NewCompressedKVCache creates a cache for one layer.
func NewCompressedKVCache(kvDim, numKVHeads, headDim int, tq *TurboQuantState, isProtected bool) *CompressedKVCache {
	if kvDim < 0 {
		kvDim = 0
	}
	if numKVHeads < 0 {
		numKVHeads = 0
	}
	if headDim < 0 {
		headDim = 0
	}
	if numKVHeads == 0 || headDim == 0 || numKVHeads*headDim != kvDim {
		numKVHeads = 0
		headDim = 0
	}
	rw := 128
	if tq != nil {
		rw = tq.Config.ResidualWindow
	}
	if rw < 0 {
		rw = 0
	}
	capHint, ok := checked.MulInt(2048, kvDim)
	if !ok {
		capHint = 0
	}
	return &CompressedKVCache{
		FullK:          make([]float32, 0, capHint),
		FullV:          make([]float32, 0, capHint),
		kvDim:          kvDim,
		numKVHeads:     numKVHeads,
		headDim:        headDim,
		tq:             tq,
		isProtected:    isProtected,
		residualWindow: rw,
	}
}

// Append adds a new K/V pair for the current position.
func (c *CompressedKVCache) Append(k, v []float32) {
	if c == nil || c.kvDim <= 0 || len(k) != c.kvDim || len(v) != c.kvDim {
		return
	}
	c.FullK = append(c.FullK, k...)
	c.FullV = append(c.FullV, v...)
	c.seqLen++

	// Compress old entries if we exceed the residual window
	if c.tq != nil && !c.isProtected && c.seqLen > c.residualWindow {
		c.compressOldest()
	}
}

// compressOldest moves the oldest full-precision entry to compressed storage.
func (c *CompressedKVCache) compressOldest() {
	if c == nil || c.kvDim <= 0 || c.numKVHeads <= 0 || c.headDim <= 0 || c.numKVHeads*c.headDim != c.kvDim || c.tq == nil {
		return
	}
	// How many full-precision entries we have
	fullCount := len(c.FullK) / c.kvDim
	if fullCount <= c.residualWindow {
		return
	}

	// Compress per-head for the oldest entry
	// Each head's K and V vectors are compressed independently
	kVec := c.FullK[:c.kvDim]
	vVec := c.FullV[:c.kvDim]

	bytesPerKeyHead, okKBytes := compressedBytesPerHead(c.headDim, c.tq.Config.KeyBits)
	bytesPerValueHead, okVBytes := compressedBytesPerHead(c.headDim, c.tq.Config.ValueBits)
	if !okKBytes || !okVBytes {
		return
	}
	var ek, ev compressedEntry
	ek.Packed = make([]byte, c.numKVHeads*bytesPerKeyHead)
	ev.Packed = make([]byte, c.numKVHeads*bytesPerValueHead)
	ek.HeadVMin = make([]float32, c.numKVHeads)
	ek.HeadScale = make([]float32, c.numKVHeads)
	ev.HeadVMin = make([]float32, c.numKVHeads)
	ev.HeadScale = make([]float32, c.numKVHeads)

	if cap(c.quantRotated) < c.headDim {
		c.quantRotated = make([]float32, c.headDim)
	}
	if cap(c.quantIndices) < c.headDim {
		c.quantIndices = make([]byte, c.headDim)
	}
	rotated := c.quantRotated[:c.headDim]
	indices := c.quantIndices[:c.headDim]
	for h := 0; h < c.numKVHeads; h++ {
		headK := kVec[h*c.headDim : (h+1)*c.headDim]
		headV := vVec[h*c.headDim : (h+1)*c.headDim]

		vMinK, scaleK, okK := c.tq.QuantizeKeyTo(ek.Packed[h*bytesPerKeyHead:(h+1)*bytesPerKeyHead], headK, rotated, indices)
		vMinV, scaleV, okV := c.tq.QuantizeValueTo(ev.Packed[h*bytesPerValueHead:(h+1)*bytesPerValueHead], headV, rotated, indices)
		if !okK || !okV {
			return
		}
		ek.HeadVMin[h] = vMinK
		ek.HeadScale[h] = scaleK
		ev.HeadVMin[h] = vMinV
		ev.HeadScale[h] = scaleV
	}

	c.CompressedK = append(c.CompressedK, ek)
	c.CompressedV = append(c.CompressedV, ev)

	// Remove oldest from full-precision
	c.FullK = c.FullK[c.kvDim:]
	c.FullV = c.FullV[c.kvDim:]
}

// GetK returns the full K cache as flat []float32 for attention.
// Decompresses compressed entries on-the-fly.
func (c *CompressedKVCache) GetK() []float32 {
	if c == nil || c.kvDim <= 0 {
		return nil
	}
	if len(c.CompressedK) == 0 {
		if c.seqLen > 0 {
			need, ok := checked.MulInt(c.seqLen, c.kvDim)
			if ok && len(c.FullK) > need {
				return c.FullK[:need]
			}
		}
		return c.FullK
	}
	if c.tq == nil || c.numKVHeads <= 0 || c.headDim <= 0 || c.numKVHeads*c.headDim != c.kvDim {
		return c.FullK
	}
	// Decompress + concatenate into reusable scratch storage.
	need, ok := checked.MulInt(c.seqLen, c.kvDim)
	if !ok {
		return c.FullK
	}
	if cap(c.scratchK) < need {
		c.scratchK = make([]float32, need)
	}
	out := c.scratchK[:need]
	write := 0
	bytesPerHead, ok := compressedBytesPerHead(c.headDim, c.tq.Config.KeyBits)
	if !ok {
		return c.FullK
	}
	if cap(c.dequantRotated) < c.headDim {
		c.dequantRotated = make([]float32, c.headDim)
	}
	if cap(c.dequantIndices) < c.headDim {
		c.dequantIndices = make([]byte, c.headDim)
	}
	rotated := c.dequantRotated[:c.headDim]
	indices := c.dequantIndices[:c.headDim]
	for _, entry := range c.CompressedK {
		if !compressedEntryValid(entry, c.numKVHeads, bytesPerHead) {
			return c.FullK
		}
		for h := 0; h < c.numKVHeads; h++ {
			packed := entry.Packed[h*bytesPerHead : (h+1)*bytesPerHead]
			if !c.tq.DequantizeKeyWithScratchTo(out[write:write+c.headDim], packed, entry.HeadVMin[h], entry.HeadScale[h], c.headDim, rotated, indices) {
				return c.FullK
			}
			write += c.headDim
		}
	}
	copy(out[write:], c.FullK)
	c.scratchK = out
	return out
}

// GetV returns the full V cache as flat []float32 for attention.
func (c *CompressedKVCache) GetV() []float32 {
	if c == nil || c.kvDim <= 0 {
		return nil
	}
	if len(c.CompressedV) == 0 {
		if c.seqLen > 0 {
			need, ok := checked.MulInt(c.seqLen, c.kvDim)
			if ok && len(c.FullV) > need {
				return c.FullV[:need]
			}
		}
		return c.FullV
	}
	if c.tq == nil || c.numKVHeads <= 0 || c.headDim <= 0 || c.numKVHeads*c.headDim != c.kvDim {
		return c.FullV
	}
	need, ok := checked.MulInt(c.seqLen, c.kvDim)
	if !ok {
		return c.FullV
	}
	if cap(c.scratchV) < need {
		c.scratchV = make([]float32, need)
	}
	out := c.scratchV[:need]
	write := 0
	bytesPerHead, ok := compressedBytesPerHead(c.headDim, c.tq.Config.ValueBits)
	if !ok {
		return c.FullV
	}
	if cap(c.dequantRotated) < c.headDim {
		c.dequantRotated = make([]float32, c.headDim)
	}
	if cap(c.dequantIndices) < c.headDim {
		c.dequantIndices = make([]byte, c.headDim)
	}
	rotated := c.dequantRotated[:c.headDim]
	indices := c.dequantIndices[:c.headDim]
	for _, entry := range c.CompressedV {
		if !compressedEntryValid(entry, c.numKVHeads, bytesPerHead) {
			return c.FullV
		}
		for h := 0; h < c.numKVHeads; h++ {
			packed := entry.Packed[h*bytesPerHead : (h+1)*bytesPerHead]
			if !c.tq.DequantizeValueWithScratchTo(out[write:write+c.headDim], packed, entry.HeadVMin[h], entry.HeadScale[h], c.headDim, rotated, indices) {
				return c.FullV
			}
			write += c.headDim
		}
	}
	copy(out[write:], c.FullV)
	c.scratchV = out
	return out
}

// SeqLen returns the total number of cached positions.
func (c *CompressedKVCache) SeqLen() int {
	if c == nil {
		return 0
	}
	return c.seqLen
}

// CompressedCount returns how many positions are compressed.
func (c *CompressedKVCache) CompressedCount() int {
	if c == nil {
		return 0
	}
	return len(c.CompressedK)
}

// FullCount returns how many positions are at full precision.
func (c *CompressedKVCache) FullCount() int {
	if c == nil || c.kvDim <= 0 {
		return 0
	}
	return len(c.FullK) / c.kvDim
}

// Reset clears the cache for reuse with a new sequence.
func (c *CompressedKVCache) Reset() {
	if c == nil {
		return
	}
	c.FullK = c.FullK[:0]
	c.FullV = c.FullV[:0]
	c.CompressedK = c.CompressedK[:0]
	c.CompressedV = c.CompressedV[:0]
	c.scratchK = c.scratchK[:0]
	c.scratchV = c.scratchV[:0]
	c.seqLen = 0
}

// MemoryBytes returns approximate stored cache usage (compressed + full, excluding slice headers and reusable scratch buffers).
func compressedBytesPerHead(headDim, bits int) (int, bool) {
	if headDim <= 0 || bits <= 0 {
		return 0, false
	}
	payloadBits, ok := checked.MulInt(headDim, bits)
	if !ok {
		return 0, false
	}
	withPadding, ok := checked.AddInt(payloadBits, 7)
	if !ok {
		return 0, false
	}
	return withPadding / 8, true
}

func compressedEntryValid(entry compressedEntry, heads, bytesPerHead int) bool {
	if heads <= 0 || bytesPerHead <= 0 {
		return false
	}
	packedLen, ok := checked.MulInt(heads, bytesPerHead)
	return ok && len(entry.Packed) >= packedLen && len(entry.HeadVMin) >= heads && len(entry.HeadScale) >= heads
}

func (c *CompressedKVCache) ScratchBytes() int64 {
	if c == nil {
		return 0
	}
	floatElems := 0
	for _, n := range []int{len(c.scratchK), len(c.scratchV), len(c.quantRotated), len(c.dequantRotated)} {
		var ok bool
		floatElems, ok = checked.AddInt(floatElems, n)
		if !ok {
			return checked.MaxInt64()
		}
	}
	byteElems, ok := checked.AddInt(len(c.quantIndices), len(c.dequantIndices))
	if !ok {
		return checked.MaxInt64()
	}
	return checked.SaturatingAddInt64(checked.SaturatingMulInt64(int64(floatElems), 4), int64(byteElems))
}

func (c *CompressedKVCache) TotalMemoryBytes() int64 {
	return checked.SaturatingAddInt64(c.MemoryBytes(), c.ScratchBytes())
}

func (c *CompressedKVCache) Stats() CompressedKVCacheStats {
	if c == nil {
		return CompressedKVCacheStats{}
	}
	stored := c.MemoryBytes()
	scratch := c.ScratchBytes()
	return CompressedKVCacheStats{SeqLen: c.SeqLen(), CompressedCount: c.CompressedCount(), FullCount: c.FullCount(), StoredBytes: stored, ScratchBytes: scratch, TotalBytes: checked.SaturatingAddInt64(stored, scratch)}
}

func AggregateCompressedKVCacheStats(caches []*CompressedKVCache) CompressedKVCacheAggregateStats {
	var out CompressedKVCacheAggregateStats
	for _, c := range caches {
		if c == nil {
			continue
		}
		st := c.Stats()
		out.Layers++
		if st.SeqLen > out.SeqLen {
			out.SeqLen = st.SeqLen
		}
		out.CompressedCount += st.CompressedCount
		out.FullCount += st.FullCount
		out.StoredBytes = checked.SaturatingAddInt64(out.StoredBytes, st.StoredBytes)
		out.ScratchBytes = checked.SaturatingAddInt64(out.ScratchBytes, st.ScratchBytes)
		out.TotalBytes = checked.SaturatingAddInt64(out.TotalBytes, st.TotalBytes)
	}
	return out
}

func (c *CompressedKVCache) MemoryBytes() int64 {
	if c == nil {
		return 0
	}
	fullElems, ok := checked.AddInt(len(c.FullK), len(c.FullV))
	if !ok {
		return checked.MaxInt64()
	}
	full := checked.SaturatingMulInt64(int64(fullElems), 4)
	compressed := int64(0)
	entryBytes := func(e compressedEntry) int64 {
		headElems, ok := checked.AddInt(len(e.HeadVMin), len(e.HeadScale))
		if !ok {
			return checked.MaxInt64()
		}
		return checked.SaturatingAddInt64(int64(len(e.Packed)), checked.SaturatingMulInt64(int64(headElems), 4))
	}
	for _, e := range c.CompressedK {
		compressed = checked.SaturatingAddInt64(compressed, entryBytes(e))
	}
	for _, e := range c.CompressedV {
		compressed = checked.SaturatingAddInt64(compressed, entryBytes(e))
	}
	return checked.SaturatingAddInt64(full, compressed)
}
