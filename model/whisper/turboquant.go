package whisper

import "math"

// TurboQuantConfig configures KV cache compression for Whisper decoder.
type TurboQuantConfig struct {
	// Bits for K quantization (4 recommended)
	KBits int
	// Bits for V quantization (2-4)
	VBits int
	// Number of recent tokens to keep at full precision
	RecentWindow int
	// Whether to compress cross-attention KV (usually not worth it since it's fixed)
	CompressCrossKV bool
}

// DefaultTurboQuantConfig returns sensible defaults for Whisper.
func DefaultTurboQuantConfig() TurboQuantConfig {
	return TurboQuantConfig{
		KBits:           4,
		VBits:           4,
		RecentWindow:    32,
		CompressCrossKV: false,
	}
}

// CompressedKVCache is a KV cache that quantizes older entries for memory savings.
type CompressedKVCache struct {
	cfg       TurboQuantConfig
	dModel    int
	numLayers int

	// Full-precision recent window
	RecentK [][]float32 // [layer][recentWindow * dModel]
	RecentV [][]float32

	// Quantized older entries
	QuantK [][]uint8 // [layer][compressed_bytes]
	QuantV [][]uint8
	ScaleK [][]float32 // [layer][numGroups]
	ScaleV [][]float32

	// Token counts
	TotalTokens int
	QuantTokens int
}

// NewCompressedKVCache creates a new compressed KV cache for Whisper decoder.
func NewCompressedKVCache(numLayers, dModel int, cfg TurboQuantConfig) *CompressedKVCache {
	c := &CompressedKVCache{
		cfg:       cfg,
		dModel:    dModel,
		numLayers: numLayers,
		RecentK:   make([][]float32, numLayers),
		RecentV:   make([][]float32, numLayers),
		QuantK:    make([][]uint8, numLayers),
		QuantV:    make([][]uint8, numLayers),
		ScaleK:    make([][]float32, numLayers),
		ScaleV:    make([][]float32, numLayers),
	}
	for l := 0; l < numLayers; l++ {
		c.RecentK[l] = make([]float32, 0, cfg.RecentWindow*dModel)
		c.RecentV[l] = make([]float32, 0, cfg.RecentWindow*dModel)
	}
	return c
}

// Append adds a new K/V token to the cache, compressing old entries if the recent window is full.
func (c *CompressedKVCache) Append(layer int, k, v []float32) {
	if layer < 0 || layer >= c.numLayers || len(k) < c.dModel || len(v) < c.dModel {
		return
	}

	c.RecentK[layer] = append(c.RecentK[layer], k[:c.dModel]...)
	c.RecentV[layer] = append(c.RecentV[layer], v[:c.dModel]...)
	c.TotalTokens++

	// Check if we need to compress
	recentTokens := len(c.RecentK[layer]) / c.dModel
	if recentTokens > c.cfg.RecentWindow {
		// Compress the oldest token in the recent window
		c.compressOldest(layer)
	}
}

func (c *CompressedKVCache) compressOldest(layer int) {
	dModel := c.dModel

	// Take the first token from recent
	kVec := c.RecentK[layer][:dModel]
	vVec := c.RecentV[layer][:dModel]

	// Quantize
	qK, sK := quantizeVector(kVec, c.cfg.KBits)
	qV, sV := quantizeVector(vVec, c.cfg.VBits)

	c.QuantK[layer] = append(c.QuantK[layer], qK...)
	c.QuantV[layer] = append(c.QuantV[layer], qV...)
	c.ScaleK[layer] = append(c.ScaleK[layer], sK)
	c.ScaleV[layer] = append(c.ScaleV[layer], sV)

	// Remove from recent
	c.RecentK[layer] = c.RecentK[layer][dModel:]
	c.RecentV[layer] = c.RecentV[layer][dModel:]
	c.QuantTokens++
}

// quantizeVector quantizes a float32 vector to nbits per element.
// Returns packed bytes and a scale factor.
func quantizeVector(vec []float32, nbits int) ([]uint8, float32) {
	if len(vec) == 0 || nbits <= 0 {
		return nil, 0
	}

	// Find absmax
	maxVal := float32(0)
	for _, v := range vec {
		a := v
		if a < 0 {
			a = -a
		}
		if a > maxVal {
			maxVal = a
		}
	}
	if maxVal == 0 {
		return make([]uint8, (len(vec)*nbits+7)/8), 0
	}

	levels := (1 << nbits) - 1
	scale := maxVal / float32(levels/2)

	// Pack quantized values
	packed := make([]uint8, (len(vec)*nbits+7)/8)
	bitPos := 0
	for _, v := range vec {
		// Quantize to [0, levels]
		q := int(math.Round(float64(v/scale))) + levels/2
		if q < 0 {
			q = 0
		}
		if q > levels {
			q = levels
		}
		// Pack bits
		byteIdx := bitPos / 8
		bitOff := bitPos % 8
		if byteIdx < len(packed) {
			packed[byteIdx] |= uint8(q << bitOff)
			if bitOff+nbits > 8 && byteIdx+1 < len(packed) {
				packed[byteIdx+1] |= uint8(q >> (8 - bitOff))
			}
		}
		bitPos += nbits
	}

	return packed, scale
}

// MemorySavings returns the compression ratio of quantized vs full-precision storage.
// EstimateSelfKVBytes returns the float32 self-attention KV footprint for a
// decoder state with the given number of generated/prompt tokens. It is useful
// for deciding whether TurboQuant is worth enabling; chunked Whisper decoding
// usually keeps this small because state is reset per chunk.
func EstimateSelfKVBytes(numLayers, dModel, tokens int) int64 {
	if numLayers <= 0 || dModel <= 0 || tokens <= 0 {
		return 0
	}
	return int64(numLayers) * int64(tokens) * int64(dModel) * 2 * 4 // K+V float32
}

// EstimateCrossKVBytes returns the float32 cross-attention KV footprint for a
// decoder state after encoder K/V precompute. Cross KV is fixed for a chunk and
// is not helped by the current self-KV TurboQuant cache.
func EstimateCrossKVBytes(numLayers, dModel, encLen int) int64 {
	if numLayers <= 0 || dModel <= 0 || encLen <= 0 {
		return 0
	}
	return int64(numLayers) * int64(encLen) * int64(dModel) * 2 * 4 // K+V float32
}

func (c *CompressedKVCache) MemorySavings() float64 {
	if c.TotalTokens == 0 {
		return 1.0
	}
	fullBytes := float64(c.TotalTokens * c.dModel * 4 * 2) // K+V, float32
	quantBytes := float64(len(c.QuantK[0])+len(c.QuantV[0])) * float64(c.numLayers)
	recentBytes := float64(c.cfg.RecentWindow * c.dModel * 4 * 2 * c.numLayers)
	actualBytes := quantBytes + recentBytes
	if fullBytes == 0 {
		return 1.0
	}
	return actualBytes / fullBytes
}
