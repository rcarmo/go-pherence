package nemotronasr

import (
	"fmt"
	"math"
	"sync"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// Cached relative positions depend only on the visible K/V length. Share
// immutable encodings across layers and streams without retaining input or
// cache state; 61 covers the 56 retained rows plus a five-row query chunk.
var cachedRelativePositionTables [asrKVWindow + 4]struct {
	once   sync.Once
	values []float32
}

func cachedRelativePositions(rows int) []float32 {
	entry := &cachedRelativePositionTables[rows]
	entry.once.Do(func() { entry.values = encoder0RelativePositions(rows) })
	return entry.values
}

// ForwardCachedChunk projects an already-normalised [rows,1024] query chunk
// and updates one stream's layer-0 key/value cache. It returns owned attention
// output. This path checks cached relative scores and the chunk mask;
// it does not advance the rest of the encoder or convolution padding cache.
// Cache state belongs to one stream and must not be shared concurrently.
func (m *Encoder0Attention) ForwardCachedChunk(input []float32, rows, lookahead int, cache *Encoder0KVCache) ([]float32, error) {
	if m == nil || m.qkv == nil || len(m.relativeWeight) != encoderWidth*encoderWidth || len(m.biasU) != encoderWidth || len(m.biasV) != encoderWidth || len(m.outputWeight) != encoderWidth*encoderWidth {
		return nil, fmt.Errorf("invalid Nemotron ASR cached attention weights")
	}
	if cache == nil || rows < 1 || rows > 5 || len(input) != rows*encoderWidth || (lookahead != 0 && lookahead != 3) {
		return nil, fmt.Errorf("invalid Nemotron ASR cached attention window")
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR cached attention input")
		}
	}
	// The input is already layer-normalised by the caller; QKV.Project would
	// normalise it again. Project these three released matrices directly.
	q, k, v := make([]float32, len(input)), make([]float32, len(input)), make([]float32, len(input))
	for _, item := range []struct{ out, weight []float32 }{{q, m.qkv.q}, {k, m.qkv.k}, {v, m.qkv.v}} {
		if !simd.DenseNTTo(item.out, input, item.weight, rows, encoderWidth, encoderWidth, 1, encoderWidth, encoderWidth, encoderWidth) {
			return nil, fmt.Errorf("Nemotron ASR cached QKV projection rejected shape")
		}
	}
	// DynamicCache uses [head, time, dim] on the released 8-head layer.
	toHeads := func(values []float32) []float32 {
		out := make([]float32, len(values))
		for head := 0; head < asrAttentionHeads; head++ {
			for row := 0; row < rows; row++ {
				copy(out[(head*rows+row)*asrAttentionHeadWidth:(head*rows+row+1)*asrAttentionHeadWidth], values[row*encoderWidth+head*asrAttentionHeadWidth:row*encoderWidth+(head+1)*asrAttentionHeadWidth])
			}
		}
		return out
	}
	// Prepare the cache transition without committing it. Scoring or output
	// rejection must leave the caller's stream state unchanged.
	retained, seen := cache.retained, cache.seen
	if retained < 0 || retained >= asrKVWindow || seen < retained || seen > int(^uint(0)>>1)-rows {
		return nil, fmt.Errorf("invalid Nemotron ASR cached attention state")
	}
	keyRows := retained + rows
	// Encoder.forward passes the visible K/V length as cached_frames +
	// current rows. Once the sliding cache is full this is retained+rows,
	// not the cumulative seen count. The prepared-row fixture previously
	// passed cumulative get_seq_length() directly; regenerate that fixture.
	// _rel_shift consumes keyRows+rows-1 positions. Reuse each four-row
	// projection once per visible K/V length. Other query sizes keep their
	// original exact projection length and position origin.
	relative, err := m.projectedCachedRelative(keyRows, rows)
	if err != nil {
		return nil, err
	}
	prepared := *cache // Update replaces its slices; it never changes their backing arrays.
	visibleK, visibleV, err := prepared.Update(toHeads(k), toHeads(v), rows)
	if err != nil {
		return nil, err
	}
	mixed := make([]float32, len(input))
	scores := make([]float32, keyRows)
	scaling := float32(1 / math.Sqrt(asrAttentionHeadWidth))
	for head := 0; head < asrAttentionHeads; head++ {
		base := head * asrAttentionHeadWidth
		for row := 0; row < rows; row++ {
			globalRow := seen + row
			for source := 0; source < keyRows; source++ {
				var content, positional float32
				// _rel_shift for q rows and cached+current keys maps to
				// position q-1+source-row (not keyRows-1+source-row).
				pos := rows - 1 + source - row
				for dim := 0; dim < asrAttentionHeadWidth; dim++ {
					off := base + dim
					content += (q[row*encoderWidth+off] + m.biasU[off]) * visibleK[(head*keyRows+source)*asrAttentionHeadWidth+dim]
					positional += (q[row*encoderWidth+off] + m.biasV[off]) * relative[pos*encoderWidth+off]
				}
				scores[source] = (content + positional) * scaling
				globalSource := seen - retained + source
				chunk := lookahead + 1
				distance := globalRow/chunk - globalSource/chunk
				if distance < 0 || distance > (asrKVWindow-1)/chunk || globalSource > globalRow && lookahead == 0 {
					scores[source] = float32(math.Inf(-1))
				}
			}
			if !simd.SoftmaxInPlace(scores) {
				return nil, fmt.Errorf("Nemotron ASR cached attention softmax failed")
			}
			for dim := 0; dim < asrAttentionHeadWidth; dim++ {
				var sum float32
				for source := 0; source < keyRows; source++ {
					sum += scores[source] * visibleV[(head*keyRows+source)*asrAttentionHeadWidth+dim]
				}
				mixed[row*encoderWidth+base+dim] = sum
			}
		}
	}
	output := make([]float32, len(input))
	if !simd.DenseNTTo(output, mixed, m.outputWeight, rows, encoderWidth, encoderWidth, 1, encoderWidth, encoderWidth, encoderWidth) {
		return nil, fmt.Errorf("Nemotron ASR cached attention output rejected shape")
	}
	for _, value := range output {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR cached attention output")
		}
	}
	*cache = prepared
	return output, nil
}
