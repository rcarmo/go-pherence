package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// ForwardCachedChunk projects an already-normalised [rows,1024] query chunk
// within a five-row window and updates one stream's layer-0 key/value cache. It returns owned attention
// output. This bounded path checks cached relative scores and the chunk mask;
// it does not advance the rest of the encoder or convolution padding cache.
// Cache state belongs to one stream and must not be shared concurrently.
func (m *Encoder0Attention) ForwardCachedChunk(input []float32, rows, lookahead int, cache *Encoder0KVCache) ([]float32, error) {
	if m == nil || m.qkv == nil || len(m.relativeWeight) != encoderWidth*encoderWidth || len(m.biasU) != encoderWidth || len(m.biasV) != encoderWidth || len(m.outputWeight) != encoderWidth*encoderWidth {
		return nil, fmt.Errorf("invalid Nemotron ASR cached attention weights")
	}
	if cache == nil || rows < 1 || rows > 5 || len(input) != rows*encoderWidth || (lookahead != 0 && lookahead != 3) {
		return nil, fmt.Errorf("invalid Nemotron ASR cached attention window")
	}
	if cache.seen < 0 || cache.seen > 5-rows {
		return nil, fmt.Errorf("Nemotron ASR cached attention exceeds qualified five-row window")
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
	encoded := encoder0RelativePositions(keyRows)
	positions := 2*keyRows - 1
	relative := make([]float32, len(encoded))
	if !simd.DenseNTTo(relative, encoded, m.relativeWeight, positions, encoderWidth, encoderWidth, 1, encoderWidth, encoderWidth, encoderWidth) {
		return nil, fmt.Errorf("Nemotron ASR cached relative projection rejected shape")
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
				if globalSource/(lookahead+1) > globalRow/(lookahead+1) {
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
