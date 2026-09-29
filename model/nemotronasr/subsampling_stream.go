package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// SubsamplingStream owns the three causal Conv2D time caches for one ASR
// stream. It accepts complete, fully valid mel chunks; terminal masks and
// encoder-layer caches have separate contracts. Weights are shared, but a
// stream must not be used concurrently. State is independent of duration.
type SubsamplingStream struct {
	Model   *Subsampling
	started bool
	last    [3][]float32 // last input time row at each stage, channel-major
}

// ForwardChunk returns owned [rows,1024] embeddings. Each Conv2D stage uses
// the previous input row; the first call also adds one leading zero row.
// The offline kernels add causal padding and one right-padded output, so
// their first and last rows are discarded to match cached convolutions.
func (s *SubsamplingStream) ForwardChunk(features []float32, frames int) ([]float32, error) {
	if s == nil || s.Model == nil || s.Model.Stem == nil || len(s.Model.linearWeight) != 1024*4352 || len(s.Model.linearBias) != 1024 || frames < 8 || frames > 128 || frames%8 != 0 || len(features) != frames*128 {
		return nil, fmt.Errorf("invalid Nemotron ASR subsampling stream chunk")
	}
	for _, v := range features {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR subsampling stream chunk")
		}
	}
	// Work on a copy: failures cannot partially advance cache state.
	state := *s
	first := !state.started
	rows, width := frames, 128
	input := features
	for layer := 0; layer < 3; layer++ {
		channels := subsamplingChannels
		if layer == 0 {
			channels = 1
		}
		padRows := 1
		if first {
			padRows = 2
		}
		previous := state.last[layer]
		if !first && len(previous) != channels*width {
			return nil, fmt.Errorf("invalid Nemotron ASR subsampling stream cache")
		}
		extendedRows := rows + padRows
		extended := make([]float32, channels*extendedRows*width)
		for ch := 0; ch < channels; ch++ {
			base := ch * extendedRows * width
			if !first {
				copy(extended[base:base+width], previous[ch*width:(ch+1)*width])
			}
			copy(extended[base+padRows*width:base+extendedRows*width], input[ch*rows*width:(ch+1)*rows*width])
		}
		next := make([]float32, channels*width)
		for ch := 0; ch < channels; ch++ {
			copy(next[ch*width:(ch+1)*width], input[(ch*rows+rows-1)*width:(ch*rows+rows)*width])
		}
		var raw []float32
		var outRows, outWidth int
		var err error
		if layer == 0 {
			raw, err = s.Model.Stem.ForwardOffline(extended, extendedRows)
			outRows, outWidth = extendedRows/2+1, 65
		} else {
			raw, outRows, outWidth, _, err = s.Model.layers[layer-1].forward(extended, extendedRows, width, extendedRows)
		}
		if err != nil {
			return nil, err
		}
		streamRows := (rows+padRows-3)/2 + 1
		if streamRows < 1 || streamRows+1 > outRows {
			return nil, fmt.Errorf("invalid Nemotron ASR stream output rows")
		}
		output := make([]float32, subsamplingChannels*streamRows*outWidth)
		for ch := 0; ch < subsamplingChannels; ch++ {
			for row := 0; row < streamRows; row++ {
				from := ch*outRows*outWidth + (row+1)*outWidth
				to := ch*streamRows*outWidth + row*outWidth
				copy(output[to:to+outWidth], raw[from:from+outWidth])
				if layer == 0 {
					for i := to; i < to+outWidth; i++ {
						if output[i] < 0 {
							output[i] = 0
						}
					}
				}
			}
		}
		state.last[layer] = next
		input, rows, width = output, streamRows, outWidth
	}
	if width*subsamplingChannels != 4352 {
		return nil, fmt.Errorf("invalid Nemotron ASR stream projection width")
	}
	flattened := make([]float32, rows*4352)
	for row := 0; row < rows; row++ {
		for ch := 0; ch < subsamplingChannels; ch++ {
			copy(flattened[row*4352+ch*width:row*4352+(ch+1)*width], input[ch*rows*width+row*width:ch*rows*width+(row+1)*width])
		}
	}
	out := make([]float32, rows*1024)
	if !simd.DenseNTTo(out, flattened, s.Model.linearWeight, rows, 1024, 4352, 1, 4352, 4352, 1024) {
		return nil, fmt.Errorf("Nemotron ASR stream projection rejected shape")
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < 1024; col++ {
			v := out[row*1024+col] + s.Model.linearBias[col]
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("non-finite Nemotron ASR stream projection")
			}
			out[row*1024+col] = v
		}
	}
	state.started = true
	*s = state
	return out, nil
}
