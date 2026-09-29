package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// SubsamplingStream owns the three causal Conv2D time caches for one ASR
// stream. It accepts valid and terminal-masked mel chunks; encoder-layer
// masks have a separate contract. Weights are shared, but a
// stream must not be used concurrently. State is independent of duration.
type SubsamplingStream struct {
	Model   *Subsampling
	started bool
	last    [3][]float32 // last input time row at each stage, channel-major
}

// ForwardChunk preserves the published full-valid, 8-aligned chunk contract.
func (s *SubsamplingStream) ForwardChunk(features []float32, frames int) ([]float32, error) {
	if frames < 8 || frames > 128 || frames%8 != 0 {
		return nil, fmt.Errorf("invalid Nemotron ASR subsampling stream chunk")
	}
	out, _, err := s.ForwardMaskedChunk(features, frames, frames)
	return out, err
}

// ForwardMaskedChunk returns owned [rows,1024] embeddings and the number of
// valid output rows. A terminal masked mel row (or right padding) is zeroed
// before convolution; each intermediate stage is masked after its stride,
// matching the pinned streaming encoder. The first call adds an extra leading
// zero row. Invalid input does not advance caches; the caller must not use a
// stream concurrently. The caller handles the encoder's output attention mask.
func (s *SubsamplingStream) ForwardMaskedChunk(features []float32, frames, valid int) ([]float32, int, error) {
	if s == nil || s.Model == nil || s.Model.Stem == nil || len(s.Model.linearWeight) != 1024*4352 || len(s.Model.linearBias) != 1024 || frames < 1 || frames > 128 || valid < 0 || valid > frames || len(features) != frames*128 {
		return nil, 0, fmt.Errorf("invalid Nemotron ASR subsampling stream chunk")
	}
	for _, v := range features {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, 0, fmt.Errorf("non-finite Nemotron ASR subsampling stream chunk")
		}
	}
	// Work on a copy: failures cannot partially advance cache state.
	state := *s
	first := !state.started
	rows, width := frames, 128
	input := make([]float32, len(features))
	copy(input, features[:valid*128])
	for layer := 0; layer < 3; layer++ {
		// The reference's streaming output_length is floor(input_length/2)
		// at every stage, including zero-length masked tails.
		valid /= 2
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
			return nil, 0, fmt.Errorf("invalid Nemotron ASR subsampling stream cache")
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
			return nil, 0, err
		}
		streamRows := (rows+padRows-3)/2 + 1
		if streamRows < 1 || streamRows+1 > outRows {
			return nil, 0, fmt.Errorf("invalid Nemotron ASR stream output rows")
		}
		output := make([]float32, subsamplingChannels*streamRows*outWidth)
		for ch := 0; ch < subsamplingChannels; ch++ {
			for row := 0; row < streamRows; row++ {
				from := ch*outRows*outWidth + (row+1)*outWidth
				to := ch*streamRows*outWidth + row*outWidth
				copy(output[to:to+outWidth], raw[from:from+outWidth])
				for i := to; i < to+outWidth; i++ {
					if row >= valid || output[i] < 0 {
						output[i] = 0
					}
				}
			}
		}
		state.last[layer] = next
		input, rows, width = output, streamRows, outWidth
	}
	if width*subsamplingChannels != 4352 {
		return nil, 0, fmt.Errorf("invalid Nemotron ASR stream projection width")
	}
	flattened := make([]float32, rows*4352)
	for row := 0; row < rows; row++ {
		for ch := 0; ch < subsamplingChannels; ch++ {
			copy(flattened[row*4352+ch*width:row*4352+(ch+1)*width], input[ch*rows*width+row*width:ch*rows*width+(row+1)*width])
		}
	}
	out := make([]float32, rows*1024)
	if !simd.DenseNTTo(out, flattened, s.Model.linearWeight, rows, 1024, 4352, 1, 4352, 4352, 1024) {
		return nil, 0, fmt.Errorf("Nemotron ASR stream projection rejected shape")
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < 1024; col++ {
			v := out[row*1024+col] + s.Model.linearBias[col]
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, 0, fmt.Errorf("non-finite Nemotron ASR stream projection")
			}
			out[row*1024+col] = v
		}
	}
	state.started = true
	*s = state
	return out, valid, nil
}
