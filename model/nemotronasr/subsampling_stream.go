package nemotronasr

import (
	"context"
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// SubsamplingStream owns the three causal Conv2D time caches for one ASR
// stream. It accepts masked mel chunks and the separate unmasked generation
// schedule; do not mix the two modes within a stream. Weights are shared,
// but a
// stream must not be used concurrently. State is independent of duration.
type SubsamplingStream struct {
	Model     *Subsampling
	Projector SubsamplingProjector // optional, per-request; nil uses CPU SIMD
	started   bool
	mode      uint8        // 0=new, 1=masked/full-valid, 2=generation
	last      [3][]float32 // last input time row at each stage, channel-major
}

// ForwardChunk preserves the published full-valid, 8-aligned chunk contract.
func (s *SubsamplingStream) ForwardChunk(features []float32, frames int) ([]float32, error) {
	if frames < 8 || frames > 128 || frames%8 != 0 {
		return nil, fmt.Errorf("invalid Nemotron ASR subsampling stream chunk")
	}
	out, _, err := s.ForwardMaskedChunk(features, frames, frames)
	return out, err
}

// ForwardUnmaskedChunk follows the pinned streaming generate path, which
// passes no attention mask to the encoder. It accepts the default lookahead-3
// first 25-row chunk followed by complete 32-row chunks. Its caches must not
// be mixed with ForwardMaskedChunk calls on the same stream.
func (s *SubsamplingStream) ForwardUnmaskedChunk(features []float32, frames int) ([]float32, error) {
	return s.ForwardUnmaskedChunkContext(context.Background(), features, frames)
}

// ForwardUnmaskedChunkContext propagates request cancellation to an optional
// projection backend. Projection failure leaves convolution caches unchanged.
func (s *SubsamplingStream) ForwardUnmaskedChunkContext(ctx context.Context, features []float32, frames int) ([]float32, error) {
	if ctx == nil {
		return nil, fmt.Errorf("nil Nemotron ASR subsampling context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s == nil || s.mode == 1 || (s.started && frames != 32) || (!s.started && frames != 25) {
		return nil, fmt.Errorf("invalid Nemotron ASR generation subsampling chunk")
	}
	out, _, err := s.forwardChunk(ctx, features, frames, frames, false, nil)
	return out, err
}

// ForwardMaskedChunk returns owned [rows,1024] embeddings and the number of
// internally valid convolution rows. A terminal masked mel row (or right
// padding) is zeroed before convolution; each intermediate stage is masked
// after its stride, matching the pinned streaming subsampler. The first call
// adds an extra leading zero row. The encoder computes its output attention
// mask separately with EncoderMaskRows: it can mark the projected masked
// convolution row visible. Invalid input does not advance caches; the caller
// must not use a stream concurrently.
func (s *SubsamplingStream) ForwardMaskedChunk(features []float32, frames, valid int) ([]float32, int, error) {
	if s != nil && s.mode == 2 {
		return nil, 0, fmt.Errorf("cannot mix Nemotron ASR subsampling cache modes")
	}
	return s.forwardChunk(context.Background(), features, frames, valid, true, nil)
}

// WithUnmaskedChunkContext consumes projection output before shared storage is
// reused. consume must not retain/mutate its input or reenter the backend. The
// convolution caches advance only after successful consumption; the generation
// caller closes the whole stream if later encoder work partially advances.
func (s *SubsamplingStream) WithUnmaskedChunkContext(ctx context.Context, features []float32, frames int, consume func([]float32) error) error {
	if ctx == nil || consume == nil {
		return fmt.Errorf("nil Nemotron ASR subsampling context/consumer")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if s == nil || s.mode == 1 || (s.started && frames != 32) || (!s.started && frames != 25) {
		return fmt.Errorf("invalid Nemotron ASR generation subsampling chunk")
	}
	_, _, err := s.forwardChunk(ctx, features, frames, frames, false, consume)
	return err
}

func (s *SubsamplingStream) forwardChunk(ctx context.Context, features []float32, frames, valid int, mask bool, consume func([]float32) error) ([]float32, int, error) {
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
					if (mask && row >= valid) || output[i] < 0 {
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
	flatten := func(flattened []float32) error {
		if len(flattened) != rows*4352 {
			return fmt.Errorf("invalid Nemotron ASR scoped projection input length")
		}
		for row := 0; row < rows; row++ {
			for ch := 0; ch < subsamplingChannels; ch++ {
				copy(flattened[row*4352+ch*width:row*4352+(ch+1)*width], input[ch*rows*width+row*width:ch*rows*width+(row+1)*width])
			}
		}
		return nil
	}
	var out []float32
	scoped, shared := s.Projector.(ScopedSubsamplingProjector)
	shared = shared && !mask && scoped.ScopedProjectionEnabled()
	if shared {
		err := scoped.ProjectScoped(ctx, s.Model.linearWeight, s.Model.linearBias, rows, flatten, func(projected []float32) error {
			if len(projected) != rows*1024 {
				return fmt.Errorf("invalid Nemotron ASR scoped projection output length")
			}
			for _, v := range projected {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return fmt.Errorf("non-finite Nemotron ASR stream projection")
				}
			}
			if consume != nil {
				return consume(projected)
			}
			out = append([]float32(nil), projected...) // public API owns output
			return nil
		})
		if err != nil {
			return nil, 0, fmt.Errorf("Nemotron ASR stream shared projection: %w", err)
		}
	} else {
	flattened := make([]float32, rows*4352)
	if err := flatten(flattened); err != nil {
		return nil, 0, err
	}
	if s.Projector != nil && !mask {
		var err error
		out, err = s.Projector.Project(ctx, flattened, s.Model.linearWeight, s.Model.linearBias, rows)
		if err != nil {
			return nil, 0, fmt.Errorf("Nemotron ASR stream device projection: %w", err)
		}
		if len(out) != rows*1024 {
			return nil, 0, fmt.Errorf("Nemotron ASR stream device projection returned %d values, want %d", len(out), rows*1024)
		}
	} else {
		out = make([]float32, rows*1024)
		if !simd.DenseNTTo(out, flattened, s.Model.linearWeight, rows, 1024, 4352, 1, 4352, 4352, 1024) {
			return nil, 0, fmt.Errorf("Nemotron ASR stream projection rejected shape")
		}
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < 1024; col++ {
			v := out[row*1024+col]
			if s.Projector == nil || mask {
				v += s.Model.linearBias[col]
			}
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, 0, fmt.Errorf("non-finite Nemotron ASR stream projection")
			}
			out[row*1024+col] = v
		}
	}
	if consume != nil {
		if err := consume(out); err != nil {
			return nil, 0, err
		}
		out = nil
	}
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	state.started = true
	if mask {
		state.mode = 1
	} else {
		state.mode = 2
	}
	*s = state
	return out, valid, nil
}
