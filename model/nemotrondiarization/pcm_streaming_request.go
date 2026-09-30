package nemotrondiarization

import (
	"context"
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const lowLatencyFrames = 9
const lowLatencyLookahead = 4
const pcmTailReserve = 200 + stackFrames*160

// PCMStreamingRequest owns one low-latency stream. AppendPCM emits only
// committed 10-ms speaker logits; the four right-context encoder rows are
// processed again by the next window. The caller owns returned slices and
// must consume/store them before discarding them. Model weights are immutable
// and may be shared, while cache and frontend state must remain per stream.
type PCMStreamingRequest struct {
	frontend  *PCMStackingStream
	window    *StreamingWindow
	Projector StackingDeviceProjector // optional per-request GPU stack projection
	pending   []float32
	pcmTail   []float32 // retain a final mel group and STFT right context
	samples   int64
	emitted   int64
	closed    bool
}

// LoadPCMStreamingRequest loads the released weights and creates a fresh cache.
// The caller may close file after loading.
func LoadPCMStreamingRequest(file *safetensors.File) (*PCMStreamingRequest, error) {
	frontend, err := LoadPCMStackingStream(file)
	if err != nil {
		return nil, err
	}
	tower, err := LoadOfflineAudioTower(file)
	if err != nil {
		return nil, err
	}
	head, err := LoadOfflineHead(file)
	if err != nil {
		return nil, err
	}
	compressor, err := LoadSpeakerCompressor(file)
	if err != nil {
		return nil, err
	}
	cache, err := NewSpeakerCache(compressor)
	if err != nil {
		return nil, err
	}
	return &PCMStreamingRequest{frontend: frontend, window: &StreamingWindow{Tower: tower, Head: head, Cache: cache}}, nil
}

// AppendPCM accepts up to five seconds of finite mono 16-kHz PCM. On error
// before model inference, prior state remains intact. A model error closes
// the request: the frontend may already have consumed the input chunk.
func (s *PCMStreamingRequest) AppendPCM(pcm []float32) ([]float32, error) {
	return s.AppendPCMContext(context.Background(), pcm)
}

// AppendPCMContext checks cancellation before accepting PCM and between
// bounded encoder windows. Cancellation closes this per-stream request;
// already emitted output from previous calls remains owned by the caller.
func (s *PCMStreamingRequest) AppendPCMContext(ctx context.Context, pcm []float32) ([]float32, error) {
	if s == nil || s.closed || s.frontend == nil || s.window == nil || len(pcm) == 0 || len(pcm) > 16000*5 || s.samples > math.MaxInt64-200-int64(len(pcm)) {
		return nil, fmt.Errorf("invalid Nemotron diarization PCM stream")
	}
	for _, v := range pcm {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite Nemotron diarization PCM stream")
		}
	}
	if ctx == nil {
		return nil, fmt.Errorf("nil Nemotron diarization context")
	}
	if err := ctx.Err(); err != nil {
		s.closed = true
		return nil, err
	}
	s.frontend.stack.Device = s.Projector
	s.frontend.stack.ctx = ctx
	// Delay the last stack group until the final chunk's attention mask is
	// known. This bounds retained PCM independently of recording duration.
	s.pcmTail = append(s.pcmTail, pcm...)
	s.samples += int64(len(pcm))
	if len(s.pcmTail) > pcmTailReserve {
		feed := len(s.pcmTail) - pcmTailReserve
		embeds, err := s.frontend.AppendPCM(s.pcmTail[:feed])
		if err != nil {
			s.closed = true
			return nil, err
		}
		s.pending = append(s.pending, embeds...)
		copy(s.pcmTail, s.pcmTail[feed:])
		s.pcmTail = s.pcmTail[:pcmTailReserve]
	}
	out, err := s.runReady(ctx)
	if err != nil {
		s.closed = true
		return nil, err
	}
	return out, nil
}

func (s *PCMStreamingRequest) runReady(ctx context.Context) ([]float32, error) {
	var out []float32
	// A complete 13-row window is always non-final. The reference then
	// processes any residual four rows as a separate final chunk.
	for len(s.pending)/projectedWidth >= lowLatencyFrames+lowLatencyLookahead {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		chunk := s.pending[:(lowLatencyFrames+lowLatencyLookahead)*projectedWidth]
		cached := len(s.window.Cache.speaker)/projectedWidth + len(s.window.Cache.fifo)/projectedWidth
		_, all, err := s.window.ForwardPreparedContext(ctx, chunk, lowLatencyFrames, lowLatencyLookahead)
		if err != nil {
			return nil, err
		}
		start := cached * diarizationUpsample * diarizationSpeakers
		end := start + lowLatencyFrames*diarizationUpsample*diarizationSpeakers
		out = append(out, all[start:end]...)
		s.emitted += lowLatencyFrames * diarizationUpsample
		copy(s.pending, s.pending[lowLatencyFrames*projectedWidth:])
		s.pending = s.pending[:len(s.pending)-lowLatencyFrames*projectedWidth]
		// Explicitly observe cancellation after a completed window. The
		// caller receives no partial logits if it cancels before return.
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Finish scores the final window without lookahead and trims its logits to
// the processor's uncentred last-chunk frame count.
func (s *PCMStreamingRequest) Finish() ([]float32, error) {
	return s.FinishContext(context.Background())
}

// FinishContext checks cancellation before finalisation and between any
// complete windows. Once finalisation begins, errors close the request.
func (s *PCMStreamingRequest) FinishContext(ctx context.Context) ([]float32, error) {
	if s == nil || s.closed || s.frontend == nil || s.window == nil || s.samples == 0 {
		return nil, fmt.Errorf("invalid Nemotron diarization PCM stream finish")
	}
	if ctx == nil {
		return nil, fmt.Errorf("nil Nemotron diarization context")
	}
	if err := ctx.Err(); err != nil {
		s.closed = true
		return nil, err
	}
	s.closed = true
	s.frontend.stack.Device = s.Projector
	s.frontend.stack.ctx = ctx
	processed := s.samples - int64(len(s.pcmTail))
	features, err := s.frontend.mel.AppendPCM(s.pcmTail)
	if err != nil {
		return nil, err
	}
	s.pcmTail = nil
	// The first chunk is centred; later reference chunks are uncentred and
	// exclude any final mel frame whose 400-sample window extends past EOF.
	valid := s.samples / 160
	if s.samples >= 16640 { // first streaming window contains 104 mel frames
		valid = (s.samples-264)/160 + 1
	}
	// A first 104-frame window can be followed by a 103-frame final output.
	// Its 104th feature is lookahead for the first pass, then masked in the
	// separately scored final window.
	firstWindowBoundary := valid == 103 && s.samples >= 16640
	// Count frames already projected before feeding the reserved PCM tail.
	// Finish adds right-padded and masked rows that final streaming omits.
	var previousFrames int64
	if processed >= 200 {
		previousFrames = (processed-200)/160 + 1
	}
	endFeatures, err := s.frontend.mel.Finish()
	if err != nil {
		return nil, err
	}
	features = append(features, endFeatures...)
	// Finish includes the processor's masked row. Drop that row and any
	// right-padded real row before stacking the last incomplete group.
	projectThrough := valid
	if firstWindowBoundary {
		projectThrough = 104
	}
	keep := projectThrough - previousFrames
	if keep < 0 {
		return nil, fmt.Errorf("invalid Nemotron diarization terminal feature count: valid=%d previous=%d", valid, previousFrames)
	}
	if keep*melBins > int64(len(features)) {
		return nil, fmt.Errorf("invalid Nemotron diarization terminal feature count")
	}
	var embeds []float32
	var terminalGroup []float32
	if firstWindowBoundary {
		// The 104th feature is needed as lookahead in the first pass, but
		// must be masked when the final four rows are scored separately.
		combined := append([]float32(nil), s.frontend.stack.pending[:s.frontend.stack.frames*melBins]...)
		combined = append(combined, features[:int(keep)*melBins]...)
		if len(combined) < stackWidth || len(combined)%stackWidth != 0 {
			return nil, fmt.Errorf("invalid Nemotron diarization first window")
		}
		terminalGroup = append([]float32(nil), combined[len(combined)-stackWidth:]...)
	}
	if keep > 0 {
		embeds, err = s.frontend.stack.AppendFeatures(features[:int(keep)*melBins])
		if err != nil {
			return nil, err
		}
	}
	last, err := s.frontend.stack.Finish()
	if err != nil {
		return nil, err
	}
	s.pending = append(s.pending, embeds...)
	s.pending = append(s.pending, last...)
	out, err := s.runReady(ctx)
	if err != nil {
		return nil, err
	}
	if firstWindowBoundary {
		// The first pass used all 104 mel rows. Reproject its right-context
		// group for the terminal pass with the 104th row masked.
		clear(terminalGroup[7*melBins:])
		var corrected []float32
		if s.Projector != nil {
			corrected, err = s.frontend.stack.Projection.ProjectWithDevice(ctx, terminalGroup, stackFrames, s.Projector)
		} else {
			corrected, err = s.frontend.stack.Projection.Project(terminalGroup, stackFrames)
		}
		if err != nil {
			return nil, err
		}
		copy(s.pending[3*projectedWidth:4*projectedWidth], corrected)
	}
	needed := (valid + stackFrames - 1) / stackFrames
	present := s.emitted/diarizationUpsample + int64(len(s.pending)/projectedWidth)
	if present != needed || len(s.pending)%projectedWidth != 0 {
		return nil, fmt.Errorf("invalid Nemotron diarization terminal stack state")
	}
	remaining := len(s.pending) / projectedWidth
	if remaining > 0 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		cached := len(s.window.Cache.speaker)/projectedWidth + len(s.window.Cache.fifo)/projectedWidth
		_, all, err := s.window.ForwardPreparedContext(ctx, s.pending, remaining, 0)
		if err != nil {
			return nil, err
		}
		start := cached * diarizationUpsample * diarizationSpeakers
		out = append(out, all[start:start+remaining*diarizationUpsample*diarizationSpeakers]...)
		s.emitted += int64(remaining * diarizationUpsample)
		s.pending = nil
	}
	if s.emitted < valid || int64(len(out)) < (s.emitted-valid)*diarizationSpeakers {
		return nil, fmt.Errorf("invalid Nemotron diarization output length")
	}
	out = out[:len(out)-int(s.emitted-valid)*diarizationSpeakers]
	return out, nil
}
