package nemotrondiarization

import "fmt"

// NewCPUStream creates fresh frontend/cache state sharing immutable loaded weights.
// GPU projectors/towers are deliberately not inherited. The template and its
// weights must remain owned and immutable until all derived streams have drained.
func (s *PCMStreamingRequest) NewCPUStream() (*PCMStreamingRequest, error) {
	if s == nil || s.frontend == nil || s.frontend.stack.Projection == nil || s.window == nil || s.window.Tower == nil || s.window.Head == nil || s.window.Cache == nil || s.window.Cache.compressor == nil {
		return nil, fmt.Errorf("invalid Nemotron diarization CPU model")
	}
	return &PCMStreamingRequest{
		frontend: &PCMStackingStream{stack: StackingStream{Projection: s.frontend.stack.Projection}},
		window:   &StreamingWindow{Tower: s.window.Tower, Head: s.window.Head, Cache: &SpeakerCache{compressor: s.window.Cache.compressor}},
	}, nil
}
