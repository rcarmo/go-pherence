package nemotrondiarization

import "testing"

func TestNewCPUStreamHasFreshStateAndSharedWeights(t *testing.T) {
	loaded := &PCMStreamingRequest{
		frontend: &PCMStackingStream{stack: StackingStream{Projection: &StackingProjection{}, pending: [stackWidth]float32{1}, frames: 7, closed: true}},
		window:   &StreamingWindow{Tower: &OfflineAudioTower{}, Head: &OfflineHead{}, Cache: &SpeakerCache{compressor: &SpeakerCompressor{}, speaker: []float32{1}, fifo: []float32{2}}},
		pending:  []float32{3}, pcmTail: []float32{4}, samples: 16000, emitted: 72, closed: true,
	}
	first, err := loaded.NewCPUStream()
	if err != nil {
		t.Fatal(err)
	}
	second, err := loaded.NewCPUStream()
	if err != nil {
		t.Fatal(err)
	}
	if first.frontend == loaded.frontend || first.window == loaded.window || first.window.Cache == loaded.window.Cache || first.frontend == second.frontend || first.window.Cache == second.window.Cache {
		t.Fatal("mutable state shared")
	}
	if first.frontend.stack.Projection != loaded.frontend.stack.Projection || first.window.Tower != loaded.window.Tower || first.window.Head != loaded.window.Head || first.window.Cache.compressor != loaded.window.Cache.compressor {
		t.Fatal("immutable weights not shared")
	}
	if first.closed || first.samples != 0 || first.emitted != 0 || len(first.pending) != 0 || len(first.pcmTail) != 0 || first.frontend.stack.closed || first.frontend.stack.frames != 0 || first.frontend.stack.pending != [stackWidth]float32{} || len(first.window.Cache.speaker) != 0 || len(first.window.Cache.fifo) != 0 || first.Projector != nil || first.window.VulkanTower != nil || first.window.PTXTower != nil || first.frontend.stack.Device != nil {
		t.Fatal("attempt or device state inherited")
	}
	first.window.Cache.fifo = []float32{5}
	first.frontend.stack.pending[0] = 6
	if len(second.window.Cache.fifo) != 0 || second.frontend.stack.pending != [stackWidth]float32{} {
		t.Fatal("attempts coupled")
	}
}

func TestNewCPUStreamRejectsMissingModel(t *testing.T) {
	for _, model := range []*PCMStreamingRequest{nil, {}, {frontend: &PCMStackingStream{}}, {frontend: &PCMStackingStream{stack: StackingStream{Projection: &StackingProjection{}}}, window: &StreamingWindow{}}} {
		if _, err := model.NewCPUStream(); err == nil {
			t.Fatal("incomplete model accepted")
		}
	}
}
