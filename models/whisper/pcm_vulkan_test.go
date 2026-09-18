package whisper

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestPCMVulkanAdmission(t *testing.T) {
	w := toyPCMModel()
	tok := checkedTestTokenizer(w.Config.VocabSize)
	reader := sampleReadFunc(func(context.Context, []float32, int64) (int, error) {
		t.Fatal("invalidVulkan reachedaudio")
		return 0, nil
	})
	emit := func(WindowTranscript) error { t.Fatal("invalidVulkan emitted"); return nil }
	newState := func() *vulkanEncoderState {
		return &vulkanEncoderState{gate: make(chan struct{}, 1), config: w.Config, stats: VulkanEncoderStats{Frames: w.Config.MaxLength}}
	}
	for _, mutate := range []func(*vulkanEncoderState){func(s *vulkanEncoderState) { s.closed = true }, func(s *vulkanEncoderState) { s.stopping = true }, func(s *vulkanEncoderState) { s.stats.Frames-- }, func(s *vulkanEncoderState) { s.config.NumMelBins++ }, func(s *vulkanEncoderState) { s.config.EncoderDModel++ }, func(s *vulkanEncoderState) { s.config.EncoderLayers++ }, func(s *vulkanEncoderState) { s.config.EncoderHeads++ }, func(s *vulkanEncoderState) { s.config.HeadDim++ }, func(s *vulkanEncoderState) { s.config.EncoderFFNDim++ }, func(s *vulkanEncoderState) { s.config.MaxLength++ }} {
		s := newState()
		mutate(s)
		if err := w.TranscribePCMWindows(context.Background(), reader, 1, tok, PCMTranscribeOptions{Language: "pt", VulkanEncoder: &VulkanEncoder{s: s}}, emit); err == nil {
			t.Fatal("acceptedbadresident")
		}
	}
	if err := w.TranscribePCMWindows(context.Background(), reader, 1, tok, PCMTranscribeOptions{Language: "pt", VulkanEncoder: &VulkanEncoder{}}, emit); err == nil {
		t.Fatal("zeroresident")
	}
	if err := w.TranscribePCMWindows(nil, reader, 1, tok, PCMTranscribeOptions{}, emit); err == nil {
		t.Fatal("nilcontext")
	}
	e := &VulkanEncoder{s: newState()}
	if err := e.checkPCMConfig(context.Background(), w.Config); err != nil {
		t.Fatal(err)
	}
	// Decoder-only differences need not reject an encoder with identical geometry.
	c := w.Config
	c.MaxDecoderLength++
	if err := e.checkPCMConfig(context.Background(), c); err != nil {
		t.Fatal("decoderdifference", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := e.checkPCMConfig(ctx, w.Config); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

// Called from the opt-in native suite; tests orchestration, not speech quality.
func nativePCMVulkanBridge(t *testing.T) {
	w := toyPCMModel()
	w.Config.MaxLength = 8
	w.Config.EncoderLayers = 1
	w.Config.EncoderFFNDim = 4
	w.Decoder.cfg = w.Config
	source := completeEncoderSource("audio", w.Config)
	e, err := LoadEncoderSource(source, "audio", w.Config)
	if err != nil {
		t.Fatal(err)
	}
	w.Encoder = e
	ctx := context.Background()
	gpu, err := NewVulkanEncoder(ctx, e, w.Config.MaxLength)
	if gpu != nil {
		t.Cleanup(func() {
			if err := gpu.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	tok := checkedTestTokenizer(w.Config.VocabSize)
	reader := sampleReadFunc(func(_ context.Context, dst []float32, _ int64) (int, error) { clear(dst); return len(dst), nil })
	opts := PCMTranscribeOptions{Language: "pt"}
	var cpu, device []WindowTranscript
	if err := w.TranscribePCMWindows(ctx, reader, 1281, tok, opts, func(out WindowTranscript) error { cpu = append(cpu, out); return nil }); err != nil {
		t.Fatal("CPUbridge", err)
	}
	// No CPU fallback: CPU encoder becomes invalid numerically after native copy.
	w.Encoder.Conv1Weight[0] = float32(math.NaN())
	opts.VulkanEncoder = gpu
	if err := w.TranscribePCMWindows(ctx, reader, 1281, tok, opts, func(out WindowTranscript) error { device = append(device, out); return nil }); err != nil {
		t.Fatal("GPUbridge", err)
	}
	if len(device) != 2 || !reflect.DeepEqual(cpu, device) {
		t.Fatal("PCMbridge difference", cpu, device)
	}
	// Once resident construction succeeds, host encoder storage is unnecessary.
	savedEncoder := w.Encoder
	w.Encoder = nil
	var released []WindowTranscript
	if err := w.TranscribePCMWindows(ctx, reader, 1281, tok, opts, func(out WindowTranscript) error { released = append(released, out); return nil }); err != nil || !reflect.DeepEqual(cpu, released) {
		t.Fatal("released hostencoder", err)
	}
	w.Encoder = savedEncoder
	opts.VulkanEncoder = nil
	if err := w.TranscribePCMWindows(ctx, reader, 1281, tok, opts, func(WindowTranscript) error { return nil }); err == nil {
		t.Fatal("poisoned CPU path should fail")
	}
	w.Encoder.Conv1Weight[0] = 0
	opts.VulkanEncoder = gpu
	stop := errors.New("callback failure")
	if err := w.TranscribePCMWindows(ctx, reader, 1281, tok, opts, func(WindowTranscript) error { return stop }); !errors.Is(err, stop) {
		t.Fatal(err)
	}
	if err := gpu.Close(); err != nil {
		t.Fatal(err)
	}
	reads := 0
	reader = sampleReadFunc(func(context.Context, []float32, int64) (int, error) { reads++; return 0, nil })
	if err := w.TranscribePCMWindows(ctx, reader, 1, tok, opts, func(WindowTranscript) error { return nil }); err == nil || reads != 0 {
		t.Fatal("closed preadmission", err, reads)
	}
	t.Log("PCM_VULKAN_BRIDGE two windows/EOT callbacks identical; poisoned/nil CPU encoder not invoked; callback error and closed-before-read pass")
}

func TestPCMVulkanDecoderOnlyValidation(t *testing.T) {
	w := contextToyModel(t)
	w.Encoder = nil
	if err := w.validatePCMModelForEncoder(true); err != nil {
		t.Fatal(err)
	}
	if err := w.validatePCMModel(); err == nil {
		t.Fatal("CPU acceptednilencoder")
	}
	bads := []func(*Whisper){
		func(w *Whisper) { w.Decoder = nil }, func(w *Whisper) { w.Config.DecoderHeads = 0 },
		func(w *Whisper) { w.Decoder.cfg.MaxLength++ }, func(w *Whisper) { w.Decoder.Layers = w.Decoder.Layers[:1] },
		func(w *Whisper) { w.Decoder.TokenEmbed = nil }, func(w *Whisper) { w.Decoder.PosEmbed = nil },
		func(w *Whisper) { w.Decoder.FinalLNWeight = nil }, func(w *Whisper) { w.Decoder.FinalLNBias = nil },
		func(w *Whisper) { w.Decoder.Layers[0].SelfQWeight = nil }, func(w *Whisper) { w.Decoder.Layers[0].CrossKWeight = nil },
		func(w *Whisper) { w.Decoder.Layers[0].CrossVBias = nil }, func(w *Whisper) { w.Decoder.Layers[0].FC2Weight = nil },
	}
	for _, mutate := range bads {
		w := contextToyModel(t)
		w.Encoder = nil
		mutate(w)
		if err := w.validatePCMModelForEncoder(true); err == nil {
			t.Fatal("invaliddecoderadmitted")
		}
	}
	var wnil *Whisper
	if err := wnil.validatePCMModelForEncoder(true); err == nil {
		t.Fatal("nilmodel")
	}
}
