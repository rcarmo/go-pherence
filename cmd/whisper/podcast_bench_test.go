package main

import (
	"os"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/models/speaker"
	"github.com/rcarmo/go-pherence/models/whisper"
)

func TestPodcastDiarizeVTT(t *testing.T) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("model not available")
	}
	audioPath := "../../testdata/podcast.wav"
	if _, err := os.Stat(audioPath); err != nil {
		t.Skip("podcast.wav not available")
	}

	cfg := whisper.Tiny()
	enc, dec, err := whisper.LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	samples, sampleRate, err := audio.WAV(audioPath)
	if err != nil {
		t.Fatalf("WAV: %v", err)
	}
	if sampleRate != 16000 {
		samples = audio.ResampleSinc(samples, sampleRate, 16000)
	}
	audioDur := float64(len(samples)) / 16000
	t.Logf("Audio: %.1fs (%d samples)", audioDur, len(samples))

	// Process first 30s only for speed
	if len(samples) > 30*16000 {
		samples = samples[:30*16000]
		audioDur = 30.0
	}

	w := &whisper.Whisper{Encoder: enc, Decoder: dec, Config: cfg}

	start := time.Now()
	segs, err := w.ChunkedTranscribe(samples, 1.0)
	if err != nil {
		t.Fatalf("ChunkedTranscribe: %v", err)
	}
	transcribeTime := time.Since(start)
	t.Logf("Transcription: %d segments in %v (RTF=%.2f)", len(segs), transcribeTime, transcribeTime.Seconds()/audioDur)

	// VAD + diarize
	vadSegs := speaker.EnergyVAD(samples, 16000, 25, 10, 0)
	t.Logf("VAD: %d segments", len(vadSegs))

	diarized := make([]whisper.DiarizedSegment, len(segs))
	for i, seg := range segs {
		spk := 0
		for j, vs := range vadSegs {
			if seg.Start >= vs.Start-0.1 && seg.Start <= vs.End+0.1 {
				spk = j % 2
				break
			}
		}
		diarized[i] = whisper.DiarizedSegment{Start: seg.Start, End: seg.End, Speaker: spk, Text: seg.Text}
	}

	outPath := "../../testdata/podcast_diarized.vtt"
	if err := whisper.WriteDiarizedVTT(outPath, diarized); err != nil {
		t.Fatalf("WriteDiarizedVTT: %v", err)
	}
	t.Logf("VTT written to %s", outPath)

	plainPath := "../../testdata/podcast.vtt"
	if err := whisper.WriteVTT(plainPath, segs); err != nil {
		t.Fatalf("WriteVTT: %v", err)
	}
	t.Logf("Plain VTT written to %s", plainPath)
}
