// Command whisper transcribes audio files using the Whisper speech recognition model.
//
// Usage:
//
//	whisper -model /path/to/model.safetensors -audio input.wav [-diarize]
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/models/speaker"
	"github.com/rcarmo/go-pherence/models/whisper"
)

func main() {
	modelPath := flag.String("model", "", "Path to Whisper safetensors model")
	audioPath := flag.String("audio", "", "Path to input WAV file")
	modelSize := flag.String("size", "tiny", "Model size: tiny, base, small, medium, large-v3")
	diarize := flag.Bool("diarize", false, "Enable speaker diarization")
	timestamps := flag.Bool("timestamps", false, "Output with timestamps")
	flag.Parse()

	if *modelPath == "" || *audioPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: whisper -model MODEL -audio AUDIO [-size SIZE] [-diarize] [-timestamps]\n")
		os.Exit(1)
	}

	// Select config
	var cfg whisper.Config
	switch *modelSize {
	case "tiny":
		cfg = whisper.Tiny()
	case "base":
		cfg = whisper.Base()
	case "small":
		cfg = whisper.Small()
	case "medium":
		cfg = whisper.Medium()
	case "large-v3":
		cfg = whisper.LargeV3()
	default:
		fmt.Fprintf(os.Stderr, "Unknown model size: %s\n", *modelSize)
		os.Exit(1)
	}

	// Load model
	fmt.Fprintf(os.Stderr, "Loading model from %s (%s)...\n", *modelPath, *modelSize)
	enc, dec, err := whisper.LoadModel(*modelPath, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}

	w := &whisper.Whisper{
		Encoder: enc,
		Decoder: dec,
		Config:  cfg,
	}

	// Load audio
	fmt.Fprintf(os.Stderr, "Loading audio from %s...\n", *audioPath)
	samples, sampleRate, err := audio.WAV(*audioPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading audio: %v\n", err)
		os.Exit(1)
	}

	// Resample to 16kHz
	if sampleRate != 16000 {
		fmt.Fprintf(os.Stderr, "Resampling from %dHz to 16000Hz...\n", sampleRate)
		samples = audio.ResampleSinc(samples, sampleRate, 16000)
	}

	fmt.Fprintf(os.Stderr, "Audio: %.1fs\n", float64(len(samples))/16000)

	if *timestamps {
		// Transcribe with timestamps
		segments := transcribeWithTimestamps(w, samples)
		if *diarize {
			diarized := diarizeSegments(samples, segments)
			for _, seg := range diarized {
				fmt.Printf("[%.2f - %.2f] Speaker %d: %s\n", seg.Start, seg.End, seg.Speaker, seg.Text)
			}
		} else {
			for _, seg := range segments {
				fmt.Printf("[%.2f - %.2f] %s\n", seg.Start, seg.End, seg.Text)
			}
		}
	} else {
		// Simple transcription
		text, err := w.TranscribeFromSamples(samples)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error transcribing: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(text)
	}
}

func transcribeWithTimestamps(w *whisper.Whisper, samples []float32) []whisper.Segment {
	melCfg := audio.MelConfig{
		SampleRate: 16000,
		FFTSize:    400,
		HopLength:  160,
		NumMels:    w.Config.NumMelBins,
		NFFTPadded: 512,
	}
	mel := audio.MelSpectrogram(samples, melCfg)
	if mel == nil {
		return nil
	}

	T := len(mel[0])
	melFlat := make([]float32, w.Config.NumMelBins*T)
	for m := 0; m < w.Config.NumMelBins; m++ {
		copy(melFlat[m*T:], mel[m])
	}

	encoderOutput := w.Encoder.Forward(melFlat, T)
	encLen := len(encoderOutput) / w.Config.EncoderDModel
	state := whisper.NewDecoderState(w.Config, encoderOutput, encLen, w.Decoder)

	return whisper.GreedyDecodeWithTimestamps(w.Decoder, state, w.Config)
}

func diarizeSegments(samples []float32, segments []whisper.Segment) []speaker.DiarizedSegment {
	// VAD to find speech segments
	vadSegs := speaker.EnergyVAD(samples, 16000, 25, 10, 0)
	if len(vadSegs) == 0 {
		// Fallback: one segment per whisper segment
		result := make([]speaker.DiarizedSegment, len(segments))
		for i, s := range segments {
			result[i] = speaker.DiarizedSegment{Start: s.Start, End: s.End, Text: s.Text, Speaker: 0}
		}
		return result
	}

	// Extract speaker embeddings per VAD segment (placeholder: zero embeddings)
	// Real implementation would use ECAPA-TDNN model
	embeddings := make([][]float32, len(vadSegs))
	for i := range embeddings {
		embeddings[i] = make([]float32, 192)
	}

	// Cluster
	labels := speaker.AgglomerativeCluster(embeddings, 0.7)

	// Align
	return speaker.AlignSpeakers(segments, vadSegs, labels)
}
