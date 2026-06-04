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
	"path/filepath"
	"strconv"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/models/speaker"
	"github.com/rcarmo/go-pherence/models/whisper"
)

func main() {
	modelPath := flag.String("model", "", "Path to Whisper safetensors model")
	audioPath := flag.String("audio", "", "Path to input WAV file")
	modelSize := flag.String("size", "tiny", "Model size: tiny, base, small, medium, large-v3, turbo")
	maxTokens := flag.Int("max-tokens", 0, "Maximum decoder tokens to generate (default: model config)")
	diarize := flag.Bool("diarize", false, "Enable speaker diarization")
	speakerModel := flag.String("speaker-model", "models/speaker-ecapa-voxceleb.safetensors", "Converted SpeechBrain ECAPA safetensors for diarization")
	speakerThreshold := flag.Float64("speaker-threshold", 0.3, "Cosine threshold for speaker clustering")
	timestamps := flag.Bool("timestamps", false, "Output with timestamps")
	output := flag.String("output", "", "Output file path (supports .vtt)")
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
	case "large-v3-turbo", "turbo":
		cfg = whisper.LargeV3Turbo()
	default:
		fmt.Fprintf(os.Stderr, "Unknown model size: %s\n", *modelSize)
		os.Exit(1)
	}

	if *maxTokens > 0 && *maxTokens < cfg.MaxDecoderLength {
		cfg.MaxDecoderLength = *maxTokens
	}

	// Load tokenizer next to the model when available; without this, transcripts
	// degrade to placeholder tokens.
	tokenizerPath := filepath.Join(filepath.Dir(*modelPath), "tokenizer.json")
	if _, err := os.Stat(tokenizerPath); err == nil {
		if err := whisper.LoadTokenizerGlobal(tokenizerPath); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not load tokenizer %s: %v\n", tokenizerPath, err)
		}
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
			diarized := diarizeSegments(samples, segments, *speakerModel, float32(*speakerThreshold))
			if *output != "" {
				if err := whisper.WriteDiarizedVTT(*output, diarized); err != nil {
					fmt.Fprintf(os.Stderr, "Error writing VTT: %v\n", err)
					os.Exit(1)
				}
				fmt.Fprintf(os.Stderr, "Written to %s\n", *output)
			} else {
				for _, seg := range diarized {
					fmt.Printf("[%.2f - %.2f] Speaker %d: %s\n", seg.Start, seg.End, seg.Speaker+1, seg.Text)
				}
			}
		} else {
			if *output != "" {
				if err := whisper.WriteVTT(*output, segments); err != nil {
					fmt.Fprintf(os.Stderr, "Error writing VTT: %v\n", err)
					os.Exit(1)
				}
				fmt.Fprintf(os.Stderr, "Written to %s\n", *output)
			} else {
				for _, seg := range segments {
					fmt.Printf("[%.2f - %.2f] %s\n", seg.Start, seg.End, seg.Text)
				}
			}
		}
	} else {
		// Simple transcription. WHISPER_REPEAT>1 runs extra warm passes (weights
		// stay quantized/packed in cache) to measure steady-state resident cost.
		reps := 1
		if rv := os.Getenv("WHISPER_REPEAT"); rv != "" {
			if n, e := strconv.Atoi(rv); e == nil && n > 0 {
				reps = n
			}
		}
		var text string
		var err error
		for r := 0; r < reps; r++ {
			ps := time.Now()
			text, err = w.TranscribeFromSamples(samples)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error transcribing: %v\n", err)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "[pass %d] %.1fs\n", r, time.Since(ps).Seconds())
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

func diarizeSegments(samples []float32, segments []whisper.Segment, modelPath string, threshold float32) []whisper.DiarizedSegment {
	singleSpeaker := func() []whisper.DiarizedSegment {
		result := make([]whisper.DiarizedSegment, len(segments))
		for i, s := range segments {
			result[i] = whisper.DiarizedSegment{Start: s.Start, End: s.End, Text: s.Text, Speaker: 0}
		}
		return result
	}

	// VAD to find speech segments.
	vadSegs := speaker.EnergyVAD(samples, 16000, 25, 10, 0)
	if len(vadSegs) == 0 {
		return singleSpeaker()
	}

	// Real ECAPA-TDNN speaker embeddings. Fall back to a single speaker if the
	// converted model is unavailable so transcription still succeeds.
	model, err := speaker.LoadSpeechBrainECAPASafetensors(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "diarize: speaker model unavailable (%v); using single-speaker fallback\n", err)
		return singleSpeaker()
	}
	embeddings := speaker.ExtractSpeechBrainEmbeddingsWithContext(samples, 16000, vadSegs, model, 0.5)

	// Cluster, then smooth singleton labels (mirrors cmd/speakercheck).
	labels := speaker.AgglomerativeCluster(embeddings, threshold)
	labels = speaker.SmoothSingletonLabels(labels, embeddings, 0.4)

	// Align and convert.
	aligned := speaker.AlignSpeakers(segments, vadSegs, labels)
	result := make([]whisper.DiarizedSegment, len(aligned))
	for i, a := range aligned {
		result[i] = whisper.DiarizedSegment{Start: a.Start, End: a.End, Speaker: a.Speaker, Text: a.Text}
	}
	return result
}
