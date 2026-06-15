// Command whisper transcribes audio files using the Whisper speech recognition model.
//
// Usage:
//
//	whisper -audio input.m4a [-model /path/to/model.safetensors] [-diarize]
//
// The default -size is large-v3-turbo (full large-v3 encoder + 4-layer distilled
// decoder): on-par transcript quality with much cheaper decode. Pass -size
// large-v3 for the full 32-layer decoder.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/models/speaker"
	"github.com/rcarmo/go-pherence/models/whisper"
)

func main() {
	modelPath := flag.String("model", "models/whisper-large-v3-turbo-hf/model.safetensors", "Path to Whisper safetensors model")
	audioPath := flag.String("audio", "", "Path to input audio file (WAV directly, other formats via ffmpeg)")
	modelSize := flag.String("size", "turbo", "Model size: tiny, base, small, medium, large-v3, turbo (default large-v3-turbo: same encoder as large-v3, 4-layer distilled decoder)")
	maxTokens := flag.Int("max-tokens", 0, "Maximum decoder tokens to generate (default: model config)")
	diarize := flag.Bool("diarize", false, "Enable speaker diarization")
	speakerModel := flag.String("speaker-model", "models/speaker-ecapa-voxceleb.safetensors", "Converted SpeechBrain ECAPA safetensors for diarization")
	speakerThreshold := flag.Float64("speaker-threshold", 0.3, "Cosine threshold for speaker clustering")
	chunkSec := flag.Float64("chunk", 30, "Window length in seconds for long-form chunking")
	chunkWorkers := flag.Int("chunk-workers", 1, "Parallel transcription windows (set with a small WHISPER_THREADS so workers*threads ~= cores)")
	timestamps := flag.Bool("timestamps", false, "Output with timestamps")
	output := flag.String("output", "", "Output file path (supports .vtt)")
	flag.Parse()

	if *audioPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: whisper -audio AUDIO [-model MODEL] [-size SIZE] [-diarize] [-timestamps]\n")
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
	wavPath, cleanup, err := materializeWAV(*audioPath)
	if cleanup != nil {
		defer cleanup()
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error decoding audio: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Loading audio from %s...\n", wavPath)
	samples, sampleRate, err := audio.WAV(wavPath)
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
		// Transcribe with timestamps (chunked + parallel for long audio).
		segments := transcribeChunked(w, samples, *chunkSec, *chunkWorkers)
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

func materializeWAV(input string) (string, func(), error) {
	isURL := strings.HasPrefix(input, "http://") || strings.HasPrefix(input, "https://")
	if !isURL && strings.EqualFold(filepath.Ext(input), ".wav") {
		return input, nil, nil
	}
	tmp, err := os.MkdirTemp("", "go-pherence-whisper-*")
	if err != nil {
		return "", nil, err
	}
	cleanup := func() { _ = os.RemoveAll(tmp) }
	wav := filepath.Join(tmp, "input.wav")
	cmd := exec.Command("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", input, "-ac", "1", "-ar", "16000", wav)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		cleanup()
		return "", nil, err
	}
	return wav, cleanup, nil
}

func transcribeWithTimestamps(w *whisper.Whisper, samples []float32) []whisper.Segment {
	return transcribeWindow(w, samples, 0)
}

// transcribeWindow runs one <=30s window (mel -> encoder -> greedy decode) and
// offsets the resulting segment timestamps by offsetSec. Encoder/decoder state
// is per-call, so multiple windows can run concurrently.
func transcribeWindow(w *whisper.Whisper, samples []float32, offsetSec float64) []whisper.Segment {
	melCfg := audio.MelConfig{
		SampleRate: 16000,
		FFTSize:    400,
		HopLength:  160,
		NumMels:    w.Config.NumMelBins,
		NFFTPadded: 512,
	}
	mel := audio.MelSpectrogram(samples, melCfg)
	if mel == nil || len(mel) == 0 || len(mel[0]) == 0 {
		return nil
	}

	T := len(mel[0])
	melFlat := make([]float32, w.Config.NumMelBins*T)
	for m := 0; m < w.Config.NumMelBins; m++ {
		copy(melFlat[m*T:], mel[m])
	}

	var encoderOutput []float32
	if hp := os.Getenv("WHISPER_ENC_H"); hp != "" {
		// Inject a precomputed encoder hidden state (e.g. from the NPU encoder),
		// raw float32 [seq, EncoderDModel] row-major. Skips the CPU encoder.
		raw, err := os.ReadFile(hp)
		if err != nil {
			fmt.Fprintf(os.Stderr, "WHISPER_ENC_H read: %v\n", err)
			return nil
		}
		encoderOutput = make([]float32, len(raw)/4)
		for i := range encoderOutput {
			encoderOutput[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		if os.Getenv("WHISPER_DEBUG") != "" {
			fmt.Fprintf(os.Stderr, "[enc] injected %d floats from %s (skipping CPU encoder)\n", len(encoderOutput), hp)
		}
	} else {
		encoderOutput = w.Encoder.Forward(melFlat, T)
	}
	encLen := len(encoderOutput) / w.Config.EncoderDModel
	dt0 := time.Now()
	state := whisper.NewDecoderState(w.Config, encoderOutput, encLen, w.Decoder)

	segs := whisper.GreedyDecodeWithTimestamps(w.Decoder, state, w.Config)
	windowDur := float64(len(samples)) / 16000.0
	for i := range segs {
		if segs[i].End > windowDur {
			segs[i].End = windowDur
		}
		if segs[i].Start > segs[i].End {
			segs[i].Start = segs[i].End
		}
	}
	if os.Getenv("WHISPER_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "[timing] xkv+decode=%.2fs\n", time.Since(dt0).Seconds())
	}
	if offsetSec != 0 {
		for i := range segs {
			segs[i].Start += offsetSec
			segs[i].End += offsetSec
		}
	}
	return segs
}

// transcribeChunked splits long audio into fixed windows and transcribes them
// across `workers` goroutines. Whisper's encoder is a fixed 30s window, so
// long-form throughput comes from running independent windows concurrently;
// pair with a small per-window WHISPER_THREADS so workers*threads ~= cores.
func transcribeChunked(w *whisper.Whisper, samples []float32, chunkSec float64, workers int) []whisper.Segment {
	const sr = 16000
	chunk := int(chunkSec * sr)
	if chunk <= 0 || len(samples) <= chunk || workers <= 1 {
		if chunk > 0 && len(samples) > chunk {
			// Sequential chunking (workers<=1) still avoids the O(T^2) full pass.
			var all []whisper.Segment
			for s := 0; s < len(samples); s += chunk {
				e := s + chunk
				if e > len(samples) {
					e = len(samples)
				}
				all = append(all, transcribeWindow(w, samples[s:e], float64(s)/sr)...)
			}
			return all
		}
		return transcribeWindow(w, samples, 0)
	}

	type span struct{ s, e int }
	var spans []span
	for s := 0; s < len(samples); s += chunk {
		e := s + chunk
		if e > len(samples) {
			e = len(samples)
		}
		spans = append(spans, span{s, e})
	}
	results := make([][]whisper.Segment, len(spans))
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup
	for i, sp := range spans {
		wg.Add(1)
		sem <- struct{}{}
		go func(i, s, e int) {
			defer wg.Done()
			defer func() { <-sem }()
			results[i] = transcribeWindow(w, samples[s:e], float64(s)/sr)
		}(i, sp.s, sp.e)
	}
	wg.Wait()
	var all []whisper.Segment
	for _, r := range results {
		all = append(all, r...)
	}
	return all
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
