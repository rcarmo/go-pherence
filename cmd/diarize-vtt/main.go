// Command diarize-vtt produces a diarized WebVTT transcript from an audio file.
//
// It is intentionally optimized for throughput over perfect word timing: audio is
// decoded to 16 kHz mono, split into bounded Whisper chunks, chunks are
// transcribed concurrently with a shared read-only model, and cues are stitched
// into a speaker-tagged VTT.
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/models/speaker"
	"github.com/rcarmo/go-pherence/models/whisper"
)

type job struct {
	idx        int
	start, end int
}

type result struct {
	idx      int
	startSec float64
	endSec   float64
	speaker  int
	text     string
	err      error
	duration time.Duration
}

func main() {
	input := flag.String("input", "", "Input audio file or URL")
	output := flag.String("output", "", "Output VTT path")
	modelPath := flag.String("model", "models/whisper-large-v3-hf/model.safetensors", "Whisper safetensors model")
	modelSize := flag.String("size", "large-v3", "Model size: tiny, base, small, medium, large-v3, turbo")
	task := flag.String("task", "translate", "Whisper task: translate or transcribe")
	language := flag.String("language", "pt", "Source language code for Whisper prompt, e.g. pt, es, en")
	workers := flag.Int("workers", max(1, min(4, runtimeWorkersDefault())), "Parallel transcription workers")
	chunkSec := flag.Float64("chunk", 25.0, "Chunk duration in seconds (<=30 recommended)")
	overlapSec := flag.Float64("overlap", 1.0, "Chunk overlap in seconds")
	maxTokens := flag.Int("max-tokens", 96, "Maximum generated tokens per chunk")
	minSpeech := flag.Float64("min-speech", 0.35, "Minimum VAD speech overlap ratio required to transcribe a chunk")
	useGPU := flag.Bool("gpu", true, "Use GPU encoder path when CUDA SGEMM is available")
	keepWav := flag.Bool("keep-wav", false, "Keep converted temporary WAV")
	flag.Parse()

	if *input == "" {
		fatalf("-input is required")
	}
	if *output == "" {
		base := strings.TrimSuffix(filepath.Base(*input), filepath.Ext(*input))
		if base == "" || strings.Contains(base, "://") {
			base = "transcript"
		}
		*output = base + ".vtt"
	}
	if *chunkSec <= 0 || *chunkSec > 30 {
		fatalf("-chunk must be >0 and <=30 seconds")
	}
	if *overlapSec < 0 || *overlapSec >= *chunkSec {
		fatalf("-overlap must be >=0 and less than -chunk")
	}
	languageToken, ok := whisper.LanguageTokens[*language]
	if !ok {
		fatalf("unknown -language %q", *language)
	}
	taskToken := whisper.TokenTranslate
	switch *task {
	case "translate":
		taskToken = whisper.TokenTranslate
	case "transcribe":
		taskToken = whisper.TokenTranscribe
	default:
		fatalf("unknown -task %q", *task)
	}

	startAll := time.Now()
	wavPath, cleanup, err := materializeWAV(*input)
	if err != nil {
		fatalf("audio decode: %v", err)
	}
	if cleanup != nil && !*keepWav {
		defer cleanup()
	}

	cfg, err := configFor(*modelSize)
	if err != nil {
		fatalf("config: %v", err)
	}
	if *maxTokens > 0 && *maxTokens < cfg.MaxDecoderLength {
		cfg.MaxDecoderLength = *maxTokens
	}

	tokPath := filepath.Join(filepath.Dir(*modelPath), "tokenizer.json")
	if err := whisper.LoadTokenizerGlobal(tokPath); err != nil {
		fatalf("tokenizer %s: %v", tokPath, err)
	}

	fmt.Fprintf(os.Stderr, "loading model %s (%s)...\n", *modelPath, *modelSize)
	enc, dec, err := whisper.LoadModel(*modelPath, cfg)
	if err != nil {
		fatalf("model: %v", err)
	}
	w := &whisper.Whisper{Encoder: enc, Decoder: dec, Config: cfg}
	var gpuEnc *whisper.GPUEncoder
	if *useGPU {
		gpuEnc = whisper.NewGPUEncoder(enc, cfg)
		fmt.Fprintf(os.Stderr, "GPU encoder enabled when CUDA SGEMM is available\n")
	}

	fmt.Fprintf(os.Stderr, "loading wav %s...\n", wavPath)
	samples, sr, err := audio.WAV(wavPath)
	if err != nil {
		fatalf("wav: %v", err)
	}
	if sr != 16000 {
		fmt.Fprintf(os.Stderr, "resampling %dHz -> 16000Hz...\n", sr)
		samples = audio.ResampleSinc(samples, sr, 16000)
	}
	audioDur := float64(len(samples)) / 16000.0
	fmt.Fprintf(os.Stderr, "audio %.1fs, chunk %.1fs overlap %.1fs, workers %d\n", audioDur, *chunkSec, *overlapSec, *workers)

	vad := speaker.EnergyVAD(samples, 16000, 25, 10, 0)
	labels := make([]int, len(vad)) // ECAPA weights are not bundled yet; single-speaker fallback.

	jobs := makeJobs(len(samples), *chunkSec, *overlapSec)
	jobs = filterSpeechJobs(jobs, vad, *minSpeech)
	fmt.Fprintf(os.Stderr, "speech chunks: %d\n", len(jobs))
	results := transcribeParallel(w, gpuEnc, samples, jobs, vad, labels, *workers, languageToken, taskToken)
	sort.Slice(results, func(i, j int) bool { return results[i].idx < results[j].idx })

	segments := make([]whisper.DiarizedSegment, 0, len(results))
	for _, r := range results {
		if r.err != nil {
			fmt.Fprintf(os.Stderr, "chunk %d failed: %v\n", r.idx, r.err)
			continue
		}
		text := strings.TrimSpace(r.text)
		if text == "" {
			continue
		}
		segments = append(segments, whisper.DiarizedSegment{Start: r.startSec, End: r.endSec, Speaker: r.speaker, Text: text})
	}
	segments = mergeDiarized(segments)

	if err := whisper.WriteDiarizedVTT(*output, segments); err != nil {
		fatalf("write vtt: %v", err)
	}
	fmt.Fprintf(os.Stderr, "wrote %s (%d cues) in %s, RTF %.2f\n", *output, len(segments), time.Since(startAll).Round(time.Second), time.Since(startAll).Seconds()/audioDur)
}

func filterSpeechJobs(jobs []job, vad []speaker.VADSegment, minRatio float64) []job {
	if minRatio <= 0 || len(vad) == 0 {
		return jobs
	}
	out := make([]job, 0, len(jobs))
	for _, j := range jobs {
		start := float64(j.start) / 16000.0
		end := float64(j.end) / 16000.0
		var speech float64
		for _, seg := range vad {
			o := minf(end, seg.End) - maxf(start, seg.Start)
			if o > 0 {
				speech += o
			}
		}
		if speech/(end-start) >= minRatio {
			out = append(out, j)
		}
	}
	return out
}

func transcribeParallel(w *whisper.Whisper, gpuEnc *whisper.GPUEncoder, samples []float32, jobs []job, vad []speaker.VADSegment, labels []int, workers int, languageToken, taskToken int) []result {
	if len(jobs) == 0 {
		return nil
	}
	if workers < 1 {
		workers = 1
	}
	if workers > len(jobs) {
		workers = len(jobs)
	}
	in := make(chan job)
	out := make(chan result, len(jobs))
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range in {
				st := time.Now()
				text, err := transcribeChunkFast(w, gpuEnc, samples[j.start:j.end], languageToken, taskToken)
				rs := float64(j.start) / 16000.0
				re := float64(j.end) / 16000.0
				out <- result{idx: j.idx, startSec: rs, endSec: re, speaker: dominantSpeaker(rs, re, vad, labels), text: text, err: err, duration: time.Since(st)}
			}
		}()
	}
	go func() {
		for _, j := range jobs {
			in <- j
		}
		close(in)
		wg.Wait()
		close(out)
	}()

	results := make([]result, 0, len(jobs))
	done := 0
	for r := range out {
		done++
		fmt.Fprintf(os.Stderr, "%3d/%3d %s-%s spk%d %q (%s)\n", done, len(jobs), vttTime(r.startSec), vttTime(r.endSec), r.speaker+1, truncate(r.text, 80), r.duration.Round(time.Second))
		results = append(results, r)
	}
	return results
}

func transcribeChunkFast(w *whisper.Whisper, gpuEnc *whisper.GPUEncoder, samples []float32, languageToken, taskToken int) (string, error) {
	cfg := w.Config
	melCfg := audio.MelConfig{SampleRate: 16000, FFTSize: 400, HopLength: 160, NumMels: cfg.NumMelBins, NFFTPadded: 512}
	mel := audio.MelSpectrogram(samples, melCfg)
	if mel == nil || len(mel) == 0 || len(mel[0]) == 0 {
		return "", nil
	}
	T := len(mel[0])
	melFlat := make([]float32, cfg.NumMelBins*T)
	for m := 0; m < cfg.NumMelBins; m++ {
		copy(melFlat[m*T:], mel[m])
	}

	var encoderOutput []float32
	if gpuEnc != nil {
		encoderOutput = gpuEnc.ForwardGPU(melFlat, T)
	} else {
		encoderOutput = w.Encoder.Forward(melFlat, T)
	}
	encLen := len(encoderOutput) / cfg.EncoderDModel
	state := whisper.NewDecoderState(cfg, encoderOutput, encLen, w.Decoder)
	tokens := whisper.GreedyDecodePrompt(w.Decoder, state, cfg, languageToken, taskToken)
	return whisper.TokensToText(tokens), nil
}

func makeJobs(n int, chunkSec, overlapSec float64) []job {
	chunk := int(chunkSec * 16000)
	step := int((chunkSec - overlapSec) * 16000)
	if step <= 0 {
		step = chunk
	}
	var jobs []job
	for off, idx := 0, 0; off < n; off, idx = off+step, idx+1 {
		end := off + chunk
		if end > n {
			end = n
		}
		if end-off < 16000 {
			break
		}
		jobs = append(jobs, job{idx: idx, start: off, end: end})
	}
	return jobs
}

func materializeWAV(input string) (string, func(), error) {
	isURL := strings.HasPrefix(input, "http://") || strings.HasPrefix(input, "https://")
	if !isURL && strings.EqualFold(filepath.Ext(input), ".wav") {
		return input, nil, nil
	}
	tmp, err := os.MkdirTemp("", "go-pherence-diarize-*")
	if err != nil {
		return "", nil, err
	}
	cleanup := func() { _ = os.RemoveAll(tmp) }
	wav := filepath.Join(tmp, "input.wav")
	args := []string{"-hide_banner", "-loglevel", "error", "-y", "-i", input, "-ac", "1", "-ar", "16000", wav}
	cmd := exec.Command("ffmpeg", args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		cleanup()
		return "", nil, err
	}
	return wav, cleanup, nil
}

func dominantSpeaker(start, end float64, vad []speaker.VADSegment, labels []int) int {
	if len(vad) == 0 || len(labels) == 0 {
		return 0
	}
	scores := map[int]float64{}
	for i, seg := range vad {
		if i >= len(labels) {
			break
		}
		o := minf(end, seg.End) - maxf(start, seg.Start)
		if o > 0 {
			scores[labels[i]] += o
		}
	}
	best, bestScore := 0, 0.0
	for spk, score := range scores {
		if score > bestScore {
			best, bestScore = spk, score
		}
	}
	return best
}

func mergeDiarized(in []whisper.DiarizedSegment) []whisper.DiarizedSegment {
	if len(in) < 2 {
		return in
	}
	out := []whisper.DiarizedSegment{in[0]}
	for _, s := range in[1:] {
		last := &out[len(out)-1]
		if s.Speaker == last.Speaker && strings.TrimSpace(s.Text) == strings.TrimSpace(last.Text) && s.Start-last.End < 0.2 {
			last.End = s.End
			continue
		}
		out = append(out, s)
	}
	return out
}

func configFor(size string) (whisper.Config, error) {
	switch size {
	case "tiny":
		return whisper.Tiny(), nil
	case "base":
		return whisper.Base(), nil
	case "small":
		return whisper.Small(), nil
	case "medium":
		return whisper.Medium(), nil
	case "large-v3":
		return whisper.LargeV3(), nil
	case "large-v3-turbo", "turbo":
		return whisper.LargeV3Turbo(), nil
	default:
		return whisper.Config{}, fmt.Errorf("unknown size %q", size)
	}
}

func runtimeWorkersDefault() int { return runtime.NumCPU() }
func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}
func truncate(s string, n int) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
func vttTime(sec float64) string {
	h := int(sec) / 3600
	m := (int(sec) % 3600) / 60
	s := int(sec) % 60
	return fmt.Sprintf("%02d:%02d:%02d", h, m, s)
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func minf(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
func maxf(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
