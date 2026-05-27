// Command diarize-vtt produces a diarized WebVTT transcript from an audio file.
//
// It is intentionally optimized for throughput over perfect word timing: audio is
// decoded to 16 kHz mono, split into bounded Whisper chunks, chunks are
// transcribed concurrently with a shared read-only model, and cues are stitched
// into a speaker-tagged VTT.
package main

import (
	"bufio"
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
	idx              int
	start, end       int
	cueStart, cueEnd int
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
	workers := flag.Int("workers", max(1, min(16, runtimeWorkersDefault())), "Parallel transcription workers")
	chunkSec := flag.Float64("chunk", 10.0, "Maximum chunk duration in seconds (<=30 recommended)")
	overlapSec := flag.Float64("overlap", 1.0, "Chunk overlap/context padding in seconds")
	vadPack := flag.Bool("vad-pack", true, "Pack VAD speech regions into chunks instead of fixed windows")
	vadGap := flag.Float64("vad-gap", 1.0, "Maximum silence gap in seconds to merge adjacent VAD regions")
	maxTokens := flag.Int("max-tokens", 40, "Maximum generated tokens per chunk")
	tokensPerSec := flag.Float64("tokens-per-sec", 4.0, "Dynamic decoder token budget per second of cue audio")
	minSpeech := flag.Float64("min-speech", 0.35, "Minimum VAD speech overlap ratio required to transcribe a chunk")
	useGPU := flag.Bool("gpu", true, "Use GPU encoder path when CUDA SGEMM is available")
	progressive := flag.Bool("progressive", true, "Write VTT after each completed chunk")
	resume := flag.Bool("resume", true, "Resume from existing output VTT by skipping completed cue intervals")
	speakerModel := flag.String("speaker-model", "", "Optional converted ECAPA safetensors speaker embedding model")
	speakerThreshold := flag.Float64("speaker-threshold", 0.3, "Cosine similarity threshold for speaker clustering")
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
	labels := speakerLabels(samples, vad, *speakerModel, float32(*speakerThreshold))

	jobs := makeJobs(len(samples), *chunkSec, *overlapSec)
	if *vadPack {
		jobs = makeVADPackedJobs(len(samples), vad, *chunkSec, *overlapSec, *vadGap)
	} else {
		jobs = filterSpeechJobs(jobs, vad, *minSpeech)
	}
	initialResults := []result(nil)
	if *resume {
		completed, existing, err := loadCompletedResults(*output)
		if err != nil {
			fmt.Fprintf(os.Stderr, "resume disabled: %v\n", err)
		} else if len(completed) > 0 {
			initialResults = existing
			jobs = filterCompletedJobs(jobs, completed)
			fmt.Fprintf(os.Stderr, "resume: loaded %d existing cues, remaining chunks: %d\n", len(existing), len(jobs))
			if *progressive {
				// Rewrite immediately so stale/degenerate cues from older partial runs
				// are cleaned even if this resume pass is interrupted before another
				// chunk completes.
				if err := whisper.WriteDiarizedVTT(*output, segmentsFromResults(initialResults)); err != nil {
					fmt.Fprintf(os.Stderr, "resume rewrite failed: %v\n", err)
				}
			}
		}
	}
	fmt.Fprintf(os.Stderr, "speech chunks: %d\n", len(jobs))
	results := transcribeParallel(w, gpuEnc, samples, jobs, vad, labels, *workers, languageToken, taskToken, *tokensPerSec, *output, *progressive, initialResults)
	sort.Slice(results, func(i, j int) bool { return results[i].idx < results[j].idx })

	segments := segmentsFromResults(results)

	if err := whisper.WriteDiarizedVTT(*output, segments); err != nil {
		fatalf("write vtt: %v", err)
	}
	fmt.Fprintf(os.Stderr, "wrote %s (%d cues) in %s, RTF %.2f\n", *output, len(segments), time.Since(startAll).Round(time.Second), time.Since(startAll).Seconds()/audioDur)
}

func speakerLabels(samples []float32, vad []speaker.VADSegment, modelPath string, threshold float32) []int {
	labels := make([]int, len(vad))
	if len(vad) == 0 || modelPath == "" {
		if modelPath == "" {
			fmt.Fprintf(os.Stderr, "speaker model not set; using single-speaker fallback\n")
		}
		return labels
	}
	embeddings, err := speakerEmbeddings(samples, vad, modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "speaker model disabled: %v\n", err)
		return labels
	}
	if len(embeddings) != len(vad) {
		fmt.Fprintf(os.Stderr, "speaker model disabled: got %d embeddings for %d VAD segments\n", len(embeddings), len(vad))
		return labels
	}
	if threshold <= 0 {
		threshold = 0.3
	}
	labels = speaker.AgglomerativeCluster(embeddings, threshold)
	maxLabel := 0
	for _, label := range labels {
		if label > maxLabel {
			maxLabel = label
		}
	}
	fmt.Fprintf(os.Stderr, "speaker model %s: clustered %d VAD segments into %d speakers\n", modelPath, len(vad), maxLabel+1)
	return labels
}

func speakerEmbeddings(samples []float32, vad []speaker.VADSegment, modelPath string) ([][]float32, error) {
	if sb, err := speaker.LoadSpeechBrainECAPASafetensors(modelPath); err == nil {
		return speaker.ExtractSpeechBrainEmbeddings(samples, 16000, vad, sb), nil
	}
	cfg := speaker.DefaultECAPAConfig()
	ecapa, err := speaker.LoadECAPASafetensors(modelPath, cfg)
	if err != nil {
		return nil, err
	}
	return speaker.ExtractEmbeddings(samples, 16000, vad, ecapa), nil
}

func makeVADPackedJobs(totalSamples int, vad []speaker.VADSegment, maxChunkSec, padSec, mergeGapSec float64) []job {
	if totalSamples <= 0 || len(vad) == 0 || maxChunkSec <= 0 {
		return makeJobs(totalSamples, maxChunkSec, padSec)
	}
	maxSamples := int(maxChunkSec * 16000)
	padSamples := int(padSec * 16000)
	mergeGapSamples := int(mergeGapSec * 16000)
	if maxSamples <= 0 {
		return nil
	}

	type region struct{ start, end int }
	regions := make([]region, 0, len(vad))
	for _, seg := range vad {
		start := int(seg.Start * 16000)
		end := int(seg.End * 16000)
		if start < 0 {
			start = 0
		}
		if end > totalSamples {
			end = totalSamples
		}
		if end <= start {
			continue
		}
		if len(regions) > 0 && start-regions[len(regions)-1].end <= mergeGapSamples {
			if end > regions[len(regions)-1].end {
				regions[len(regions)-1].end = end
			}
			continue
		}
		regions = append(regions, region{start: start, end: end})
	}

	var jobs []job
	idx := 0
	for i := 0; i < len(regions); {
		cueStart := regions[i].start
		cueEnd := regions[i].end
		j := i + 1
		for j < len(regions) && regions[j].end-cueStart <= maxSamples {
			cueEnd = regions[j].end
			j++
		}
		// If one VAD region exceeds max duration, split it into max-sized jobs.
		if cueEnd-cueStart > maxSamples {
			for off := cueStart; off < cueEnd; off += maxSamples {
				end := off + maxSamples
				if end > cueEnd {
					end = cueEnd
				}
				jobs = appendVADJob(jobs, idx, off, end, totalSamples, padSamples)
				idx++
			}
			i++
			continue
		}
		jobs = appendVADJob(jobs, idx, cueStart, cueEnd, totalSamples, padSamples)
		idx++
		i = j
	}
	return jobs
}

func appendVADJob(jobs []job, idx, cueStart, cueEnd, totalSamples, padSamples int) []job {
	start := cueStart - padSamples
	if start < 0 {
		start = 0
	}
	end := cueEnd + padSamples
	if end > totalSamples {
		end = totalSamples
	}
	if end-start < 16000 {
		// Keep very short speech regions decodable with at least one second of context.
		need := 16000 - (end - start)
		start -= need / 2
		end += need - need/2
		if start < 0 {
			end -= start
			start = 0
		}
		if end > totalSamples {
			start -= end - totalSamples
			end = totalSamples
			if start < 0 {
				start = 0
			}
		}
	}
	if end <= start || cueEnd <= cueStart {
		return jobs
	}
	return append(jobs, job{idx: idx, start: start, end: end, cueStart: cueStart, cueEnd: cueEnd})
}

type cueKey struct{ startMS, endMS int }

func filterCompletedJobs(jobs []job, completed map[cueKey]bool) []job {
	if len(completed) == 0 {
		return jobs
	}
	out := make([]job, 0, len(jobs))
	for _, j := range jobs {
		startMS, endMS := sampleMS(j.cueStart), sampleMS(j.cueEnd)
		if !coveredByCompletedCue(startMS, endMS, completed) {
			out = append(out, j)
		}
	}
	return out
}

func coveredByCompletedCue(startMS, endMS int, completed map[cueKey]bool) bool {
	if endMS <= startMS {
		return false
	}
	jobDur := endMS - startMS
	for c := range completed {
		overlap := min(endMS, c.endMS) - max(startMS, c.startMS)
		if overlap <= 0 {
			continue
		}
		// Resume from older partial VTTs whose cue boundaries may not match the
		// current VAD-packed chunking exactly. Treat a job as complete when an
		// existing cue covers most of its interval.
		if overlap*100 >= jobDur*80 {
			return true
		}
	}
	return false
}

func loadCompletedResults(path string) (map[cueKey]bool, []result, error) {
	if path == "" {
		return nil, nil, fmt.Errorf("empty output path")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	completed := map[cueKey]bool{}
	var results []result
	scanner := bufio.NewScanner(f)
	idx := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.Contains(line, "-->") {
			continue
		}
		parts := strings.Split(line, "-->")
		if len(parts) != 2 {
			continue
		}
		startMS, ok1 := parseVTTMillis(strings.TrimSpace(parts[0]))
		endMS, ok2 := parseVTTMillis(strings.TrimSpace(parts[1]))
		if !ok1 || !ok2 || endMS <= startMS {
			continue
		}
		text := ""
		if scanner.Scan() {
			text = strings.TrimSpace(scanner.Text())
			text = strings.TrimPrefix(text, "<v Speaker 1>")
		}
		key := cueKey{startMS: startMS, endMS: endMS}
		completed[key] = true
		results = append(results, result{idx: idx, startSec: float64(startMS) / 1000, endSec: float64(endMS) / 1000, speaker: 0, text: text})
		idx++
	}
	if err := scanner.Err(); err != nil {
		return nil, nil, err
	}
	return completed, results, nil
}

func parseVTTMillis(s string) (int, bool) {
	var h, m, sec, ms int
	if _, err := fmt.Sscanf(s, "%d:%d:%d.%d", &h, &m, &sec, &ms); err != nil {
		return 0, false
	}
	return ((h*60+m)*60+sec)*1000 + ms, true
}

func sampleMS(sample int) int { return int(float64(sample) * 1000 / 16000) }

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

func transcribeParallel(w *whisper.Whisper, gpuEnc *whisper.GPUEncoder, samples []float32, jobs []job, vad []speaker.VADSegment, labels []int, workers int, languageToken, taskToken int, tokensPerSec float64, outputPath string, progressive bool, initial []result) []result {
	if len(jobs) == 0 {
		return initial
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
				rs := float64(j.cueStart) / 16000.0
				re := float64(j.cueEnd) / 16000.0
				text, err := transcribeChunkFast(w, gpuEnc, samples[j.start:j.end], languageToken, taskToken, dynamicMaxTokens(w.Config.MaxDecoderLength, re-rs, tokensPerSec))
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

	results := append([]result(nil), initial...)
	done := 0
	for r := range out {
		done++
		fmt.Fprintf(os.Stderr, "%3d/%3d %s-%s spk%d %q (%s)\n", done, len(jobs), vttTime(r.startSec), vttTime(r.endSec), r.speaker+1, truncate(r.text, 80), r.duration.Round(time.Second))
		results = append(results, r)
		if progressive && outputPath != "" {
			ordered := append([]result(nil), results...)
			sort.Slice(ordered, func(i, j int) bool { return ordered[i].idx < ordered[j].idx })
			if err := whisper.WriteDiarizedVTT(outputPath, segmentsFromResults(ordered)); err != nil {
				fmt.Fprintf(os.Stderr, "progressive write failed: %v\n", err)
			}
		}
	}
	return results
}

func segmentsFromResults(results []result) []whisper.DiarizedSegment {
	ordered := append([]result(nil), results...)
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].startSec == ordered[j].startSec {
			return ordered[i].endSec < ordered[j].endSec
		}
		return ordered[i].startSec < ordered[j].startSec
	})
	segments := make([]whisper.DiarizedSegment, 0, len(ordered))
	for _, r := range ordered {
		if r.err != nil {
			fmt.Fprintf(os.Stderr, "chunk %d failed: %v\n", r.idx, r.err)
			continue
		}
		text := strings.TrimSpace(r.text)
		if text == "" || degenerateCueText(text) {
			continue
		}
		segments = append(segments, whisper.DiarizedSegment{Start: r.startSec, End: r.endSec, Speaker: r.speaker, Text: text})
	}
	return mergeDiarized(segments)
}

func degenerateCueText(text string) bool {
	words := strings.Fields(strings.ToLower(text))
	if len(words) <= 2 {
		return lowValueShortCue(words)
	}
	if len(words) < 8 {
		return false
	}
	for n := 3; n <= 5; n++ {
		if hasRepeatedWordNGram(words, n) {
			return true
		}
	}
	return false
}

func lowValueShortCue(words []string) bool {
	if len(words) == 0 {
		return true
	}
	filler := map[string]bool{"and": true, "or": true, "the": true, "a": true, "um": true, "uh": true, "eh": true, "ah": true}
	if len(words) == 1 {
		return filler[strings.Trim(words[0], ".,!?;:")]
	}
	return filler[strings.Trim(words[0], ".,!?;:")] && filler[strings.Trim(words[1], ".,!?;:")]
}

func hasRepeatedWordNGram(words []string, n int) bool {
	if n <= 0 || len(words) < 2*n {
		return false
	}
	seen := map[string]bool{}
	for i := 0; i+n <= len(words); i++ {
		key := strings.Join(words[i:i+n], " ")
		if seen[key] {
			return true
		}
		seen[key] = true
	}
	return false
}

func transcribeChunkFast(w *whisper.Whisper, gpuEnc *whisper.GPUEncoder, samples []float32, languageToken, taskToken, maxTokens int) (string, error) {
	cfg := w.Config
	if maxTokens > 0 && maxTokens < cfg.MaxDecoderLength {
		cfg.MaxDecoderLength = maxTokens
	}
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
	var state *whisper.DecoderState
	if gpuEnc != nil {
		encoderOutput = gpuEnc.ForwardGPU(melFlat, T)
		encLen := len(encoderOutput) / cfg.EncoderDModel
		state = whisper.NewDecoderStateGPU(cfg, encoderOutput, encLen, w.Decoder)
	} else {
		encoderOutput = w.Encoder.Forward(melFlat, T)
		encLen := len(encoderOutput) / cfg.EncoderDModel
		state = whisper.NewDecoderState(cfg, encoderOutput, encLen, w.Decoder)
	}
	tokens := whisper.GreedyDecodePrompt(w.Decoder, state, cfg, languageToken, taskToken)
	return whisper.TokensToText(tokens), nil
}

func dynamicMaxTokens(defaultMax int, cueSeconds, tokensPerSec float64) int {
	if defaultMax <= 0 || cueSeconds <= 0 || tokensPerSec <= 0 {
		return defaultMax
	}
	budget := int(cueSeconds*tokensPerSec) + 8
	if budget < 12 {
		budget = 12
	}
	if budget > defaultMax {
		budget = defaultMax
	}
	return budget
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
		cueStart := off
		cueEnd := end
		if idx > 0 {
			cueStart = off + int(overlapSec*16000/2)
		}
		if end < n {
			cueEnd = end - int(overlapSec*16000/2)
		}
		if cueStart < off {
			cueStart = off
		}
		if cueEnd > end {
			cueEnd = end
		}
		if cueEnd <= cueStart {
			cueStart, cueEnd = off, end
		}
		jobs = append(jobs, job{idx: idx, start: off, end: end, cueStart: cueStart, cueEnd: cueEnd})
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
