// Command nemotron runs the bounded native Nemotron ASR or diarization stream.
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/nemotronasr"
	"github.com/rcarmo/go-pherence/model/nemotrondiarization"
)

const chunkSamples = 16000 * 5

type options struct {
	task, backend, input, model, vocab string
	vulkanTower, ptxTower              bool
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "nemotron:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("nemotron", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var opts options
	fs.StringVar(&opts.task, "task", "", "asr or diarization")
	fs.StringVar(&opts.backend, "backend", "simd", "simd, ptx, or vulkan (GPU backends offload one projection only)")
	fs.StringVar(&opts.input, "input", "", "mono 16-kHz PCM WAV")
	fs.StringVar(&opts.model, "model", "", "matching released model.safetensors")
	fs.StringVar(&opts.vocab, "tokenizer", "", "ASR tokenizer.json (defaults beside model)")
	fs.BoolVar(&opts.vulkanTower, "vulkan-tower", false, "opt-in resident Vulkan diarization tower (hybrid; requires -backend vulkan)")
	fs.BoolVar(&opts.ptxTower, "ptx-tower", false, "opt-in resident PTX diarization tower (hybrid; requires -backend ptx)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || (opts.task != "asr" && opts.task != "diarization") || (opts.backend != "simd" && opts.backend != "ptx" && opts.backend != "vulkan") || opts.input == "" || opts.model == "" {
		return fmt.Errorf("require -task asr|diarization, -backend simd|ptx|vulkan, -input and -model, with no positional arguments")
	}
	if opts.task == "diarization" && opts.vocab != "" {
		return fmt.Errorf("-tokenizer only applies to ASR")
	}
	if opts.vulkanTower && (opts.task != "diarization" || opts.backend != "vulkan" || opts.ptxTower) {
		return fmt.Errorf("-vulkan-tower requires -task diarization -backend vulkan without -ptx-tower")
	}
	if opts.ptxTower && (opts.task != "diarization" || opts.backend != "ptx" || opts.vulkanTower) {
		return fmt.Errorf("-ptx-tower requires -task diarization -backend ptx without -vulkan-tower")
	}
	if strings.ToLower(filepath.Ext(opts.input)) != ".wav" {
		return fmt.Errorf("input must be a WAV file")
	}
	if err := requireMonoWAV(opts.input); err != nil {
		return err
	}
	pcm, rate, err := audio.WAV(opts.input)
	if err != nil {
		return err
	}
	if rate != 16000 || len(pcm) == 0 {
		return fmt.Errorf("expected nonempty mono 16-kHz WAV, got %d Hz and %d samples", rate, len(pcm))
	}
	file, err := safetensors.Open(opts.model)
	if err != nil {
		return err
	}
	// Loading owns decoded weights; the checkpoint can close before inference.
	var asr *nemotronasr.PCMGenerationModel
	var diar *nemotrondiarization.PCMStreamingRequest
	if opts.task == "asr" {
		asr, err = nemotronasr.LoadPCMGenerationModel(file)
	} else {
		diar, err = nemotrondiarization.LoadPCMStreamingRequest(file)
	}
	closeErr := file.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if opts.backend != "simd" {
		projection := "stacking"
		if opts.task == "asr" {
			projection = "subsampling"
		}
		if opts.vulkanTower {
			fmt.Fprintln(stderr, "backend=vulkan: hybrid request; stacking projection and 30 resident audio layers/final norm run on GPU; frontend, audio layer 0, head and speaker cache run on CPU")
		} else if opts.ptxTower {
			fmt.Fprintln(stderr, "backend=ptx: hybrid request; stacking projection and 30 resident audio layers/final norm run on GPU; frontend, audio layer 0, head and speaker cache run on CPU")
		} else {
			fmt.Fprintf(stderr, "backend=%s: hybrid request; only the %s projection runs on GPU; other model stages run on CPU\n", opts.backend, projection)
		}
	}
	started := time.Now()
	if opts.task == "asr" {
		err = transcribe(context.Background(), asr, opts, pcm, stdout)
	} else {
		err = diarize(context.Background(), diar, opts, pcm, stdout)
	}
	if err != nil {
		return err
	}
	fmt.Fprintf(stderr, "task=%s backend=%s request_elapsed=%s (excludes WAV/checkpoint load; includes GPU setup, transfers and teardown)\n", opts.task, opts.backend, time.Since(started))
	return nil
}

// WAV downmixes stereo. Nemotron's pinned processor contract accepts mono
// input only, so reject multichannel files before decoding them.
func requireMonoWAV(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	var header [12]byte
	if _, err := io.ReadFull(file, header[:]); err != nil {
		return err
	}
	if string(header[:4]) != "RIFF" || string(header[8:]) != "WAVE" {
		return fmt.Errorf("expected RIFF/WAVE input")
	}
	for {
		var chunk [8]byte
		if _, err := io.ReadFull(file, chunk[:]); err != nil {
			return err
		}
		size := binary.LittleEndian.Uint32(chunk[4:])
		if string(chunk[:4]) == "fmt " {
			if size < 16 {
				return fmt.Errorf("invalid WAV format chunk")
			}
			var format [4]byte
			if _, err := io.ReadFull(file, format[:]); err != nil {
				return err
			}
			if binary.LittleEndian.Uint16(format[2:]) != 1 {
				return fmt.Errorf("Nemotron requires mono WAV")
			}
			return nil
		}
		if string(chunk[:4]) == "data" {
			return fmt.Errorf("WAV data precedes format chunk")
		}
		if _, err := io.CopyN(io.Discard, file, int64(size)+int64(size%2)); err != nil {
			return err
		}
	}
}

func transcribe(ctx context.Context, model *nemotronasr.PCMGenerationModel, opts options, pcm []float32, stdout io.Writer) (err error) {
	path := opts.vocab
	if path == "" {
		path = filepath.Join(filepath.Dir(opts.model), "tokenizer.json")
	}
	vocab, err := tokenizer.Load(path)
	if err != nil {
		return err
	}
	stream := &nemotronasr.PCMGenerationStream{Model: model}
	if opts.backend != "simd" {
		projector := &nemotronasr.DeviceSubsamplingProjector{Backend: opts.backend}
		defer func() { err = errors.Join(err, projector.Close()) }()
		stream.Projector = projector
	}
	var decisions []int
	for offset := 0; offset < len(pcm); offset += chunkSamples {
		end := min(offset+chunkSamples, len(pcm))
		part, _, err := stream.AppendPCM(ctx, pcm[offset:end])
		if err != nil {
			return err
		}
		decisions = append(decisions, part...)
	}
	last, _, err := stream.Finish(ctx)
	if err != nil {
		return err
	}
	decisions = append(decisions, last...)
	text, err := nemotronasr.DecodeRNNTText(vocab, decisions)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(stdout, text)
	return err
}

func diarize(ctx context.Context, request *nemotrondiarization.PCMStreamingRequest, opts options, pcm []float32, stdout io.Writer) (err error) {
	if opts.vulkanTower {
		if err := request.EnableVulkanTower(); err != nil {
			return err
		}
		defer func() { err = errors.Join(err, request.CloseVulkanTower()) }()
	}
	if opts.ptxTower {
		if err := request.EnablePTXTower(); err != nil {
			return err
		}
		defer func() { err = errors.Join(err, request.ClosePTXTower()) }()
	}
	if opts.backend != "simd" {
		projector := &nemotrondiarization.DeviceStackingProjector{Backend: opts.backend}
		defer func() { err = errors.Join(err, projector.Close()) }()
		request.Projector = projector
	}
	var stream nemotrondiarization.SegmentStream
	segments := make([]nemotrondiarization.Segment, 0)
	consume := func(logits []float32) error {
		if len(logits) == 0 {
			return nil
		}
		part, err := stream.Append(logits)
		if err != nil {
			return err
		}
		segments = append(segments, part...)
		return nil
	}
	for offset := 0; offset < len(pcm); offset += chunkSamples {
		end := min(offset+chunkSamples, len(pcm))
		part, err := request.AppendPCMContext(ctx, pcm[offset:end])
		if err != nil {
			return err
		}
		if err := consume(part); err != nil {
			return err
		}
	}
	last, err := request.FinishContext(ctx)
	if err != nil {
		return err
	}
	if err := consume(last); err != nil {
		return err
	}
	part, err := stream.Finish()
	if err != nil {
		return err
	}
	segments = append(segments, part...)
	sortSegments(segments)
	err = json.NewEncoder(stdout).Encode(segments)
	return err
}

// Streaming closure order can differ from global start order for overlapping
// speakers. Match the processor's start-then-speaker ordering at export.
func sortSegments(segments []nemotrondiarization.Segment) {
	sort.Slice(segments, func(i, j int) bool {
		if segments[i].Start != segments[j].Start {
			return segments[i].Start < segments[j].Start
		}
		return segments[i].Speaker < segments[j].Speaker
	})
}
