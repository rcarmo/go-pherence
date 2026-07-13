// Command moss-transcribe runs MOSS-Transcribe-Diarize entirely in native Go.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/model/mosstranscribe"
)

type options struct {
	modelDir     string
	audioPath    string
	outputPath   string
	format       string
	prompt       string
	maxNew       int
	cpuOnly      bool
	capabilities bool
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "moss-transcribe:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("moss-transcribe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var opts options
	fs.StringVar(&opts.modelDir, "model-dir", "", "Hugging Face MOSS model directory")
	fs.StringVar(&opts.audioPath, "audio", "", "Input mono/stereo PCM WAV (resampled natively to 16 kHz)")
	fs.StringVar(&opts.outputPath, "output", "", "Output path (default stdout)")
	fs.StringVar(&opts.format, "format", "text", "Output format: text, raw, json, srt, ass")
	fs.StringVar(&opts.prompt, "prompt", "", "Instruction override (default pinned transcription/diarization prompt)")
	fs.IntVar(&opts.maxNew, "max-new-tokens", mosstranscribe.GenerationMaxNewTokens, "Greedy decoder token limit")
	fs.BoolVar(&opts.cpuOnly, "cpu", false, "Disable automatic NVIDIA GPU acceleration")
	fs.BoolVar(&opts.capabilities, "capabilities", false, "Print native CPU/SIMD/GPU capabilities and exit")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", fs.Args())
	}
	if opts.capabilities {
		fmt.Fprintf(stdout, "goarch=%s avx2_fma=%v nvidia_ptx=%v gpu_auto=true native=true sampling=false formats=text,raw,json,srt,ass\n", runtime.GOARCH, simd.HasVecAsm, nvidia.Available())
		return nil
	}
	if opts.modelDir == "" || opts.audioPath == "" {
		return errors.New("-model-dir and -audio are required")
	}
	if err := validateFormat(opts.format); err != nil {
		return err
	}
	if ext := strings.ToLower(filepath.Ext(opts.audioPath)); ext != ".wav" {
		return fmt.Errorf("unsupported audio %q: native CLI currently accepts PCM WAV only", ext)
	}
	if err := mosstranscribe.CheckModelDirectory(opts.modelDir); err != nil {
		return err
	}

	started := time.Now()
	fmt.Fprintf(stderr, "loading native model from %s\n", opts.modelDir)
	model, err := mosstranscribe.LoadNativeModel(opts.modelDir)
	if err != nil {
		return err
	}
	defer model.Close()
	fmt.Fprintf(stderr, "model loaded in %s\n", time.Since(started).Round(time.Millisecond))
	if opts.cpuOnly {
		fmt.Fprintln(stderr, "backend=cpu/simd (-cpu)")
	} else if model.EnableGPU() {
		fmt.Fprintln(stderr, "backend=nvidia-ptx+cpu/simd (GPU audio encoder/adaptor enabled)")
	} else {
		fmt.Fprintln(stderr, "warning: NVIDIA PTX acceleration unavailable; falling back to CPU/SIMD")
	}

	samples, err := mosstranscribe.ReadAudioWAV(opts.audioPath)
	if err != nil {
		return err
	}
	fmt.Fprintf(stderr, "audio %.2fs, encoding\n", float64(len(samples))/mosstranscribe.AudioSampleRate)
	audioEmbeddings, audioTokens, err := model.EncodeAudio(samples)
	if err != nil {
		return err
	}
	prompt := mosstranscribe.BuildTranscriptionPrompt(opts.prompt)
	inputIDs, err := model.Processor.EncodePrompt(prompt, audioTokens, model.Decoder.Config.MaxSeqLen)
	if err != nil {
		return err
	}
	fmt.Fprintf(stderr, "audio_tokens=%d prompt_tokens=%d generating<=%d\n", audioTokens, len(inputIDs), opts.maxNew)
	generated, err := model.GenerateGreedy(inputIDs, audioEmbeddings, opts.maxNew)
	if err != nil {
		return err
	}
	raw := model.Processor.Decode(generated)
	segments := mosstranscribe.ParseTranscript(raw)
	payload, err := render(opts.format, raw, segments)
	if err != nil {
		return err
	}
	if opts.outputPath == "" {
		_, err = io.WriteString(stdout, payload)
		return err
	}
	if err := os.WriteFile(opts.outputPath, []byte(payload), 0o644); err != nil {
		return err
	}
	fmt.Fprintf(stderr, "wrote %s\n", opts.outputPath)
	return nil
}

func validateFormat(format string) error {
	switch format {
	case "text", "raw", "json", "srt", "ass":
		return nil
	default:
		return fmt.Errorf("unsupported format %q (want text, raw, json, srt, or ass)", format)
	}
}

func render(format, raw string, segments []mosstranscribe.TranscriptSegment) (string, error) {
	switch format {
	case "raw":
		return raw + "\n", nil
	case "text":
		var out strings.Builder
		for _, segment := range segments {
			fmt.Fprintf(&out, "[%.2f-%.2f][%s] %s\n", segment.Start, segment.End, segment.Speaker, segment.Text)
		}
		return out.String(), nil
	case "srt":
		return mosstranscribe.TranscriptSRT(segments), nil
	case "ass":
		return mosstranscribe.TranscriptASS(segments), nil
	case "json":
		data, err := json.MarshalIndent(struct {
			Text     string                             `json:"text"`
			Segments []mosstranscribe.TranscriptSegment `json:"segments"`
		}{raw, segments}, "", "  ")
		return string(data) + "\n", err
	default:
		return "", fmt.Errorf("unsupported format %q", format)
	}
}
