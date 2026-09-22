package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"

	"github.com/rcarmo/go-pherence/model/pockettts"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("pockettts", flag.ContinueOnError)
	configPath := fs.String("config", "", "Pocket TTS YAML config")
	weights := fs.String("model", "", "Pocket TTS model.safetensors")
	tokenizer := fs.String("tokenizer", "", "Pocket TTS tokenizer.json")
	voice := fs.String("voice", "", "Pocket TTS exported voice-state safetensors")
	text := fs.String("text", "Hello world!", "text to synthesize")
	out := fs.String("out", "pocket-tts.wav", "exclusive output WAV path")
	maxFrames := fs.Int("max-frames", 375, "maximum generated latent frames")
	steps := fs.Int("steps", 1, "LSD decode steps")
	temperature := fs.Float64("temperature", .3, "noise temperature")
	eos := fs.Float64("eos-threshold", -4, "EOS logit threshold")
	seed := fs.Uint64("seed", 0, "deterministic noise seed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *configPath == "" || *weights == "" || *tokenizer == "" || *voice == "" || *maxFrames <= 0 {
		return fmt.Errorf("usage: pockettts -config english.yaml -model model.safetensors -tokenizer tokenizer.json -voice alba.safetensors [-text text] [-out file]")
	}
	data, err := os.ReadFile(*configPath)
	if err != nil {
		return err
	}
	cfg, err := pockettts.ParseConfig(data)
	if err != nil {
		return err
	}
	gen, err := pockettts.LoadGeneratorCPU(*weights, cfg)
	if err != nil {
		return err
	}
	tf, err := os.Open(*tokenizer)
	if err != nil {
		return err
	}
	tok, loadErr := pockettts.LoadTokenizer(tf)
	closeErr := tf.Close()
	if loadErr != nil {
		return loadErr
	}
	if closeErr != nil {
		return closeErr
	}
	prepared, framesAfter, err := pockettts.PrepareText(*text)
	if err != nil {
		return err
	}
	ids, err := tok.Encode(prepared)
	if err != nil {
		return err
	}
	voiceState, err := pockettts.LoadVoiceState(*voice, gen.FlowLM.Transformer, len(ids)+*maxFrames)
	if err != nil {
		return err
	}
	session, err := pockettts.NewSession(gen, voiceState, *maxFrames)
	if err != nil {
		return err
	}
	if math.IsNaN(*temperature) || math.IsInf(*temperature, 0) || *temperature <= 0 {
		return fmt.Errorf("invalid temperature %g", *temperature)
	}
	rng := rand.New(rand.NewPCG(*seed, *seed^0x9e3779b97f4a7c15))
	std := math.Sqrt(*temperature)
	noise := func(_ int, dst []float32) error {
		for i := range dst {
			dst[i] = float32(rng.NormFloat64() * std)
		}
		return nil
	}
	pcm := make([]float32, *maxFrames*pockettts.SamplesPerFrame)
	samples, err := session.GenerateInto(pcm, ids, *maxFrames, framesAfter, *steps, float32(*eos), noise)
	if err != nil {
		return err
	}
	pcm = pcm[:samples]
	if err := pockettts.WritePCM16Mono24k(*out, pcm); err != nil {
		return err
	}
	return json.NewEncoder(stdout).Encode(map[string]any{"output": *out, "text": prepared, "tokens": len(ids), "frames": samples / pockettts.SamplesPerFrame, "samples": samples, "sample_rate": pockettts.SampleRate, "seed": *seed})
}
