// qwen3tts-twoframe is a bounded 160 ms native CPU parity probe for the
// pinned Qwen3-TTS 0.6B CustomVoice checkpoint, not a general TTS CLI.
package main

import (
	"crypto/sha256"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/rcarmo/go-pherence/model/qwen3tts"
)

const (
	modelSHA256     = "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb"
	modelBytes      = 1811626576
	tokenizerSHA256 = "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"
	tokenizerBytes  = 682293092
)

func main() {
	modelDir := flag.String("model-dir", "", "pinned Qwen3-TTS 12Hz 0.6B CustomVoice directory")
	output := flag.String("out", "", "new WAV output path (will not overwrite)")
	flag.Parse()
	if *modelDir == "" || *output == "" || flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: qwen3tts-twoframe -model-dir DIR -out FILE.wav")
		os.Exit(2)
	}
	if err := run(*modelDir, *output); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(dir, output string) error {
	for _, item := range []struct {
		path, hash string
		size       int64
	}{
		{filepath.Join(dir, "model.safetensors"), modelSHA256, modelBytes},
		{filepath.Join(dir, "speech_tokenizer", "model.safetensors"), tokenizerSHA256, tokenizerBytes},
	} {
		if err := verifyArtifact(item.path, item.hash, item.size); err != nil {
			return err
		}
	}
	cfg, err := qwen3tts.ReadModelDir(dir)
	if err != nil {
		return err
	}
	if cfg.ModelType != qwen3tts.CustomVoice || cfg.ModelSize != "0b6" {
		return fmt.Errorf("unsupported Qwen3-TTS model %s", cfg.Label())
	}
	tok, err := qwen3tts.LoadTokenizer(dir)
	if err != nil {
		return err
	}
	prompt, err := qwen3tts.BuildCustomVoicePrompt(tok, "Hello world", qwen3tts.Ryan, qwen3tts.English)
	if err != nil {
		return err
	}
	plan, err := qwen3tts.NewRuntimeRequestPlan(cfg, qwen3tts.RuntimeRequest{Conditioning: qwen3tts.ConditioningRequest{Speaker: qwen3tts.Ryan, Language: qwen3tts.English}, Prompt: prompt, MaxFrames: 2})
	if err != nil {
		return err
	}
	talker, err := qwen3tts.LoadTalkerCPUFromDir(dir, cfg)
	if err != nil {
		return err
	}
	predictor, err := qwen3tts.LoadCodePredictorCPUFromDir(dir, cfg)
	if err != nil {
		return err
	}
	decoder, err := qwen3tts.LoadDecoder12HzCPUFromDir(dir)
	if err != nil {
		return err
	}
	result, err := qwen3tts.GenerateTwoFramesCPU(plan, talker, predictor, decoder)
	if err != nil {
		return err
	}
	if err := qwen3tts.WritePCM16Mono24k(output, result.Waveform); err != nil {
		return err
	}
	fmt.Printf("wrote %s: 24000 Hz mono, %d samples, semantic=%v\n", output, len(result.Waveform), result.Semantic)
	return nil
}

func verifyArtifact(path, expected string, size int64) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() != size {
		return fmt.Errorf("%s: size/type mismatch: size=%d want=%d", path, info.Size(), size)
	}
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return err
	}
	if fmt.Sprintf("%x", hash.Sum(nil)) != expected {
		return fmt.Errorf("%s: SHA-256 mismatch", path)
	}
	return nil
}
