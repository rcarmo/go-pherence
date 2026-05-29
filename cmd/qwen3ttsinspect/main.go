package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/qwen3tts"
)

type report struct {
	ModelDir         string                    `json:"model_dir"`
	Label            string                    `json:"label"`
	Config           qwen3tts.ParsedConfig     `json:"config"`
	TensorCoverage   *qwen3tts.TensorCoverage  `json:"tensor_coverage,omitempty"`
	RuntimePlan      qwen3tts.RuntimePlan      `json:"runtime_plan"`
	Capabilities     qwen3tts.Capabilities     `json:"capabilities"`
	CustomVoiceProbe *customVoicePrefixSummary `json:"custom_voice_prefix,omitempty"`
}

type customVoicePrefixSummary struct {
	Speaker       qwen3tts.Speaker  `json:"speaker"`
	Language      qwen3tts.Language `json:"language"`
	FirstTextID   uint32            `json:"first_text_id"`
	TextStream    []uint32          `json:"text_stream"`
	CodecStream   []uint32          `json:"codec_stream"`
	PrefillLength int               `json:"prefill_length"`
}

func main() {
	modelDir := flag.String("model", "", "Qwen3-TTS model directory containing config.json")
	safetensorPath := flag.String("safetensors", "", "optional safetensors path; defaults to model.safetensors or sharded index in -model")
	jsonOut := flag.Bool("json", false, "emit JSON report")
	speakerName := flag.String("speaker", "ryan", "CustomVoice speaker for prefix probe")
	langName := flag.String("language", "en", "language for prefix probe")
	firstTextID := flag.Uint("first-text-id", 0, "optional first token ID for CustomVoice prefix probe")
	promptText := flag.String("text", "", "optional text to tokenize and build a CustomVoice prompt from tokenizer files in -model")
	flag.Parse()
	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "usage: qwen3ttsinspect -model <dir> [-safetensors path] [-json]")
		os.Exit(2)
	}
	cfg, err := qwen3tts.ReadModelDir(*modelDir)
	if err != nil {
		fatal(err)
	}
	plan, err := qwen3tts.NewRuntimePlan(cfg)
	if err != nil {
		fatal(err)
	}
	caps, err := cfg.Capabilities()
	if err != nil {
		fatal(err)
	}
	out := report{ModelDir: *modelDir, Label: cfg.Label(), Config: cfg, RuntimePlan: plan, Capabilities: caps}
	if names, err := safetensorNames(*modelDir, *safetensorPath); err == nil {
		cov := qwen3tts.InspectTensorNames(names)
		out.TensorCoverage = &cov
	}
	if *firstTextID != 0 || *promptText != "" {
		speaker, err := qwen3tts.ParseSpeaker(*speakerName)
		if err != nil {
			fatal(err)
		}
		lang, err := qwen3tts.ParseLanguage(*langName)
		if err != nil {
			fatal(err)
		}
		if *promptText != "" {
			tok, err := qwen3tts.LoadTokenizer(*modelDir)
			if err != nil {
				fatal(err)
			}
			prompt, err := qwen3tts.BuildCustomVoicePrompt(tok, *promptText, speaker, lang)
			if err != nil {
				fatal(err)
			}
			out.CustomVoiceProbe = &customVoicePrefixSummary{Speaker: speaker, Language: lang, FirstTextID: prompt.Text[qwen3tts.CustomVoiceFirstTextIndex], TextStream: prompt.Text, CodecStream: prompt.Codec, PrefillLength: len(prompt.Text)}
		} else {
			text, codec, err := qwen3tts.CustomVoicePrefixIDs(uint32(*firstTextID), speaker, lang)
			if err != nil {
				fatal(err)
			}
			out.CustomVoiceProbe = &customVoicePrefixSummary{Speaker: speaker, Language: lang, FirstTextID: uint32(*firstTextID), TextStream: text, CodecStream: codec, PrefillLength: len(text)}
		}
	}
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fatal(err)
		}
		return
	}
	printText(out)
}

func safetensorNames(modelDir, explicit string) ([]string, error) {
	if explicit != "" {
		f, err := safetensors.Open(explicit)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return f.Names(), nil
	}
	if sf, err := safetensors.OpenSharded(filepath.Join(modelDir, "model.safetensors.index.json")); err == nil {
		defer sf.Close()
		return sf.Names(), nil
	}
	f, err := safetensors.Open(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return f.Names(), nil
}

func printText(r report) {
	c := r.Config
	fmt.Printf("Qwen3-TTS: %s (%s)\n", r.Label, r.ModelDir)
	fmt.Printf("  variant: %s size=%s speaker_encoder=%v conditioning=%s\n", c.ModelType, c.ModelSize, c.SpeakerEncoder != nil, r.Capabilities.Conditioning)
	fmt.Printf("  talker: hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d text_hidden=%d text_vocab=%d codec_vocab=%d mrope=%v\n", c.TalkerHiddenSize, c.TalkerNumHiddenLayers, c.TalkerNumAttentionHeads, c.TalkerNumKeyValueHeads, c.TalkerHeadDim, c.TalkerTextHiddenSize, c.TalkerTextVocabSize, c.TalkerVocabSize, c.HasMRoPESection)
	fmt.Printf("  code predictor: hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d code_groups=%d\n", c.CPHiddenSize, c.CPNumHiddenLayers, c.CPNumAttentionHeads, c.CPNumKeyValueHeads, c.CPHeadDim, c.CPVocabSize, c.CPNumCodeGroups)
	fmt.Printf("  runtime plan: talker_kv_floats/token=%d cp_kv_floats/token=%d decoder=%dHz/%d_codes\n", r.RuntimePlan.Talker.KVFloatsPerToken, r.RuntimePlan.CodePredictor.KVFloatsPerToken, r.RuntimePlan.Decoder12Hz.FrameRateHz, r.RuntimePlan.Decoder12Hz.CodeGroups)
	if r.TensorCoverage != nil {
		t := r.TensorCoverage
		fmt.Printf("  tensors: total=%d talker=%d code_predictor=%d speech_tokenizer=%d speaker_encoder=%d other=%d ready=%v missing=%v\n", t.Total, t.Talker, t.CodePredictor, t.SpeechTokenizer, t.SpeakerEncoder, t.Other, t.Readiness.Ready, t.Readiness.MissingRequired)
	}
	if r.CustomVoiceProbe != nil {
		p := r.CustomVoiceProbe
		fmt.Printf("  custom voice prefix: speaker=%s language=%s first_text_id=%d prefill_len=%d\n", p.Speaker, p.Language, p.FirstTextID, p.PrefillLength)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "qwen3ttsinspect:", err)
	os.Exit(1)
}
