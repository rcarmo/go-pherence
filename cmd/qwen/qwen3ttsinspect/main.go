package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/qwen3tts"
)

type report struct {
	ModelDir               string                           `json:"model_dir"`
	Label                  string                           `json:"label"`
	Config                 qwen3tts.ParsedConfig            `json:"config"`
	TensorCoverage         *qwen3tts.TensorCoverage         `json:"tensor_coverage,omitempty"`
	TensorShapes           *qwen3tts.TensorShapeSummary     `json:"tensor_shapes,omitempty"`
	ShapeValidation        *qwen3tts.TensorShapeValidation  `json:"shape_validation,omitempty"`
	RuntimePlan            qwen3tts.RuntimePlan             `json:"runtime_plan"`
	RuntimeRequestPlan     *qwen3tts.RuntimeRequestPlan     `json:"runtime_request_plan,omitempty"`
	RuntimeStatus          qwen3tts.RuntimeStatus           `json:"runtime_status"`
	Readiness              qwen3tts.RuntimeReadinessReport  `json:"readiness"`
	Capabilities           qwen3tts.Capabilities            `json:"capabilities"`
	ConditioningValidation *qwen3tts.ConditioningValidation `json:"conditioning_validation,omitempty"`
	CustomVoiceProbe       *customVoicePrefixSummary        `json:"custom_voice_prefix,omitempty"`
	ReferenceCoverage      *qwen3tts.ReferenceCoverage      `json:"reference_coverage,omitempty"`
}

type customVoicePrefixSummary struct {
	Speaker       qwen3tts.Speaker             `json:"speaker"`
	Language      qwen3tts.Language            `json:"language"`
	FirstTextID   uint32                       `json:"first_text_id"`
	TextStream    []uint32                     `json:"text_stream"`
	CodecStream   []uint32                     `json:"codec_stream"`
	PrefillLength int                          `json:"prefill_length"`
	RuntimeLayout qwen3tts.PromptRuntimeLayout `json:"runtime_layout"`
}

func main() {
	modelDir := flag.String("model", "", "Qwen3-TTS model directory containing config.json")
	safetensorPath := flag.String("safetensors", "", "optional safetensors path; defaults to model.safetensors or sharded index in -model")
	jsonOut := flag.Bool("json", false, "emit JSON report")
	fixturePath := flag.String("fixture", "", "optional Qwen3-TTS reference fixture path for coverage reporting")
	requireCompleteFixture := flag.Bool("require-complete-fixture", false, "exit non-zero when -fixture reference coverage is incomplete")
	requireNumericParity := flag.Bool("require-numeric-parity", false, "exit non-zero when -fixture contains placeholder parity values")
	requireReady := flag.Bool("require-ready", false, "exit non-zero until runtime and numeric parity coverage are both ready")
	requireRuntime := flag.Bool("require-runtime", false, "exit non-zero when runtime execution stages are not implemented")
	speakerName := flag.String("speaker", "ryan", "CustomVoice speaker for prefix probe")
	langName := flag.String("language", "en", "language for prefix probe")
	firstTextID := flag.Uint("first-text-id", 0, "optional first token ID for CustomVoice prefix probe")
	promptText := flag.String("text", "", "optional text to tokenize and build a CustomVoice prompt from tokenizer files in -model")
	referenceAudio := flag.String("reference-audio", "", "optional reference audio path marker for Base conditioning validation")
	voicePrompt := flag.String("voice-prompt", "", "optional VoiceDesign prompt for conditioning validation")
	strict := flag.Bool("strict", false, "exit non-zero when tensor readiness, shape validation, or requested conditioning validation fails")
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
	out := report{ModelDir: *modelDir, Label: cfg.Label(), Config: cfg, RuntimePlan: plan, RuntimeStatus: qwen3tts.CurrentRuntimeStatus(), Capabilities: caps}
	if *fixturePath != "" {
		fixture, err := qwen3tts.LoadReferenceFixture(*fixturePath)
		if err != nil {
			fatal(err)
		}
		coverage := fixture.Coverage()
		out.ReferenceCoverage = &coverage
		if fixture.Runtime != nil {
			requestPlan, err := qwen3tts.NewRuntimeRequestPlan(cfg, qwen3tts.RuntimeRequest{Conditioning: qwen3tts.ConditioningRequest{Speaker: fixture.Speaker, Language: fixture.Language}, Prompt: fixture.Prompt, MaxFrames: fixture.Runtime.MaxFrames})
			if err != nil {
				fatal(err)
			}
			out.RuntimeRequestPlan = &requestPlan
		}
	}
	if infos, err := safetensors.TensorInfosFrom(*modelDir, *safetensorPath); err == nil {
		names := make([]string, 0, len(infos))
		for name := range infos {
			names = append(names, name)
		}
		cov := qwen3tts.InspectTensorNames(names)
		out.TensorCoverage = &cov
		shapes := qwen3tts.InspectTensorShapes(infos)
		out.TensorShapes = &shapes
		shapeValidation := qwen3tts.ValidateTensorShapes(cfg, infos)
		out.ShapeValidation = &shapeValidation
	}
	var parsedSpeaker qwen3tts.Speaker
	var hasSpeaker bool
	if *speakerName != "" {
		speaker, err := qwen3tts.ParseSpeaker(*speakerName)
		if err != nil {
			fatal(err)
		}
		parsedSpeaker = speaker
		hasSpeaker = true
	}
	lang, err := qwen3tts.ParseLanguage(*langName)
	if err != nil {
		fatal(err)
	}
	if *firstTextID != 0 || *promptText != "" {
		if !hasSpeaker {
			fatal(fmt.Errorf("CustomVoice prompt probe requires -speaker"))
		}
		if *promptText != "" {
			tok, err := qwen3tts.LoadTokenizer(*modelDir)
			if err != nil {
				fatal(err)
			}
			prompt, err := qwen3tts.BuildCustomVoicePrompt(tok, *promptText, parsedSpeaker, lang)
			if err != nil {
				fatal(err)
			}
			runtimeLayout, err := qwen3tts.NewPromptRuntimeLayout(cfg, prompt)
			if err != nil {
				fatal(err)
			}
			out.CustomVoiceProbe = &customVoicePrefixSummary{Speaker: parsedSpeaker, Language: lang, FirstTextID: prompt.Text[qwen3tts.CustomVoiceFirstTextIndex], TextStream: prompt.Text, CodecStream: prompt.Codec, PrefillLength: len(prompt.Text), RuntimeLayout: runtimeLayout}
		} else {
			text, codec, err := qwen3tts.CustomVoicePrefixIDs(uint32(*firstTextID), parsedSpeaker, lang)
			if err != nil {
				fatal(err)
			}
			runtimeLayout, err := qwen3tts.NewPromptRuntimeLayout(cfg, qwen3tts.PromptIDs{Text: text, Codec: codec})
			if err != nil {
				fatal(err)
			}
			out.CustomVoiceProbe = &customVoicePrefixSummary{Speaker: parsedSpeaker, Language: lang, FirstTextID: uint32(*firstTextID), TextStream: text, CodecStream: codec, PrefillLength: len(text), RuntimeLayout: runtimeLayout}
		}
	}
	if *referenceAudio != "" || *voicePrompt != "" || *firstTextID != 0 || *promptText != "" {
		req := qwen3tts.ConditioningRequest{Language: lang, ReferenceAudio: *referenceAudio, VoicePrompt: *voicePrompt}
		if hasSpeaker && (*firstTextID != 0 || *promptText != "" || cfg.ModelType == qwen3tts.CustomVoice) {
			req.Speaker = parsedSpeaker
		}
		validation := cfg.CheckConditioning(req)
		out.ConditioningValidation = &validation
	}
	out.Readiness = qwen3tts.NewRuntimeReadinessReport(out.RuntimeStatus, out.ReferenceCoverage)
	if *jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fatal(err)
		}
	} else {
		printText(out)
	}
	if *strict && !reportValid(out) {
		os.Exit(1)
	}
	if *requireCompleteFixture && (out.ReferenceCoverage == nil || !out.ReferenceCoverage.CompleteRuntimeTrace) {
		os.Exit(1)
	}
	if *requireNumericParity && (out.ReferenceCoverage == nil || !out.ReferenceCoverage.NumericParityReady) {
		os.Exit(1)
	}
	if *requireReady && !out.Readiness.ReadyForExecution {
		os.Exit(1)
	}
	if *requireRuntime && !out.RuntimeStatus.RuntimeImplemented {
		os.Exit(1)
	}
}

func printText(r report) {
	c := r.Config
	fmt.Printf("Qwen3-TTS: %s (%s)\n", r.Label, r.ModelDir)
	fmt.Printf("  variant: %s size=%s speaker_encoder=%v speaker_dim=%d speaker_sample_rate=%d conditioning=%s\n", c.ModelType, c.ModelSize, r.RuntimePlan.SpeakerEncoderLayout.Present, r.RuntimePlan.SpeakerEncoderLayout.EmbeddingDim, r.RuntimePlan.SpeakerEncoderLayout.SampleRateHz, r.Capabilities.Conditioning)
	fmt.Printf("  talker: hidden=%d layers=%d heads=%d kv_heads=%d q_per_kv=%d head_dim=%d text_hidden=%d text_vocab=%d codec_vocab=%d rope_theta=%g max_pos=%d mrope=%v\n", c.TalkerHiddenSize, c.TalkerNumHiddenLayers, c.TalkerNumAttentionHeads, c.TalkerNumKeyValueHeads, r.RuntimePlan.TalkerAttentionLayout.QueriesPerKV, c.TalkerHeadDim, c.TalkerTextHiddenSize, c.TalkerTextVocabSize, c.TalkerVocabSize, r.RuntimePlan.TalkerAttentionLayout.RoPETheta, r.RuntimePlan.TalkerAttentionLayout.MaxPositionEmbeddings, c.HasMRoPESection)
	fmt.Printf("  code predictor: hidden=%d layers=%d heads=%d kv_heads=%d q_per_kv=%d head_dim=%d vocab=%d code_groups=%d acoustic_heads=%d rope_theta=%g\n", c.CPHiddenSize, c.CPNumHiddenLayers, c.CPNumAttentionHeads, c.CPNumKeyValueHeads, r.RuntimePlan.CPAttentionLayout.QueriesPerKV, c.CPHeadDim, c.CPVocabSize, c.CPNumCodeGroups, r.RuntimePlan.CodePredictorHeadLayout.Heads, r.RuntimePlan.CPAttentionLayout.RoPETheta)
	fmt.Printf("  runtime plan: talker_kv_floats/token=%d cp_kv_floats/token=%d talker_ffn_floats/layer=%d cp_ffn_floats/layer=%d embedding_bridge_floats=%d decoder=%dHz/%d_codes decoder_groups=%d-%d waveform=%dHz/%dspf pipeline=%d_stages\n", r.RuntimePlan.Talker.KVFloatsPerToken, r.RuntimePlan.CodePredictor.KVFloatsPerToken, r.RuntimePlan.TalkerFFNLayout.FloatsPerLayer, r.RuntimePlan.CPFFNLayout.FloatsPerLayer, r.RuntimePlan.EmbeddingLayout.TotalBridgeFloats, r.RuntimePlan.Decoder12Hz.FrameRateHz, r.RuntimePlan.Decoder12Hz.CodeGroups, r.RuntimePlan.DecoderInputLayout.FirstCodeGroup, r.RuntimePlan.DecoderInputLayout.LastCodeGroup, r.RuntimePlan.WaveformLayout.SampleRateHz, r.RuntimePlan.WaveformLayout.SamplesPerFrame, len(r.RuntimePlan.Pipeline.Steps))
	fmt.Printf("  runtime status: implemented=%v pending=%v ready_for_execution=%v blockers=%v\n", r.RuntimeStatus.RuntimeImplemented, r.RuntimeStatus.Pending, r.Readiness.ReadyForExecution, r.Readiness.Blockers)
	if r.RuntimeRequestPlan != nil {
		fmt.Printf("  runtime request: frames=%d samples=%d codes=%d\n", r.RuntimeRequestPlan.MaxFrames, r.RuntimeRequestPlan.MaxSamples, r.RuntimeRequestPlan.MaxCodes)
	}
	if r.ReferenceCoverage != nil {
		fmt.Printf("  references: complete=%v numeric_parity_ready=%v missing=%v placeholders=%v\n", r.ReferenceCoverage.CompleteRuntimeTrace, r.ReferenceCoverage.NumericParityReady, r.ReferenceCoverage.Missing, r.ReferenceCoverage.PlaceholderValues)
	}
	if r.TensorCoverage != nil {
		t := r.TensorCoverage
		shapeValid := true
		if r.ShapeValidation != nil {
			shapeValid = r.ShapeValidation.Valid
		}
		fmt.Printf("  tensors: total=%d talker=%d code_predictor=%d speech_tokenizer=%d speaker_encoder=%d other=%d ready=%v missing=%v shapes_valid=%v\n", t.Total, t.Talker, t.CodePredictor, t.SpeechTokenizer, t.SpeakerEncoder, t.Other, t.Readiness.Ready, t.Readiness.MissingRequired, shapeValid)
	}
	if r.CustomVoiceProbe != nil {
		p := r.CustomVoiceProbe
		fmt.Printf("  custom voice prefix: speaker=%s language=%s first_text_id=%d prefill_len=%d fused_input_floats=%d\n", p.Speaker, p.Language, p.FirstTextID, p.PrefillLength, p.RuntimeLayout.TalkerInput.FusedInputFloats)
	}
}

func reportValid(r report) bool {
	if r.TensorCoverage != nil && !r.TensorCoverage.Readiness.Ready {
		return false
	}
	if r.ShapeValidation != nil && !r.ShapeValidation.Valid {
		return false
	}
	if r.ConditioningValidation != nil && !r.ConditioningValidation.Valid {
		return false
	}
	return true
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "qwen3ttsinspect:", err)
	os.Exit(1)
}
