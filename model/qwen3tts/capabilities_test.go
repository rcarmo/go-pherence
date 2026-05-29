package qwen3tts

import "testing"

func TestCapabilitiesByVariant(t *testing.T) {
	for _, tc := range []struct {
		modelType ModelType
		want      ConditioningMode
	}{
		{CustomVoice, ConditioningCustomVoice},
		{Base, ConditioningReferenceAudio},
		{VoiceDesign, ConditioningVoiceDesign},
	} {
		cfg := ParsedConfig{ModelType: tc.modelType}
		caps, err := cfg.Capabilities()
		if err != nil {
			t.Fatal(err)
		}
		if caps.Conditioning != tc.want {
			t.Fatalf("%s conditioning=%s want=%s", tc.modelType, caps.Conditioning, tc.want)
		}
		if !caps.RequiresLanguage || len(caps.SupportedLanguages) != len(AllLanguages()) {
			t.Fatalf("%s languages=%+v", tc.modelType, caps)
		}
	}
}

func TestValidateConditioning(t *testing.T) {
	custom := ParsedConfig{ModelType: CustomVoice}
	if err := custom.ValidateConditioning(ConditioningRequest{Language: English, Speaker: Ryan}); err != nil {
		t.Fatal(err)
	}
	if err := custom.ValidateConditioning(ConditioningRequest{Language: English, HasReferenceAudio: true}); err == nil {
		t.Fatal("expected CustomVoice to reject reference audio")
	}
	base := ParsedConfig{ModelType: Base}
	if err := base.ValidateConditioning(ConditioningRequest{Language: English, HasReferenceAudio: true}); err != nil {
		t.Fatal(err)
	}
	if err := base.ValidateConditioning(ConditioningRequest{Language: English, Speaker: Ryan, HasReferenceAudio: true}); err == nil {
		t.Fatal("expected Base to reject fixed speaker")
	}
	voiceDesign := ParsedConfig{ModelType: VoiceDesign}
	if err := voiceDesign.ValidateConditioning(ConditioningRequest{Language: English, VoicePrompt: "warm studio voice"}); err != nil {
		t.Fatal(err)
	}
	if err := voiceDesign.ValidateConditioning(ConditioningRequest{Language: English}); err == nil {
		t.Fatal("expected VoiceDesign to require prompt")
	}
}

func TestCheckConditioningIncludesRequestAndError(t *testing.T) {
	base := ParsedConfig{ModelType: Base}
	bad := base.CheckConditioning(ConditioningRequest{Language: English})
	if bad.Valid || bad.Error == "" {
		t.Fatalf("bad=%+v", bad)
	}
	good := base.CheckConditioning(ConditioningRequest{Language: English, ReferenceAudio: "voice.wav"})
	if !good.Valid || !good.Request.HasReferenceAudio {
		t.Fatalf("good=%+v", good)
	}
}

func TestAllSpeakerLanguageIDs(t *testing.T) {
	for _, lang := range AllLanguages() {
		if _, err := lang.TokenID(); err != nil {
			t.Fatalf("language %q: %v", lang, err)
		}
	}
	for _, speaker := range AllSpeakers() {
		if _, err := speaker.TokenID(); err != nil {
			t.Fatalf("speaker %q: %v", speaker, err)
		}
		if _, err := speaker.NativeLanguage(); err != nil {
			t.Fatalf("speaker native %q: %v", speaker, err)
		}
	}
}
