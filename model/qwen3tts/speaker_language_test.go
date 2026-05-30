package qwen3tts

import "testing"

func TestSpeakerLanguageCompatibility(t *testing.T) {
	compat, err := CheckSpeakerLanguage(Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	if compat.NativeLanguage != English || !compat.NativeMatch {
		t.Fatalf("compat=%+v", compat)
	}
	compat, err = CheckSpeakerLanguage(Ryan, Chinese)
	if err != nil {
		t.Fatal(err)
	}
	if compat.NativeLanguage != English || compat.NativeMatch {
		t.Fatalf("compat=%+v", compat)
	}
	if err := RequireNativeSpeakerLanguage(Ryan, English); err != nil {
		t.Fatal(err)
	}
	if err := RequireNativeSpeakerLanguage(Ryan, Chinese); err == nil {
		t.Fatal("expected native-language mismatch")
	}
}

func TestSpeakerLanguageCompatibilityRejectsUnknown(t *testing.T) {
	if _, err := CheckSpeakerLanguage(Speaker("bad"), English); err == nil {
		t.Fatal("expected bad speaker error")
	}
	if _, err := CheckSpeakerLanguage(Ryan, Language("bad")); err == nil {
		t.Fatal("expected bad language error")
	}
}
