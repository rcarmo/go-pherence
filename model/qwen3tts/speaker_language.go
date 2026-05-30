package qwen3tts

import "fmt"

// SpeakerLanguageCompatibility records CustomVoice speaker/language pairing
// metadata. Cross-language synthesis may be supported by checkpoints, but native
// language information is useful for validation and future quality warnings.
type SpeakerLanguageCompatibility struct {
	Speaker        Speaker  `json:"speaker"`
	Language       Language `json:"language"`
	NativeLanguage Language `json:"native_language"`
	NativeMatch    bool     `json:"native_match"`
}

func CheckSpeakerLanguage(speaker Speaker, language Language) (SpeakerLanguageCompatibility, error) {
	if _, err := speaker.TokenID(); err != nil {
		return SpeakerLanguageCompatibility{}, err
	}
	if _, err := language.TokenID(); err != nil {
		return SpeakerLanguageCompatibility{}, err
	}
	native, err := speaker.NativeLanguage()
	if err != nil {
		return SpeakerLanguageCompatibility{}, err
	}
	return SpeakerLanguageCompatibility{Speaker: speaker, Language: language, NativeLanguage: native, NativeMatch: native == language}, nil
}

func RequireNativeSpeakerLanguage(speaker Speaker, language Language) error {
	compat, err := CheckSpeakerLanguage(speaker, language)
	if err != nil {
		return err
	}
	if !compat.NativeMatch {
		return fmt.Errorf("Qwen3-TTS speaker %s native language is %s, got %s", speaker, compat.NativeLanguage, language)
	}
	return nil
}
