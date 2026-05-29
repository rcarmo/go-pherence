// Package qwen3tts contains metadata, token constants, and validation helpers
// for Qwen3-TTS checkpoints. Runtime inference is intentionally staged after
// config/tensor coverage and reference fixtures are in place.
package qwen3tts

import (
	"fmt"
	"strings"
)

const (
	IMStart   uint32 = 151644
	IMEnd     uint32 = 151645
	Assistant uint32 = 77091
	Newline   uint32 = 198
)

const (
	TTSPad uint32 = 151671
	TTSBOS uint32 = 151672
	TTSEOS uint32 = 151673
)

const (
	CodecPad      uint32 = 2148
	CodecBOS      uint32 = 2149
	CodecEOS      uint32 = 2150
	CodecThink    uint32 = 2154
	CodecNoThink  uint32 = 2155
	CodecThinkBOS uint32 = 2156
	CodecThinkEOS uint32 = 2157

	CodecVocabSize = 3072
)

type Language string

const (
	Chinese    Language = "zh"
	English    Language = "en"
	Japanese   Language = "ja"
	Korean     Language = "ko"
	German     Language = "de"
	French     Language = "fr"
	Russian    Language = "ru"
	Portuguese Language = "pt"
	Spanish    Language = "es"
	Italian    Language = "it"
)

func ParseLanguage(s string) (Language, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "chinese", "zh":
		return Chinese, nil
	case "english", "en":
		return English, nil
	case "japanese", "ja":
		return Japanese, nil
	case "korean", "ko":
		return Korean, nil
	case "german", "de":
		return German, nil
	case "french", "fr":
		return French, nil
	case "russian", "ru":
		return Russian, nil
	case "portuguese", "pt":
		return Portuguese, nil
	case "spanish", "es":
		return Spanish, nil
	case "italian", "it":
		return Italian, nil
	default:
		return "", fmt.Errorf("unknown Qwen3-TTS language %q", s)
	}
}

func (l Language) TokenID() (uint32, error) {
	switch l {
	case Chinese:
		return 2055, nil
	case English:
		return 2050, nil
	case Japanese:
		return 2058, nil
	case Korean:
		return 2064, nil
	case German:
		return 2053, nil
	case French:
		return 2061, nil
	case Russian:
		return 2069, nil
	case Portuguese:
		return 2071, nil
	case Spanish:
		return 2054, nil
	case Italian:
		return 2070, nil
	default:
		return 0, fmt.Errorf("unknown Qwen3-TTS language %q", l)
	}
}

type Speaker string

const (
	Serena  Speaker = "serena"
	Vivian  Speaker = "vivian"
	UncleFu Speaker = "uncle_fu"
	Ryan    Speaker = "ryan"
	Aiden   Speaker = "aiden"
	OnoAnna Speaker = "ono_anna"
	Sohee   Speaker = "sohee"
	Eric    Speaker = "eric"
	Dylan   Speaker = "dylan"
)

func ParseSpeaker(s string) (Speaker, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "serena":
		return Serena, nil
	case "vivian":
		return Vivian, nil
	case "uncle_fu", "unclefu":
		return UncleFu, nil
	case "ryan":
		return Ryan, nil
	case "aiden":
		return Aiden, nil
	case "ono_anna", "onoanna":
		return OnoAnna, nil
	case "sohee":
		return Sohee, nil
	case "eric":
		return Eric, nil
	case "dylan":
		return Dylan, nil
	default:
		return "", fmt.Errorf("unknown Qwen3-TTS speaker %q", s)
	}
}

func (s Speaker) TokenID() (uint32, error) {
	switch s {
	case Serena:
		return 3066, nil
	case Vivian:
		return 3065, nil
	case UncleFu:
		return 3010, nil
	case Ryan:
		return 3061, nil
	case Aiden:
		return 2861, nil
	case OnoAnna:
		return 2873, nil
	case Sohee:
		return 2864, nil
	case Eric:
		return 2875, nil
	case Dylan:
		return 2878, nil
	default:
		return 0, fmt.Errorf("unknown Qwen3-TTS speaker %q", s)
	}
}

func (s Speaker) NativeLanguage() (Language, error) {
	switch s {
	case Serena, Vivian, UncleFu, Eric, Dylan:
		return Chinese, nil
	case Ryan, Aiden:
		return English, nil
	case OnoAnna:
		return Japanese, nil
	case Sohee:
		return Korean, nil
	default:
		return "", fmt.Errorf("unknown Qwen3-TTS speaker %q", s)
	}
}

// CustomVoicePrefixIDs returns the deterministic token/control prefix used to
// build Qwen3-TTS CustomVoice prefill embeddings. The final text token is
// overlaid with CodecBOS by the runtime; keeping both streams explicit here
// makes reference fixture comparison straightforward.
func CustomVoicePrefixIDs(firstTextToken uint32, speaker Speaker, language Language) ([]uint32, []uint32, error) {
	langID, err := language.TokenID()
	if err != nil {
		return nil, nil, err
	}
	speakerID, err := speaker.TokenID()
	if err != nil {
		return nil, nil, err
	}
	text := []uint32{IMStart, Assistant, Newline, TTSPad, TTSPad, TTSPad, TTSPad, TTSPad, TTSBOS, firstTextToken}
	codec := []uint32{CodecThink, CodecThinkBOS, langID, CodecThinkEOS, speakerID, CodecPad, CodecBOS}
	return text, codec, nil
}
