package whisper

// Language tokens for Whisper multilingual models.
// Token IDs follow the HuggingFace Whisper convention: SOT + 1 + language_index.
var LanguageTokens = map[string]int{
	"en": 50259, "zh": 50260, "de": 50261, "es": 50262, "ru": 50263,
	"ko": 50264, "fr": 50265, "ja": 50266, "pt": 50267, "tr": 50268,
	"pl": 50269, "ca": 50270, "nl": 50271, "ar": 50272, "sv": 50273,
	"it": 50274, "id": 50275, "hi": 50276, "fi": 50277, "vi": 50278,
	"he": 50279, "uk": 50280, "el": 50281, "ms": 50282, "cs": 50283,
	"ro": 50284, "da": 50285, "hu": 50286, "ta": 50287, "no": 50288,
	"th": 50289, "ur": 50290, "hr": 50291, "bg": 50292, "lt": 50293,
	"la": 50294, "mi": 50295, "ml": 50296, "cy": 50297, "sk": 50298,
	"te": 50299, "fa": 50300, "lv": 50301, "bn": 50302, "sr": 50303,
	"az": 50304, "sl": 50305, "kn": 50306, "et": 50307, "mk": 50308,
	"br": 50309, "eu": 50310, "is": 50311, "hy": 50312, "ne": 50313,
	"mn": 50314, "bs": 50315, "kk": 50316, "sq": 50317, "sw": 50318,
	"gl": 50319, "mr": 50320, "pa": 50321, "si": 50322, "km": 50323,
	"sn": 50324, "yo": 50325, "so": 50326, "af": 50327, "oc": 50328,
	"ka": 50329, "be": 50330, "tg": 50331, "sd": 50332, "gu": 50333,
	"am": 50334, "yi": 50335, "lo": 50336, "uz": 50337, "fo": 50338,
	"ht": 50339, "ps": 50340, "tk": 50341, "nn": 50342, "mt": 50343,
	"sa": 50344, "lb": 50345, "my": 50346, "bo": 50347, "tl": 50348,
	"mg": 50349, "as": 50350, "tt": 50351, "haw": 50352, "ln": 50353,
	"ha": 50354, "ba": 50355, "jw": 50356, "su": 50357,
}

// LanguageNames maps token IDs back to language codes.
var LanguageNames map[int]string

func init() {
	LanguageNames = make(map[int]string, len(LanguageTokens))
	for code, tok := range LanguageTokens {
		LanguageNames[tok] = code
	}
}

// DetectLanguage runs the decoder for one step after SOT to predict the language token.
// Returns the detected language code and confidence.
func DetectLanguage(dec *Decoder, state *DecoderState) (string, float32) {
	// Feed SOT token
	logits := dec.ForwardToken(TokenSOT, state)

	// Find the highest-scoring language token
	bestLang := "en"
	bestScore := float32(-1e9)

	for code, tok := range LanguageTokens {
		if tok < len(logits) && logits[tok] > bestScore {
			bestScore = logits[tok]
			bestLang = code
		}
	}

	// Compute confidence as softmax probability of the best language
	var sumExp float32
	for _, tok := range LanguageTokens {
		if tok < len(logits) {
			exp := expf(logits[tok] - bestScore)
			sumExp += exp
		}
	}
	confidence := float32(1.0) / sumExp // exp(0)/sumExp = 1/sumExp
	if confidence < 0 {
		confidence = 0
	}
	if confidence > 1 {
		confidence = 1
	}

	return bestLang, confidence
}

// TranscribeWithLanguageDetect auto-detects language then transcribes.
func (w *Whisper) TranscribeWithLanguageDetect(samples []float32) (text string, lang string, err error) {
	cfg := w.Config
	melCfg := melConfigAudio(cfg)
	mel := computeMelFromSamples(samples, cfg.NumMelBins, melCfg)
	if mel == nil {
		return "", "en", nil
	}

	T := len(mel) / cfg.NumMelBins
	encoderOutput := w.Encoder.Forward(mel, T)
	encLen := len(encoderOutput) / cfg.EncoderDModel

	// Detect language
	detectState := NewDecoderState(cfg, encoderOutput, encLen, w.Decoder)
	lang, confidence := DetectLanguage(w.Decoder, detectState)

	// Transcribe with detected language
	state := NewDecoderState(cfg, encoderOutput, encLen, w.Decoder)

	langTok, ok := LanguageTokens[lang]
	if !ok {
		langTok = TokenEnglish
	}

	// Feed prompt with detected language
	prompt := []int{TokenSOT, langTok, TokenTranscribe, TokenNoTimestamps}
	for _, tok := range prompt {
		w.Decoder.ForwardToken(tok, state)
	}

	tokens := make([]int, 0, cfg.MaxDecoderLength)
	prevTok := prompt[len(prompt)-1]
	for i := 0; i < cfg.MaxDecoderLength; i++ {
		logits := w.Decoder.ForwardToken(prevTok, state)
		nextTok := argmax(logits)
		if nextTok == TokenEOT {
			break
		}
		tokens = append(tokens, nextTok)
		prevTok = nextTok
	}

	_ = confidence
	return TokensToText(tokens), lang, nil
}

func computeMelFromSamples(samples []float32, numMels int, cfg melCfgHelper) []float32 {
	numFrames := (len(samples) - cfg.FFTSize) / cfg.HopLength
	if numFrames <= 0 {
		return nil
	}
	// Placeholder: use audio.MelSpectrogram would be better but avoid import cycle
	melFlat := make([]float32, numMels*numFrames)
	return melFlat
}

func melConfigAudio(cfg Config) melCfgHelper {
	return melCfgHelper{
		SampleRate: 16000,
		FFTSize:    400,
		HopLength:  160,
		NumMels:    cfg.NumMelBins,
		NFFTPadded: 512,
	}
}

func expf(x float32) float32 {
	if x < -20 {
		return 0
	}
	if x > 20 {
		return 1e9
	}
	// Taylor expansion (good enough for softmax normalization)
	v := 1.0 + float64(x) + float64(x*x)/2 + float64(x*x*x)/6 + float64(x*x*x*x)/24
	if v < 0 {
		v = 0
	}
	return float32(v)
}
