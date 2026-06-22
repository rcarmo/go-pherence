package minicpmv

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/config"
)

type AudioSpecialTokenIDs struct {
	Audio      int `json:"audio"`
	AudioStart int `json:"audio_start"`
	AudioEnd   int `json:"audio_end"`
	AudioPatch int `json:"audio_patch"`
}

func ResolveAudioSpecialTokenIDs(tok *config.MiniCPMVTokenizerMetadata) (AudioSpecialTokenIDs, bool, error) {
	out := AudioSpecialTokenIDs{}
	if tok == nil {
		return out, false, nil
	}
	seen := false
	if id, ok := lookupAny(tok.TokenIDs, "<audio>"); ok {
		out.Audio = id
		seen = true
	}
	if id, ok := lookupAny(tok.TokenIDs, "<audio_start>", "<|audio_start|>"); ok {
		out.AudioStart = id
		seen = true
	}
	if id, ok := lookupAny(tok.TokenIDs, "<audio_end>", "<|audio_end|>"); ok {
		out.AudioEnd = id
		seen = true
	}
	if id, ok := lookupAny(tok.TokenIDs, "<audio_patch>"); ok {
		out.AudioPatch = id
		seen = true
	}
	if !seen && (tok.Audio != "" || tok.AudioStart != "" || tok.AudioEnd != "" || tok.AudioPatch != "") {
		return out, true, fmt.Errorf("MiniCPM-O audio token strings are present but audio token ids are missing")
	}
	return out, seen, nil
}
