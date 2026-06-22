package minicpmv

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestResolveAudioSpecialTokenIDs(t *testing.T) {
	tok := &config.MiniCPMVTokenizerMetadata{TokenIDs: map[string]int{"<audio>": 30, "<audio_start>": 31, "<audio_end>": 32, "<audio_patch>": 33}}
	ids, ok, err := ResolveAudioSpecialTokenIDs(tok)
	if err != nil || !ok {
		t.Fatalf("ResolveAudioSpecialTokenIDs ok=%v err=%v", ok, err)
	}
	if ids.Audio != 30 || ids.AudioStart != 31 || ids.AudioEnd != 32 || ids.AudioPatch != 33 {
		t.Fatalf("bad audio ids: %+v", ids)
	}
}

func TestResolveAudioSpecialTokenIDsNoAudio(t *testing.T) {
	ids, ok, err := ResolveAudioSpecialTokenIDs(&config.MiniCPMVTokenizerMetadata{TokenIDs: map[string]int{"<im_patch>": 20}})
	if err != nil || ok || ids != (AudioSpecialTokenIDs{}) {
		t.Fatalf("expected no audio ids ok=%v ids=%+v err=%v", ok, ids, err)
	}
}

func TestResolveAudioSpecialTokenIDsMissingIDs(t *testing.T) {
	_, ok, err := ResolveAudioSpecialTokenIDs(&config.MiniCPMVTokenizerMetadata{Audio: "<audio>"})
	if !ok || err == nil {
		t.Fatalf("expected missing audio id error ok=%v err=%v", ok, err)
	}
}
