package pockettts

import (
	"fmt"
	"strings"
	"unicode"
)

func PrepareText(text string) (string, int, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return "", 0, fmt.Errorf("Pocket TTS text is empty")
	}
	text = strings.ReplaceAll(text, "\n", " ")
	text = strings.ReplaceAll(text, "\r", " ")
	for strings.Contains(text, "  ") {
		text = strings.ReplaceAll(text, "  ", " ")
	}
	framesAfter := 1
	if len(strings.Fields(text)) <= 4 {
		framesAfter = 3
	}
	runes := []rune(text)
	if len(runes) > 0 && !unicode.IsUpper(runes[0]) {
		runes[0] = unicode.ToUpper(runes[0])
		text = string(runes)
	}
	core := strings.TrimRight(text, "\"'”’)]» ")
	closers := strings.TrimSpace(text[len(core):])
	coreRunes := []rune(core)
	if len(coreRunes) > 0 && !strings.ContainsRune(".!?…", coreRunes[len(coreRunes)-1]) {
		core = strings.TrimRight(core, ",;:-–— ") + "."
		text = core + closers
	}
	return text, framesAfter, nil
}
