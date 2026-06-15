package diffusiongemma

import "fmt"

// ExpandImagePlaceholderTokens converts tokenizer-level image placeholders into
// the prompt token span expected by the multimodal Gemma processor:
//
//	<image_token> -> <boi> <image_token> × softTokens <eoi>
//
// Text-only prompts are returned as a copy with no changes. This is the prompt
// processor boundary for multimodal inputs; the remaining runtime work is to
// replace these soft-token slots with vision-encoder image embeddings.
func ExpandImagePlaceholderTokens(ids []int, specials SpecialTokenIDs, softTokens int) ([]int, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	count := 0
	for _, id := range ids {
		if id == specials.IMAGE {
			count++
		}
	}
	if count == 0 {
		return append([]int(nil), ids...), nil
	}
	if specials.IMAGE < 0 || specials.BOI < 0 || specials.EOI < 0 {
		return nil, fmt.Errorf("DiffusionGemma image prompt tokens unavailable")
	}
	if softTokens <= 0 {
		return nil, fmt.Errorf("DiffusionGemma invalid image soft token count %d", softTokens)
	}
	out := make([]int, 0, len(ids)+count*(softTokens+1))
	for _, id := range ids {
		if id != specials.IMAGE {
			out = append(out, id)
			continue
		}
		out = append(out, specials.BOI)
		for i := 0; i < softTokens; i++ {
			out = append(out, specials.IMAGE)
		}
		out = append(out, specials.EOI)
	}
	return out, nil
}
