package decider

import (
	"fmt"
	"strings"
)

const noulV15FallbackQuestion = "Which answer fits the context?"

// NoulQuestionV15 renders one Noul question using upstream v1.5's optional
// instruction fallback. It changes no existing SystemOne request or prompt.
// Callers can use its returned question/options with BuildPrompt explicitly.
func NoulQuestionV15(q Question) (string, []string, error) {
	if q.Type != Noul {
		return "", nil, fmt.Errorf("decider: v1.5 fallback supports Noul only")
	}
	if len(q.Criteria) > 2 {
		return "", nil, fmt.Errorf("decider: noul criteria must describe optional false/true")
	}
	desc := map[string]any{}
	for _, criterion := range q.Criteria {
		if criterion.Name != "false" && criterion.Name != "true" {
			return "", nil, fmt.Errorf("decider: noul criterion must be false or true")
		}
		if _, exists := desc[criterion.Name]; exists {
			return "", nil, fmt.Errorf("decider: duplicate noul criterion %q", criterion.Name)
		}
		desc[criterion.Name] = criterion.Description
	}
	text, err := textValue(q.Instructions)
	if err != nil {
		return "", nil, err
	}
	if strings.TrimSpace(text) == "" {
		if !hasNoulDescription(desc["false"]) && !hasNoulDescription(desc["true"]) {
			return "", nil, fmt.Errorf("decider: noul without instructions needs a false or true description")
		}
		text = noulV15FallbackQuestion
	}
	options := []string{"no", "yes"}
	for i, name := range []string{"false", "true"} {
		if !hasNoulDescription(desc[name]) {
			continue
		}
		detail, err := textValue(desc[name])
		if err != nil {
			return "", nil, err
		}
		options[i] += ": " + detail
	}
	return text, options, nil
}

func hasNoulDescription(value any) bool {
	if value == nil {
		return false
	}
	if text, ok := value.(string); ok {
		return text != ""
	}
	return true
}
