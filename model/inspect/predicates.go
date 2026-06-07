// Package inspect provides small, type-independent predicates shared by the
// model-family tensor-readiness/fixture inspection layers (e.g. model/lfm2 and
// model/qwen3tts). It imports only the standard library so any model package may
// depend on it without risking an import cycle.
package inspect

import "strings"

// MatrixMatches reports whether shape is a 2-D matrix matching rows×cols in
// either orientation.
func MatrixMatches(shape []int, rows, cols int) bool {
	return len(shape) == 2 && ((shape[0] == rows && shape[1] == cols) || (shape[0] == cols && shape[1] == rows))
}

// IsPlaceholder reports whether a fixture value is an unresolved "pending-" marker.
func IsPlaceholder(value string) bool {
	return strings.HasPrefix(value, "pending-")
}

// AnyTensorMarker reports whether any name contains any of the given markers
// (case-insensitive substring match).
func AnyTensorMarker(names, markers []string) bool {
	for _, name := range names {
		s := strings.ToLower(name)
		for _, marker := range markers {
			if strings.Contains(s, marker) {
				return true
			}
		}
	}
	return false
}
