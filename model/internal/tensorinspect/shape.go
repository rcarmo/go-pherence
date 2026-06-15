package tensorinspect

// ShapeValidation is the shared result of a per-tensor shape check: a validity
// flag plus a list of human-readable issues.
type ShapeValidation struct {
	Valid  bool     `json:"valid"`
	Issues []string `json:"issues,omitempty"`
}

// Add records an issue and marks the result invalid.
func (v *ShapeValidation) Add(issue string) {
	v.Valid = false
	v.Issues = append(v.Issues, issue)
}
