package mojev

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

// ValidateTextControls rejects caller text that the released tokenizer would
// interpret as a reserved control token. It does not parse a request, open a
// checkpoint, or run inference. The upstream server does not enforce this gate.
// Validate the tokenizer asset hash before calling it with untrusted input.
func ValidateTextControls(req TextRequest, tok *tokenizer.Tokenizer) error {
	if tok == nil || req.State == "" || len(req.Fields) == 0 || len(req.Fields) > 256 {
		return fmt.Errorf("mojev: invalid text control boundary")
	}
	check := func(label, text string) error {
		if err := tok.ValidateUserText(text); err != nil {
			return fmt.Errorf("mojev: %s: %w", label, err)
		}
		return nil
	}
	if err := check("state", req.State); err != nil {
		return err
	}
	for i, field := range req.Fields {
		if field.ID == "" || len(field.Options) < 2 || len(field.Options) > 64 || len(field.Keys) != len(field.Options) {
			return fmt.Errorf("mojev: invalid field %d", i)
		}
		for _, item := range []struct{ label, text string }{{"question ID", field.ID}, {"instructions", field.Instructions}} {
			if err := check(item.label, item.text); err != nil {
				return err
			}
		}
		for _, option := range field.Keys {
			if err := check("criterion ID", option); err != nil {
				return err
			}
		}
		for _, option := range field.Options {
			if err := check("candidate", option); err != nil {
				return err
			}
		}
	}
	return nil
}
