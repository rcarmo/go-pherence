package whisper

import "context"

// A nil internal context preserves the legacy non-cancellable paths. Public
// context APIs require a non-nil context, as the standard library does.
func speechContextErr(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}
