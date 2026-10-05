package training

import "context"

// ResidentDeviceScaleDriver uses the read-only four-scalar global norm result;
// scale is element 3. All six buffers must be disjoint. GradientScale must be 1:
// a device scale replaces it, rather than multiplying two clipping decisions.
// Validation admits only metadata/constants, never reads device contents.
// The coordinator validates diagnostics before mutation; errors retain owners.
type ResidentDeviceScaleDriver interface {
	ValidateScaledUpdate(context.Context, ResidentBuffer, ResidentBuffer, ResidentBuffer, ResidentBuffer, ResidentBuffer, ResidentBuffer, int, AdamWStep) error
	UpdateScaled(context.Context, ResidentBuffer, ResidentBuffer, ResidentBuffer, ResidentBuffer, ResidentBuffer, ResidentBuffer, int, AdamWStep) error
}

// NewDeviceScaleResidentState selects device global norm and direct scale-buffer
// AdamW consumption. Four diagnostics still download for fail-closed admission;
// bias corrections, post-update checks and coherent snapshots remain on host.
// It owns the same twelve extra F32 scalars as NewGlobalNormResidentState.
func NewDeviceScaleResidentState(ctx context.Context, driver ResidentDriver, snapshot ResidentSnapshot, maxBytes int64) (*ResidentState, error) {
	return newResidentState(ctx, driver, snapshot, maxBytes, true, true, false)
}
