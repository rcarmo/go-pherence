package whisper

import (
	"context"
	"errors"
	"testing"
)

func TestVulkanEncoderScoreILPRejectsBeforeDevice(t *testing.T) {
	enc := vulkanToyEncoder(t, vulkanToyConfig())
	if x, err := NewVulkanEncoderTile64Key32ScoreILP(nil, enc, 17); x != nil || err == nil {
		t.Fatal("nil context")
	}
	if x, err := NewVulkanEncoderTile64Key32ScoreILP(context.Background(), nil, 17); x != nil || err == nil {
		t.Fatal("nil source")
	}
	if x, err := NewVulkanEncoderTile64Key32ScoreILP(context.Background(), enc, 0); x != nil || err == nil {
		t.Fatal("invalid frames")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if x, err := NewVulkanEncoderTile64Key32ScoreILP(ctx, enc, 17); x != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("cancel", x, err)
	}
	for _, name := range []string{"layer0.fc1.w", "layer0.k.w", "layer0.q.w"} {
		if vulkanQ8WeightSelected(vulkanLinearF32Tile64Key32ScoreILP, name) {
			t.Fatal("unexpected quantisation", name)
		}
	}
	if vulkanDefaultLinearMode != vulkanLinearF32RegTile {
		t.Fatal("default changed")
	}
}
