package whisper

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"testing"
)

func TestOriginalQ5SourceIdentity(t *testing.T) {
	raw, err := os.ReadFile("../../backends/vulkan/testdata/q5-grouped-input.bin")
	if err != nil {
		t.Fatal(err)
	}
	ref, err := os.ReadFile("../../backends/vulkan/testdata/q5-grouped-output.f32")
	if err != nil {
		t.Fatal(err)
	}
	values := make([]float32, len(ref)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(ref[i*4:]))
	}
	if err := checkOriginalQ5Values(context.Background(), raw, values); err != nil {
		t.Fatal(err)
	}
	values[17] += 1
	if err := checkOriginalQ5Values(context.Background(), raw, values); err == nil {
		t.Fatal("checkpoint differs")
	}
	for _, r := range [][]byte{nil, raw[:21]} {
		if err := checkOriginalQ5Values(context.Background(), r, values); err == nil {
			t.Fatal("extent")
		}
	}
	if err := checkOriginalQ5Values(nil, raw, values); err == nil {
		t.Fatal("nilctx")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := checkOriginalQ5Values(ctx, raw, values); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if x, err := NewVulkanEncoderOriginalQ5MLP(context.Background(), nil, 1, nil); x != nil || err == nil {
		t.Fatal("nil file")
	}
	for _, name := range []string{"layer0.fc1.w", "layer31.fc2.w"} {
		if !isVulkanMLPWeight(name) {
			t.Fatal(name)
		}
	}
	if isVulkanMLPWeight("layer0.q.w") {
		t.Fatal("placement")
	}
}
