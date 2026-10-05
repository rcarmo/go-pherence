package whisper

import (
	"context"
	"errors"
	"fmt"
	legacy "github.com/rcarmo/go-pherence/loader/whisperggml"
	"testing"
)

func TestPackedOnlyCheckedSourceContract(t *testing.T) {
	c := checkedLoadConfig()
	c.EncoderLayers = 1
	c.DecoderLayers = 1
	source := checkedLoadFixture(c, "F32")
	model, err := loadModelSourceCheckedMode(context.Background(), source, c, true)
	if err != nil {
		t.Fatal(err)
	}
	skipped := 0
	for name := range source.infos {
		if isEncoderFFNSourceName(name) {
			skipped++
		}
	}
	if skipped != 2 {
		t.Fatal("selected weights", skipped)
	}
	if source.reads != len(source.infos)-skipped {
		t.Fatal("read count", source.reads)
	}
	if model.Encoder.Layers[0].FC1Weight != nil || model.Encoder.Layers[0].FC2Weight != nil {
		t.Fatal("fake tensors")
	}
	if err = model.validatePCMModel(); err == nil {
		t.Fatal("partial host model admitted")
	}
	if err = model.validatePCMModelForEncoder(true); err != nil {
		t.Fatal("decoder admission", err)
	}
	// Bad skipped metadata still fails before the first source read.
	for _, arm := range []string{"shape", "dtype", "missing", "overlap"} {
		s := checkedLoadFixture(c, "F32")
		name := "model.encoder.layers.0.fc1.weight"
		info := s.infos[name]
		switch arm {
		case "shape":
			info.Shape = []int{1}
		case "dtype":
			info.DType = "I8"
		case "missing":
			delete(s.infos, name)
		case "overlap":
			info.DataOffsets = [2]int{0, info.DataOffsets[1] - info.DataOffsets[0]}
		}
		if arm != "missing" {
			s.infos[name] = info
		}
		if x, e := loadModelSourceCheckedMode(context.Background(), s, c, true); x != nil || e == nil || s.reads != 0 {
			t.Fatal("metadata before load", arm, e, s.reads)
		}
	}
	s := checkedLoadFixture(c, "F32")
	ctx, cancel := context.WithCancel(context.Background())
	s.afterRead = cancel
	if x, e := loadModelSourceCheckedMode(ctx, s, c, true); x != nil || !errors.Is(e, context.Canceled) {
		t.Fatal("cancel", e)
	}
	if _, e := loadModelSourceCheckedMode(nil, source, c, true); e == nil {
		t.Fatal("nilctx")
	}
}
func TestPackedOnlyVulkanLayoutContract(t *testing.T) {
	c := vulkanToyConfig()
	e := vulkanToyEncoder(t, c)
	for i := range e.Layers {
		e.Layers[i].FC1Weight = nil
		e.Layers[i].FC2Weight = nil
	}
	if _, err := describeVulkanEncoder(context.Background(), e, 17); err == nil {
		t.Fatal("default permits missing weight")
	}
	layout, err := describeVulkanEncoderMode(context.Background(), e, 17, true)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, weights := range layout.weights {
		for _, w := range weights {
			if isVulkanMLPWeight(w.name) {
				count++
				if len(w.data) != 0 || len(w.shape) != 2 {
					t.Fatal(w)
				}
			}
		}
	}
	if count != c.EncoderLayers*2 {
		t.Fatal(count)
	}
	e.Layers[0].FC1Weight = []float32{1}
	if _, err := describeVulkanEncoderMode(context.Background(), e, 17, true); err == nil {
		t.Fatal("borrowed CPU weights")
	}
	if _, _, err := LoadOriginalQ5ResidentChecked(context.Background(), nil, c); err == nil {
		t.Fatal("nil file")
	}
	if _, _, err := LoadOriginalQ5ResidentChecked(nil, nil, c); err == nil {
		t.Fatal("nilctx")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := LoadOriginalQ5ResidentChecked(ctx, nil, c); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := checkOriginalQ5FFNMetadata((*legacy.File)(nil), c); err == nil {
		t.Fatal("nil metadata")
	}
}
func TestOriginalQ5MetadataAdmission(t *testing.T) {
	c := checkedLoadConfig()
	c.EncoderDModel = 32
	c.DecoderDModel = 32
	c.EncoderFFNDim = 64
	c.DecoderFFNDim = 64
	c.EncoderHeads = 1
	c.DecoderHeads = 1
	c.HeadDim = 32
	var tensors []legacy.Tensor
	for layer := 0; layer < c.EncoderLayers; layer++ {
		for _, m := range []struct{ part, k, n int }{{0, 32, 64}, {2, 64, 32}} {
			tensors = append(tensors, legacy.Tensor{Name: fmt.Sprintf("encoder.blocks.%d.mlp.%d.weight", layer, m.part), Type: 6, Shape: []int{m.k, m.n}, Bytes: int64(m.k / 32 * m.n * 22)})
		}
	}
	if err := checkOriginalQ5FFNTensors(tensors, c); err != nil {
		t.Fatal(err)
	}
	for _, arm := range []string{"type", "shape", "bytes", "missing"} {
		bad := append([]legacy.Tensor(nil), tensors...)
		bad[0].Shape = append([]int(nil), bad[0].Shape...)
		switch arm {
		case "type":
			bad[0].Type = 1
		case "shape":
			bad[0].Shape[0] = 31
		case "bytes":
			bad[0].Bytes--
		case "missing":
			bad = bad[1:]
		}
		if err := checkOriginalQ5FFNTensors(bad, c); err == nil {
			t.Fatal("metadata", arm)
		}
	}
	bad := c
	bad.EncoderLayers = 0
	if err := checkOriginalQ5FFNTensors(tensors, bad); err == nil {
		t.Fatal("layers")
	}
	bad = c
	bad.EncoderDModel = 0
	if err := checkOriginalQ5FFNTensors(tensors, bad); err == nil {
		t.Fatal("config")
	}
}

func TestPackedSourceNamePlacement(t *testing.T) {
	for _, name := range []string{"model.encoder.layers.0.fc1.weight", "model.encoder.layers.31.fc2.weight"} {
		if !isEncoderFFNSourceName(name) {
			t.Fatal(name)
		}
	}
	for _, name := range []string{"model.decoder.layers.0.fc1.weight", "model.encoder.layers.0.fc1.bias", "model.encoder.layers.-1.fc1.weight", "model.encoder.layers.01.fc1.weight", "model.encoder.layers.0.self_attn.q_proj.weight"} {
		if isEncoderFFNSourceName(name) {
			t.Fatal(name)
		}
	}
}
