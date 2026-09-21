package community1

import (
	"context"
	"testing"
)

func TestReleaseVulkanHostWeights(t *testing.T) {
	seg, _ := experimentalFixture(t)
	emb, _ := embeddingPCMFixture(t)
	if _, err := seg.checkpoint.ForwardFeatures(context.Background(), make([]float32, 3*seg.checkpoint.cfg.LSTM.InputSize), 3, LSTMScalar, HeadScalar); err != nil {
		t.Fatal(err)
	}
	if _, err := emb.model.FrameShape(9); err != nil {
		t.Fatal(err)
	}
	ReleaseVulkanHostWeights(seg.checkpoint, emb.model)
	ReleaseVulkanHostWeights(seg.checkpoint, emb.model)
	if len(seg.checkpoint.sincnet.LowHz) != 0 || len(seg.checkpoint.recurrent.layers) != 0 || len(seg.checkpoint.head.layers) != 0 || len(seg.checkpoint.head.classifier.Weight) != 0 {
		t.Fatal("segmentation host tensors retained")
	}
	if len(emb.model.stem) != 0 || len(emb.model.stages[0]) != 0 || len(emb.model.projection.Weight) != 0 {
		t.Fatal("embedding host tensors retained")
	}
	if _, err := seg.checkpoint.ForwardFeatures(context.Background(), make([]float32, 3*seg.checkpoint.cfg.LSTM.InputSize), 3, LSTMScalar, HeadScalar); err == nil {
		t.Fatal("released segmentation remained executable")
	}
	if _, err := emb.model.FrameShape(9); err == nil {
		t.Fatal("released embedding remained executable")
	}
	ReleaseVulkanHostWeights(nil, nil)
}
