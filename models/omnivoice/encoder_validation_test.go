package omnivoice

import (
	"context"
	"testing"
)

func TestEncoderNilValidation(t *testing.T) {
	var h *Hubert
	if _, err := h.Prepare(1600); err == nil {
		t.Fatal("nil Hubert")
	}
	if _, _, err := h.Extract(context.Background(), nil); err == nil {
		t.Fatal("nil Hubert extract")
	}
	if err := h.ExtractInto(context.Background(), nil, nil); err == nil {
		t.Fatal("nil Hubert into")
	}
	var c *CodecEncoder
	if err := c.Prepare(2400, 3); err == nil {
		t.Fatal("nil codec")
	}
	if _, err := c.EncodeFeatures(context.Background(), nil, nil, 3); err == nil {
		t.Fatal("nil codec encode")
	}
	if err := c.EncodeFeaturesInto(context.Background(), nil, nil, nil, 3); err == nil {
		t.Fatal("nil codec into")
	}
}

func TestCodecIntoNilWithValidShapes(t *testing.T) {
	var c *CodecEncoder
	if err := c.EncodeFeaturesInto(context.Background(), make([]int, 8), make([]float32, 960), make([]float32, 768), 1); err == nil {
		t.Fatal("nil codec accepted")
	}
}

func TestReferenceBufferCapacityReuse(t *testing.T) {
	b := make([]float32, 100)
	p := &b[0]
	b = reuseReferenceBuffer(b, 50)
	b = reuseReferenceBuffer(b, 100)
	if &b[0] != p {
		t.Fatal("buffer capacity not reused")
	}
}
