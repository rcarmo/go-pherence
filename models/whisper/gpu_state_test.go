package whisper

import "testing"

func TestNewDecoderStateGPUKeepsCPUCrossAttentionFallback(t *testing.T) {
	cfg := Config{
		EncoderDModel:    4,
		DecoderDModel:    4,
		EncoderLayers:    1,
		DecoderLayers:    1,
		EncoderHeads:     2,
		DecoderHeads:     2,
		HeadDim:          2,
		EncoderFFNDim:    8,
		DecoderFFNDim:    8,
		MaxDecoderLength: 8,
		VocabSize:        16,
	}
	dec := NewDecoder(cfg)
	layer := &dec.Layers[0]
	layer.CrossKWeight = []float32{
		0.1, 0.2, -0.1, 0.0,
		0.0, -0.2, 0.3, 0.1,
		0.2, 0.1, 0.0, -0.3,
		-0.1, 0.0, 0.2, 0.4,
	}
	layer.CrossVWeight = []float32{
		-0.2, 0.1, 0.0, 0.3,
		0.2, -0.1, 0.4, 0.0,
		0.0, 0.3, -0.2, 0.1,
		0.1, 0.0, 0.2, -0.4,
	}
	layer.CrossKBias = []float32{0.01, -0.02, 0.03, -0.04}
	layer.CrossVBias = []float32{-0.03, 0.02, -0.01, 0.04}
	encLen := 3
	encoderOutput := []float32{
		0.1, -0.2, 0.3, 0.4,
		-0.5, 0.6, -0.7, 0.8,
		0.9, -1.0, 1.1, -1.2,
	}

	cpu := NewDecoderState(cfg, encoderOutput, encLen, dec)
	gpu := NewDecoderStateGPU(cfg, encoderOutput, encLen, dec)
	if len(gpu.CrossKHead) != cfg.DecoderLayers || len(gpu.CrossVHead) != cfg.DecoderLayers {
		t.Fatalf("GPU state missing CPU fallback cross-attention slices")
	}
	assertClose(t, gpu.CrossK[0], cpu.CrossK[0], 1e-6)
	assertClose(t, gpu.CrossV[0], cpu.CrossV[0], 1e-6)
	assertClose(t, gpu.CrossKHead[0], cpu.CrossKHead[0], 1e-6)
	assertClose(t, gpu.CrossVHead[0], cpu.CrossVHead[0], 1e-6)
}
