package whisper

import "testing"

func TestEncoderReleaseHostWeights(t *testing.T) {
	layer := EncoderLayer{
		AttnLNWeight: []float32{1}, AttnLNBias: []float32{1}, QWeight: []float32{1}, QBias: []float32{1},
		KWeight: []float32{1}, KBias: []float32{1}, VWeight: []float32{1}, VBias: []float32{1},
		OWeight: []float32{1}, OBias: []float32{1}, MLPLNWeight: []float32{1}, MLPLNBias: []float32{1},
		FC1Weight: []float32{1}, FC1Bias: []float32{1}, FC2Weight: []float32{1}, FC2Bias: []float32{1},
	}
	e := &Encoder{Conv1Weight: []float32{1}, Conv1Bias: []float32{1}, Conv2Weight: []float32{1}, Conv2Bias: []float32{1}, PosEmbed: []float32{1}, FinalLNWeight: []float32{1}, FinalLNBias: []float32{1}, Layers: []EncoderLayer{layer}}
	e.ReleaseHostWeights()
	e.ReleaseHostWeights()
	if e.Conv1Weight != nil || e.PosEmbed != nil || e.FinalLNWeight != nil || e.Layers != nil {
		t.Fatal("host weights retained")
	}
	var nilEncoder *Encoder
	nilEncoder.ReleaseHostWeights()
}
