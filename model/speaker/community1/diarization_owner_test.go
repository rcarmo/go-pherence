package community1

import (
	"context"
	"testing"
)

func TestExperimentalDiarizationOwnedValidationRelease(t *testing.T) {
	m := &ExperimentalDiarization{segmentation: &ExperimentalSegmentation{checkpoint: &SegmentationCheckpoint{recurrent: &LSTM{}, head: &SegmentationHead{}}, frontend: &SincNet{}}, embedding: &ExperimentalEmbedding{model: &WeSpeakerResNet34{cfg: WeSpeakerResNetConfig{EmbedDim: 2}}}, plda: &PreparedPLDA{cfg: PLDAConfig{InputDim: 2}}}
	if e := m.ValidateOwned(); e != nil {
		t.Fatal(e)
	}
	m.ReleaseOwnedModels()
	m.ReleaseOwnedModels()
	if e := m.ValidateOwned(); e == nil {
		t.Fatal("released graph valid")
	}
	var nilModel *ExperimentalDiarization
	nilModel.ReleaseOwnedModels()
	if nilModel.ValidateOwned() == nil {
		t.Fatal("nil valid")
	}
	mismatch := &ExperimentalDiarization{segmentation: &ExperimentalSegmentation{checkpoint: &SegmentationCheckpoint{recurrent: &LSTM{}, head: &SegmentationHead{}}, frontend: &SincNet{}}, embedding: &ExperimentalEmbedding{model: &WeSpeakerResNet34{cfg: WeSpeakerResNetConfig{EmbedDim: 2}}}, plda: &PreparedPLDA{cfg: PLDAConfig{InputDim: 3}}}
	if mismatch.ValidateOwned() == nil {
		t.Fatal("dimension mismatch")
	}
	if _, e := NewExperimentalDiarization(context.Background(), m.segmentation, m.embedding, m.plda); e == nil {
		t.Fatal("released input")
	}
}
