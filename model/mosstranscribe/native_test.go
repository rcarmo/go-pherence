package mosstranscribe

import (
	"testing"

	basetokenizer "github.com/rcarmo/go-pherence/loader/tokenizer"
	llmmodel "github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/model/whisper"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestNativeGenerateRejectsUnsupportedLimit(t *testing.T) {
	if _, err := (&NativeModel{}).GenerateGreedy(nil, nil, GenerationMaxNewTokens+1); err == nil {
		t.Fatal("accepted generation limit beyond pinned configuration")
	}
}

func TestNativePromptEmbeddingsInsertion(t *testing.T) {
	decoder := &llmmodel.LlamaModel{
		Config: llmmodel.LlamaConfig{HiddenSize: 2, VocabSize: 3},
		EmbedTokens: tensor.FromFloat32([]float32{
			1, 2,
			3, 4,
			5, 6,
		}, []int{3, 2}),
	}
	m := &NativeModel{
		Decoder: decoder,
		Processor: &Processor{
			Tokenizer:    &basetokenizer.Tokenizer{},
			AudioTokenID: 1,
		},
	}
	got, err := m.PromptEmbeddings([]int{0, 1, 2, 1}, []float32{10, 11, 20, 21})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 10, 11, 5, 6, 20, 21}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("embeddings=%v want %v", got, want)
		}
	}
}

func TestClosedAudioModelDropsBorrowedAdaptor(t *testing.T) {
	src := &fakeMOSSSource{}
	a := &AudioBackbone{Encoder: &whisper.Encoder{}, Adaptor: AdaptorWeights{Linear1Weight: []uint16{1}}, source: src}
	m := &NativeModel{Audio: a, Decoder: &llmmodel.LlamaModel{}, Processor: &Processor{}}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if len(a.Adaptor.Linear1Weight) != 0 || a.Encoder != nil || m.Audio != nil || m.Decoder != nil || m.Processor != nil {
		t.Fatal("closed model retained mapped state")
	}
	if m.EnableGPU() {
		t.Fatal("closed model enabled GPU")
	}
	if _, _, err := m.EncodeAudio([]float32{1}); err == nil {
		t.Fatal("closed model encoded audio")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
}
