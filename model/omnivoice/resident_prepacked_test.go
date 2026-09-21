package omnivoice

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

type residentPrepackedFixture struct {
	config    loader.Config
	tokens    int
	ids       []int
	audioMask []bool
}

func (f residentPrepackedFixture) logitsLen(tokens int) int {
	return f.config.NumAudioCodebook * tokens * f.config.AudioVocabSize
}

func loadResidentPrepackedFixture(t *testing.T) (*Backbone, residentPrepackedFixture) {
	t.Helper()
	dir := t.TempDir()
	cfg := loader.Config{
		Architectures:        []string{"OmniVoice"},
		AudioCodebookWeights: []int{1, 1},
		AudioMaskID:          7,
		AudioVocabSize:       8,
		DType:                "float32",
		EOSTokenID:           1,
		LLMConfig: loader.LLMConfig{
			Architectures:         []string{"Qwen3ForCausalLM"},
			AttentionBias:         false,
			AttentionDropout:      0,
			ChunkSizeFeedForward:  0,
			DType:                 "float32",
			EOSTokenID:            1,
			HeadDim:               8,
			HiddenAct:             "silu",
			HiddenSize:            32,
			InitializerRange:      0.02,
			IntermediateSize:      48,
			LayerTypes:            []string{"full_attention", "full_attention"},
			MaxPositionEmbeddings: 64,
			MaxWindowLayers:       2,
			ModelType:             "qwen3",
			NumAttentionHeads:     4,
			NumHiddenLayers:       2,
			NumKeyValueHeads:      2,
			RMSNormEps:            1e-6,
			RopeParameters:        loader.RopeParameters{RopeTheta: 10000, RopeType: "default"},
			TieWordEmbeddings:     false,
			UseCache:              true,
			UseSlidingWindow:      false,
			VocabSize:             32,
		},
		ModelType:           "omnivoice",
		NumAudioCodebook:    2,
		TransformersVersion: "test",
	}
	writeSyntheticResidentModel(t, dir, cfg)
	w, err := loader.OpenWeights(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = w.Close() })
	b, err := NewBackbone(w, 8)
	if err != nil {
		t.Fatal(err)
	}
	fixture := residentPrepackedFixture{
		config: cfg,
		tokens: 8,
		ids: []int{
			1, 3, 5, 7, 0, 2, 4, 6,
			2, 4, 6, 0, 1, 3, 5, 7,
		},
		audioMask: []bool{false, true, true, false, true, true, false, true},
	}
	return b, fixture
}

func writeSyntheticResidentModel(t *testing.T, dir string, cfg loader.Config) {
	t.Helper()
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	shapes := loader.ExpectedShapes(cfg)
	if shapes == nil {
		t.Fatal("expected shapes missing")
	}
	names := make([]string, 0, len(shapes))
	for name := range shapes {
		names = append(names, name)
	}
	sort.Strings(names)
	header := map[string]any{"__metadata__": map[string]any{}}
	payload := make([]byte, 0)
	offset := 0
	for _, name := range names {
		shape := shapes[name]
		tensor := syntheticResidentTensorBytes(t, cfg, name, shape)
		dtype := "F16"
		if name == "codebook_layer_offsets" {
			dtype = "I64"
		}
		header[name] = map[string]any{
			"dtype":        dtype,
			"shape":        shape,
			"data_offsets": []int{offset, offset + len(tensor)},
		}
		payload = append(payload, tensor...)
		offset += len(tensor)
	}
	rawHeader, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, uint64(len(rawHeader)))
	buf = append(buf, rawHeader...)
	buf = append(buf, payload...)
	if err = os.WriteFile(filepath.Join(dir, "model.safetensors"), buf, 0o600); err != nil {
		t.Fatal(err)
	}
}

func syntheticResidentTensorBytes(t *testing.T, cfg loader.Config, name string, shape []int64) []byte {
	t.Helper()
	numel := 1
	for _, dim := range shape {
		if dim <= 0 {
			t.Fatalf("invalid shape for %s: %v", name, shape)
		}
		numel *= int(dim)
	}
	if name == "codebook_layer_offsets" {
		buf := make([]byte, numel*8)
		for i := 0; i < numel; i++ {
			binary.LittleEndian.PutUint64(buf[i*8:], uint64(i*cfg.AudioVocabSize))
		}
		return buf
	}
	seed := 0
	for _, ch := range name {
		seed += int(ch)
	}
	buf := make([]byte, numel*2)
	for i := 0; i < numel; i++ {
		value := float32((i+seed)%19-9) * 0.03125
		if strings.Contains(name, "norm.weight") {
			value = 1 + float32((i+seed)%7-3)*0.03125
		}
		binary.LittleEndian.PutUint16(buf[i*2:], half.F32ToF16(value))
	}
	return buf
}

func TestBackboneResidentPrepackedParityAndAccounting(t *testing.T) {
	rawBackbone, f := loadResidentPrepackedFixture(t)
	prepackedBackbone, err := NewBackbone(rawBackbone.weights, f.tokens)
	if err != nil {
		t.Fatal(err)
	}
	rawRequired := rawBackbone.ResidentRequiredBytes()
	totalRequired := rawBackbone.ResidentPrepackedRequiredBytes()
	if totalRequired <= rawRequired {
		t.Fatalf("prepacked bytes not reported raw=%d total=%d", rawRequired, totalRequired)
	}
	if err = rawBackbone.EnableResident(context.Background(), rawRequired); err != nil {
		t.Fatal(err)
	}
	if err = prepackedBackbone.EnableResidentPrepacked(context.Background(), totalRequired); err != nil {
		t.Fatal(err)
	}
	if prepackedBackbone.PrepackedBytes() != totalRequired-rawRequired {
		t.Fatalf("prepacked bytes %d want %d", prepackedBackbone.PrepackedBytes(), totalRequired-rawRequired)
	}
	if prepackedBackbone.ResidentBytes() != totalRequired {
		t.Fatalf("resident bytes %d want %d", prepackedBackbone.ResidentBytes(), totalRequired)
	}
	want := make([]float32, f.logitsLen(f.tokens))
	got := make([]float32, len(want))
	if err = rawBackbone.ForwardInto(context.Background(), want, f.ids, f.audioMask, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err = prepackedBackbone.ForwardInto(context.Background(), got, f.ids, f.audioMask, nil, nil); err != nil {
		t.Fatal(err)
	}
	assertFloat32Exact(t, got, want)
}

func TestBackboneResidentPrepackedReconfigureParityAndZeroAllocs(t *testing.T) {
	b, f := loadResidentPrepackedFixture(t)
	fullWant := make([]float32, f.logitsLen(f.tokens))
	if err := b.ForwardInto(context.Background(), fullWant, f.ids, f.audioMask, nil, nil); err != nil {
		t.Fatal(err)
	}
	books, vocab := f.config.NumAudioCodebook, f.config.AudioVocabSize
	ids4 := sliceBookMajor(f.ids, books, f.tokens, 4)
	audio4 := append([]bool(nil), f.audioMask[:4]...)
	stream4, err := NewBackbone(b.weights, 4)
	if err != nil {
		t.Fatal(err)
	}
	want4 := make([]float32, books*4*vocab)
	if err = stream4.ForwardInto(context.Background(), want4, ids4, audio4, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err = b.EnableResidentPrepacked(context.Background(), b.ResidentPrepackedRequiredBytes()); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		tokens int
		ids    []int
		audio  []bool
		want   []float32
	}{
		{tokens: 4, ids: ids4, audio: audio4, want: want4},
		{tokens: f.tokens, ids: f.ids, audio: f.audioMask, want: fullWant},
		{tokens: 4, ids: ids4, audio: audio4, want: want4},
	} {
		if err = b.Reconfigure(tc.tokens); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, len(tc.want))
		if err = b.ForwardInto(context.Background(), got, tc.ids, tc.audio, nil, nil); err != nil {
			t.Fatal(err)
		}
		assertFloat32Exact(t, got, tc.want)
	}
	out4 := make([]float32, len(want4))
	out8 := make([]float32, len(fullWant))
	var allocErr error
	if n := testing.AllocsPerRun(10, func() {
		if allocErr = b.Reconfigure(4); allocErr != nil {
			return
		}
		allocErr = b.ForwardInto(context.Background(), out4, ids4, audio4, nil, nil)
		if allocErr != nil {
			return
		}
		if allocErr = b.Reconfigure(f.tokens); allocErr != nil {
			return
		}
		allocErr = b.ForwardInto(context.Background(), out8, f.ids, f.audioMask, nil, nil)
	}); allocErr != nil {
		t.Fatal(allocErr)
	} else if n != 0 {
		t.Fatalf("prepacked reuse allocations %g", n)
	}
}

func TestBackboneResidentPrepackedBudgetUpgradeTransactional(t *testing.T) {
	b, _ := loadResidentPrepackedFixture(t)
	rawRequired := b.ResidentRequiredBytes()
	totalRequired := b.ResidentPrepackedRequiredBytes()
	if err := b.EnableResident(context.Background(), rawRequired); err != nil {
		t.Fatal(err)
	}
	rawResident := b.resident
	if err := b.EnableResidentPrepacked(context.Background(), totalRequired-1); err == nil {
		t.Fatal("resident prepacked cache accepted undersized budget")
	}
	if b.resident != rawResident {
		t.Fatal("budget failure replaced raw resident cache")
	}
	if b.PrepackedBytes() != 0 || b.ResidentBytes() != rawRequired {
		t.Fatalf("budget failure left bytes raw=%d packed=%d total=%d", rawRequired, b.PrepackedBytes(), b.ResidentBytes())
	}
}

func TestBackboneResidentPrepackedCancelUpgradeAndSiblingInheritance(t *testing.T) {
	b, f := loadResidentPrepackedFixture(t)
	if err := b.EnableResident(context.Background(), b.ResidentRequiredBytes()); err != nil {
		t.Fatal(err)
	}
	rawResident := b.resident
	older, err := NewBackboneSibling(b, 4)
	if err != nil {
		t.Fatal(err)
	}
	if older.resident != rawResident {
		t.Fatal("existing sibling did not inherit raw resident cache")
	}
	ctx := &cancelAfterChecks{Context: context.Background(), remaining: 3}
	if err = b.EnableResidentPrepacked(ctx, b.ResidentPrepackedRequiredBytes()); err != context.Canceled {
		t.Fatalf("expected cancellation, got %v", err)
	}
	if b.resident != rawResident || b.PrepackedBytes() != 0 {
		t.Fatal("cancellation mutated resident cache")
	}
	if older.resident != rawResident || older.PrepackedBytes() != 0 {
		t.Fatal("cancellation mutated sibling resident cache")
	}
	if err = b.EnableResidentPrepacked(context.Background(), b.ResidentPrepackedRequiredBytes()); err != nil {
		t.Fatal(err)
	}
	if b.resident == rawResident || b.PrepackedBytes() == 0 {
		t.Fatal("resident prepacked upgrade did not replace cache")
	}
	if older.resident != rawResident || older.PrepackedBytes() != 0 {
		t.Fatal("existing sibling upgraded retroactively")
	}
	newer, err := NewBackboneSibling(b, 4)
	if err != nil {
		t.Fatal(err)
	}
	if newer.resident != b.resident || newer.PrepackedBytes() != b.PrepackedBytes() {
		t.Fatal("new sibling did not inherit prepacked resident cache")
	}
	ids4 := sliceBookMajor(f.ids, f.config.NumAudioCodebook, f.tokens, 4)
	audio4 := append([]bool(nil), f.audioMask[:4]...)
	want := make([]float32, f.logitsLen(4))
	got := make([]float32, len(want))
	if err = older.ForwardInto(context.Background(), want, ids4, audio4, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err = newer.ForwardInto(context.Background(), got, ids4, audio4, nil, nil); err != nil {
		t.Fatal(err)
	}
	assertFloat32Exact(t, got, want)
}

func TestResidentPrepackedByteAccountingMatchesManualShapeMath(t *testing.T) {
	b, f := loadResidentPrepackedFixture(t)
	shapes := loader.ExpectedShapes(f.config)
	if shapes == nil {
		t.Fatal("expected shapes missing")
	}
	var rawPerLayer, packedPerLayer int64
	for _, suffix := range residentLayerSuffixes {
		shape := shapes[fmt.Sprintf("llm.layers.0.%s", suffix)]
		elems := int64(1)
		for _, dim := range shape {
			elems *= dim
		}
		rawPerLayer += elems
	}
	for _, suffix := range []string{
		"self_attn.q_proj.weight",
		"self_attn.k_proj.weight",
		"self_attn.v_proj.weight",
		"self_attn.o_proj.weight",
		"mlp.gate_proj.weight",
		"mlp.up_proj.weight",
		"mlp.down_proj.weight",
	} {
		shape := shapes[fmt.Sprintf("llm.layers.0.%s", suffix)]
		packedPerLayer += (shape[0] / 16) * shape[1] * 16
	}
	layers := int64(f.config.LLMConfig.NumHiddenLayers)
	rawWant := rawPerLayer * 4 * layers
	packedWant := packedPerLayer * 4 * layers
	if b.ResidentRequiredBytes() != rawWant {
		t.Fatalf("raw bytes %d want %d", b.ResidentRequiredBytes(), rawWant)
	}
	if b.ResidentPrepackedRequiredBytes() != rawWant+packedWant {
		t.Fatalf("prepacked total bytes %d want %d", b.ResidentPrepackedRequiredBytes(), rawWant+packedWant)
	}
}
