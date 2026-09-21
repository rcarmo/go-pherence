package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestPublishedClassificationParity(t *testing.T) {
	dir := os.Getenv("GLINER_MODEL_DIR")
	if dir == "" {
		t.Skip("set GLINER_MODEL_DIR")
	}
	var ref struct {
		Text, Task string
		Labels     []string
		IDs        []int
		Logits     []float32
	}
	raw, err := os.ReadFile("testdata/published_classification_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	m, err := LoadEntityModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	input, err := m.Tokenizer.PrepareClassification(ref.Text, ref.Task, ref.Labels, 512)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(input.IDs, ref.IDs) {
		t.Fatalf("routing mismatch %v %v", input.IDs, ref.IDs)
	}
	got, err := m.Classify(ref.Text, ref.Task, ref.Labels, 512)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got.Logits {
		if math.Abs(float64(v-ref.Logits[i])) > 2e-3 {
			t.Fatalf("%d got %g want %g", i, v, ref.Logits[i])
		}
	}
	t.Logf("classification logits %v", got.Logits)
}
