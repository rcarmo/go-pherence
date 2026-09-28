package decider

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestTemperatureByTypeResolution(t *testing.T) {
	q := []renderedQuestion{{typeName: Choice}, {typeName: Noul}, {typeName: Score, legend: []string{"low", "medium", "high"}}}
	rows := planRows(q)
	if len(rows) != 5 {
		t.Fatalf("rows=%d want=5", len(rows))
	}
	r := &Runtime{Temperature: 1.03}
	wantScalar := []float32{1.03, 1.03, 1.03, 1.03, 1.03}
	for i, row := range rows {
		if got := r.temperatureForType(q[row.owner].typeName); got != wantScalar[i] {
			t.Fatalf("scalar row %d temperature=%g want=%g", i, got, wantScalar[i])
		}
	}
	r.temperatureByType = map[QuestionType]float32{Noul: 2.22, Score: 1.38}
	wantTyped := []float32{1.03, 2.22, 1.38, 1.38, 1.38}
	for i, row := range rows {
		if got := r.temperatureForType(q[row.owner].typeName); got != wantTyped[i] {
			t.Fatalf("typed row %d temperature=%g want=%g", i, got, wantTyped[i])
		}
	}
	logits := [][]float32{{1, 2}, {-2, 0}, {1, 3}, {0, 2}, {2, 4}}
	got := make([][]float32, len(rows))
	for i, row := range rows {
		var err error
		got[i], err = softmax(logits[i], r.temperatureForType(q[row.owner].typeName))
		if err != nil {
			t.Fatal(err)
		}
		want, err := softmax(logits[i], wantTyped[i])
		if err != nil || !reflect.DeepEqual(got[i], want) {
			t.Fatalf("row %d got=%v want=%v err=%v", i, got[i], want, err)
		}
		// Independent logistic calculation checks that the selected value
		// reaches the numerical softmax, not only the temperature resolver.
		delta := float64(logits[i][1]-logits[i][0]) / float64(wantTyped[i])
		expected := 1 / (1 + math.Exp(-delta))
		if math.Abs(float64(got[i][1])-expected) > 1e-7 {
			t.Fatalf("row %d probability=%g want=%g", i, got[i][1], expected)
		}
	}
	if got[1][1] == got[3][1] {
		t.Fatal("per-type temperature did not affect equal-logit-gap probabilities")
	}
	if got[2][1] != got[3][1] {
		t.Fatal("isolated Score rows did not share Score temperature")
	}
}

func TestTemperatureByTypeValidation(t *testing.T) {
	for _, tc := range []map[QuestionType]float32{
		{"bool": 1}, {"scale": 1}, {"Choice": 1},
		{Choice: 0}, {Noul: -1}, {Score: float32(math.NaN())}, {Choice: float32(math.Inf(1))},
	} {
		if err := validateTemperaturesByType(tc); err == nil || !strings.Contains(err.Error(), "temperature_by_type") {
			t.Fatalf("accepted invalid map %v: %v", tc, err)
		}
	}
	if err := validateTemperaturesByType(nil); err != nil {
		t.Fatal(err)
	}
	if err := validateTemperaturesByType(map[QuestionType]float32{Choice: .9, Noul: 2, Score: 1.4}); err != nil {
		t.Fatal(err)
	}
}

func TestTemperatureByTypeFixtureLoad(t *testing.T) {
	dir := tinyDir(t)
	configPath := filepath.Join(dir, "decider_config.json")
	base, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	withMap := strings.Replace(string(base), `"temperature":1.03`, `"temperature":1.03,"temperature_by_type":{"noul":2.22,"score":1.38}`, 1)
	if err := os.WriteFile(configPath, []byte(withMap), 0600); err != nil {
		t.Fatal(err)
	}
	r, err := LoadFixture(dir, 32)
	if err != nil {
		t.Fatal(err)
	}
	if r.temperatureForType(Choice) != 1.03 || r.temperatureForType(Noul) != 2.22 || r.temperatureForType(Score) != 1.38 {
		t.Fatal("fixture per-type temperature fallback mismatch")
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	for _, item := range []string{`"bool":2`, `"noul":0`, `"score":-1`, `"choice":"2"`, `"choice":true`, `"choice":null`} {
		bad := strings.Replace(string(base), `"temperature":1.03`, `"temperature":1.03,"temperature_by_type":{`+item+`}`, 1)
		if err := os.WriteFile(configPath, []byte(bad), 0600); err != nil {
			t.Fatal(err)
		}
		if r, err := LoadFixture(dir, 32); err == nil {
			_ = r.Close()
			t.Fatalf("accepted invalid config temperature_by_type %s", item)
		}
	}
}
