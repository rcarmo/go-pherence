package tensorinspect

import (
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"reflect"
	"testing"
)

func TestInspectionOwnsOutputAndStableExamples(t *testing.T) {
	names := []string{"z.layers.weight", "a.embedding", "a.layers.weight"}
	original := append([]string(nil), names...)
	c := InspectTensorNames(names, map[string][]string{"embed": {"EMBEDDING"}, "missing": {"absent"}}, nil)
	if !reflect.DeepEqual(names, original) || c.Examples["layers"] != "a.layers.weight" || c.Readiness.Ready || len(c.Readiness.MissingRequired) != 1 {
		t.Fatal(c, names)
	}
	infos := map[string]safetensors.TensorInfo{"x": {DType: "F32", Shape: []int{2, 3}}}
	shapes := InspectTensorShapes(infos, nil)
	example := shapes.Examples["other"]
	example.Shape[0] = 99
	if infos["x"].Shape[0] != 2 {
		t.Fatal("shape aliases source")
	}
	var v ShapeValidation
	v.Valid = true
	v.Add("bad")
	if v.Valid || len(v.Issues) != 1 {
		t.Fatal(v)
	}
}
