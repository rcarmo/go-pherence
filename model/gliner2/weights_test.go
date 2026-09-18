package gliner2

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

type shapeSource map[string][]int

func (s shapeSource) GetFloat32(name string) ([]float32, []int, error) {
	shape, ok := s[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing %s", name)
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	return make([]float32, n), shape, nil
}
func TestPublishedBoundaryBindings(t *testing.T) {
	f, err := os.Open("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	c, err := LoadConfig(f)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/boundary_shapes.json")
	if err != nil {
		t.Fatal(err)
	}
	var s shapeSource
	if err = json.Unmarshal(raw, &s); err != nil {
		t.Fatal(err)
	}
	e, h, err := LoadBoundaryModules(s, 768, c.BoundaryHead)
	if err != nil {
		t.Fatal(err)
	}
	if len(e.AttentionBlocks) != 2 || len(e.RefinementBlocks) != 1 || h.StartQuery.InDim != 768 {
		t.Fatal("binding mismatch")
	}
	delete(s, "boundary_head.boundary_encoder.bos_state")
	if _, _, err = LoadBoundaryModules(s, 768, c.BoundaryHead); err == nil {
		t.Fatal("missing tensor accepted")
	}
	s["boundary_head.boundary_encoder.bos_state"] = []int{1}
	if _, _, err = LoadBoundaryModules(s, 768, c.BoundaryHead); err == nil {
		t.Fatal("wrong shape accepted")
	}
}
