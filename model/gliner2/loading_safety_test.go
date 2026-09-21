package gliner2

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

type auditTensorSource struct {
	calls  int
	shape  []int
	values []float32
	err    error
}

func (s *auditTensorSource) GetFloat32(string) ([]float32, []int, error) {
	s.calls++
	return s.values, s.shape, s.err
}

func auditBoundaryConfig(t *testing.T) BoundaryHeadConfig {
	t.Helper()
	f, err := os.Open("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	c, err := LoadConfig(f)
	if err != nil {
		t.Fatal(err)
	}
	return c.BoundaryHead
}
func auditDebertaConfig(t *testing.T) DebertaConfig {
	t.Helper()
	raw, err := os.ReadFile("testdata/deberta_encoder_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct{ Config DebertaConfig }
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture.Config
}

func TestWeightReaderRejectsShapeOverflowBeforeSource(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, shape := range [][]int{{maxInt/2 + 1, 4}, {0}, {-1}} {
		s := &auditTensorSource{shape: shape}
		r := weightReader{source: s}
		if r.tensor("bad", shape...) != nil || r.err == nil {
			t.Fatalf("accepted shape %v", shape)
		}
		if s.calls != 0 {
			t.Fatalf("invalid shape reached source: %v", shape)
		}
	}
}

func TestWeightReaderCopiesValidSourceAndPreservesFirstError(t *testing.T) {
	s := &auditTensorSource{shape: []int{2}, values: []float32{1, 2}}
	r := weightReader{source: s}
	got := r.tensor("valid", 2)
	if r.err != nil {
		t.Fatal(r.err)
	}
	got[0] = 9
	if s.values[0] != 1 {
		t.Fatal("borrowed source values")
	}
	r.tensor("wrong", 3)
	first := r.err
	calls := s.calls
	r.tensor("later", 2)
	if first == nil || r.err != first || s.calls != calls {
		t.Fatal("first error not sticky")
	}
}

func TestLoadersStopAllocatingAfterMissingTensor(t *testing.T) {
	bc := auditBoundaryConfig(t)
	bc.BoundaryAttentionLayers = 10000
	bc.BoundaryRefinementLayers = 10000
	dc := auditDebertaConfig(t)
	dc.NumHiddenLayers = 10000
	for name, load := range map[string]func(TensorSource) error{
		"boundary":      func(s TensorSource) error { _, _, err := LoadBoundaryModules(s, 768, bc); return err },
		"deberta":       func(s TensorSource) error { _, err := LoadDeberta(s, dc); return err },
		"shared scorer": func(s TensorSource) error { _, err := LoadSharedPoolScorer(s, 768, bc); return err },
	} {
		t.Run(name, func(t *testing.T) {
			s := &auditTensorSource{err: fmt.Errorf("missing weight")}
			allocations := testing.AllocsPerRun(2, func() {
				if err := load(s); err == nil {
					t.Fatal("missing tensor accepted")
				}
			})
			// A failed initial lookup must not allocate thousands of empty layer structs.
			if allocations > 100 {
				t.Fatalf("failed load continued allocating: %.0f", allocations)
			}
			if s.calls != 3 {
				t.Fatalf("expected one source call per attempt, got %d", s.calls)
			}
		})
	}
}

func TestBoundaryLoaderRejectsUnrepresentableProjectionWidths(t *testing.T) {
	base := auditBoundaryConfig(t)
	for _, mult := range []float64{math.NaN(), math.Inf(1), math.MaxFloat64} {
		c := base
		c.BoundaryFFNMultiplier = mult
		s := &auditTensorSource{err: fmt.Errorf("unexpected source read")}
		if _, _, err := LoadBoundaryModules(s, 768, c); err == nil {
			t.Fatal("accepted", mult)
		}
		if s.calls != 0 {
			t.Fatal("invalid FFN width reached source", mult)
		}
	}
	c := base
	c.BoundaryDim = int(^uint(0)>>1) - 7
	s := &auditTensorSource{err: fmt.Errorf("unexpected source read")}
	if _, _, err := LoadBoundaryModules(s, 768, c); err == nil {
		t.Fatal("accepted overflowing dimension")
	}
	if s.calls != 0 {
		t.Fatal("invalid projection width reached source")
	}
}

func TestDebertaLoaderRejectsRelativeRowOverflowBeforeSource(t *testing.T) {
	c := auditDebertaConfig(t)
	c.PositionBuckets = int(^uint(0)>>1) - 1
	c.MaxRelativePositions = int(^uint(0) >> 1)
	s := &auditTensorSource{err: fmt.Errorf("unexpected source read")}
	if _, err := LoadDeberta(s, c); err == nil {
		t.Fatal("accepted overflowing relative rows")
	}
	if s.calls != 0 {
		t.Fatal("invalid relative rows reached source")
	}
}
