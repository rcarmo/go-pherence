package mojev

import (
	"math"
	"reflect"
	"sync"
	"testing"
)

func smallHead(t *testing.T) *HeadWeights {
	t.Helper()
	gamma := []float32{1, 1}
	beta := []float32{0, 0}
	context := []float32{1, 0}
	option := []float32{1, 0}
	head, err := NewHeadWeights(2, 1, gamma, beta, context, option)
	if err != nil {
		t.Fatal(err)
	}
	gamma[0] = 90
	context[0] = 90
	if head.NormWeight[0] != 1 || head.ContextProjection[0] != 1 {
		t.Fatal("head aliased caller weights")
	}
	return head
}

func TestScoreHiddenSynthetic(t *testing.T) {
	head := smallHead(t)
	hidden := []float32{2, 0, 2, 0, 2, 0, 0, 2, 0, 0}
	state := []bool{true, false, false, false, false}
	questions := [][]bool{{false, true, false, false, false}}
	candidates := [][][]bool{{{false, false, true, false, false}, {false, false, false, true, false}}}
	mask := [][]bool{{true, true}}
	got, err := head.ScoreHidden(hidden, state, questions, candidates, mask)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || len(got[0]) != 2 || math.Abs(float64(got[0][0]-2)) > 1e-4 || math.Abs(float64(got[0][1]+2)) > 1e-4 {
		t.Fatalf("unexpected synthetic logits %v", got)
	}
	got[0][0] = -100
	if hidden[0] != 2 || !state[0] || !questions[0][1] || !candidates[0][0][2] || !mask[0][0] {
		t.Fatal("mutated input")
	}
	if got[0][1] == -100 {
		t.Fatal("output aliases sibling")
	}
	masked, err := head.ScoreHidden(hidden, state, questions, candidates, [][]bool{{true, false}})
	if err != nil || math.Float32bits(masked[0][1]) != math.Float32bits(-math.MaxFloat32) {
		t.Fatalf("masked option: %v %v", masked, err)
	}
	// Repeated and concurrent calls must not retain request scratch.
	var group sync.WaitGroup
	for i := 0; i < 8; i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			v, err := head.ScoreHidden(hidden, state, questions, candidates, mask)
			if err != nil || math.Abs(float64(v[0][0]-2)) > 1e-4 || math.Abs(float64(v[0][1]+2)) > 1e-4 {
				t.Errorf("concurrent score %v %v", v, err)
			}
		}()
	}
	group.Wait()
}

func TestHeadRejectsMalformed(t *testing.T) {
	for name, tc := range map[string]struct{ gamma, beta, context, option []float32 }{
		"short norm":           {[]float32{1}, []float32{0, 0}, []float32{1, 0}, []float32{1, 0}},
		"short projection":     {[]float32{1, 1}, []float32{0, 0}, []float32{1}, []float32{1, 0}},
		"nonfinite norm":       {[]float32{1, float32(math.NaN())}, []float32{0, 0}, []float32{1, 0}, []float32{1, 0}},
		"nonfinite projection": {[]float32{1, 1}, []float32{0, 0}, []float32{1, 0}, []float32{1, float32(math.Inf(1))}},
	} {
		t.Run(name, func(t *testing.T) {
			v, err := NewHeadWeights(2, 1, tc.gamma, tc.beta, tc.context, tc.option)
			if err == nil || v != nil {
				t.Fatalf("accepted %v", v)
			}
		})
	}
	if v, err := NewHeadWeights(0, 1, nil, nil, nil, nil); err == nil || v != nil {
		t.Fatal("accepted zero width")
	}
	if v, err := NewHeadWeights(4097, 1, nil, nil, nil, nil); err == nil || v != nil {
		t.Fatal("accepted overlong width")
	}
	if v, err := NewHeadWeights(2, 4097, nil, nil, nil, nil); err == nil || v != nil {
		t.Fatal("accepted overlong rank")
	}
	head := smallHead(t)
	state := []bool{true, false, false}
	q := [][]bool{{false, true, false}}
	c := [][][]bool{{{false, false, true}}}
	hidden := []float32{2, 0, 0, 2, 2, 0}
	for name, tc := range map[string]struct {
		hidden []float32
		state  []bool
		q      [][]bool
		c      [][][]bool
		mask   [][]bool
	}{
		"short hidden":    {hidden[:4], state, q, c, [][]bool{{true}}},
		"nan hidden":      {[]float32{2, 0, 0, float32(math.NaN()), 2, 0}, state, q, c, [][]bool{{true}}},
		"infinite hidden": {[]float32{2, 0, 0, float32(math.Inf(1)), 2, 0}, state, q, c, [][]bool{{true}}},
		"overlap":         {hidden, []bool{true, true, false}, q, c, [][]bool{{true}}},
		"bad mask width":  {hidden, state, q, c, [][]bool{{}}},
		"bad mask count":  {hidden, state, q, c, nil},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := head.ScoreHidden(tc.hidden, tc.state, tc.q, tc.c, tc.mask)
			if err == nil || got != nil {
				t.Fatalf("accepted malformed request: %v", got)
			}
		})
	}
	if got, err := (*HeadWeights)(nil).ScoreHidden(hidden, state, q, c, [][]bool{{true}}); err == nil || got != nil {
		t.Fatal("accepted nil head")
	}
	if got, err := (&HeadWeights{}).ScoreHidden(hidden, state, q, c, [][]bool{{true}}); err == nil || got != nil {
		t.Fatal("accepted empty head")
	}
	corrupt := *head
	corrupt.NormWeight = append([]float32(nil), head.NormWeight...)
	corrupt.NormWeight[0] = float32(math.NaN())
	if got, err := corrupt.ScoreHidden(hidden, state, q, c, [][]bool{{true}}); err == nil || got != nil {
		t.Fatal("accepted mutated non-finite weights")
	}
	before := append([]float32(nil), hidden...)
	if _, err := head.ScoreHidden(hidden, state, q, c, [][]bool{{false}}); err != nil || !reflect.DeepEqual(hidden, before) {
		t.Fatalf("mutated caller %v", err)
	}
}
