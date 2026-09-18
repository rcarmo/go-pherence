package community1

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"
)

type referenceCase struct {
	Speakers     int       `json:"speakers"`
	MaxActive    int       `json:"max_active"`
	Classes      int       `json:"classes"`
	Frames       int       `json:"frames"`
	Scores       []float32 `json:"log_scores"`
	Hard, Soft   []float32
	Permutations []struct{ Slots, Classes []int }
}

func powersetReference(t *testing.T) []referenceCase {
	t.Helper()
	data, err := os.ReadFile("testdata/powerset-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema    int
		Cases     []referenceCase
		Tolerance float64 `json:"soft_absolute_tolerance"`
		Reference struct {
			SHA string `json:"sha256"`
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || len(fixture.Cases) != 4 || fixture.Tolerance != 1e-6 || fixture.Reference.SHA != "7eeb5691c20337bd24462f6a4ee2e56e279564bfef7875776df0a41f245b0325" {
		t.Fatal("reference contract changed")
	}
	return fixture.Cases
}
func closePowerset(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("output shape")
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.Abs(float64(v-want[i])) > tolerance {
			t.Fatalf("element%d got %.9g want %.9g", i, v, want[i])
		}
	}
}

func TestPowersetPinnedOracle(t *testing.T) {
	for _, c := range powersetReference(t) {
		p, err := NewPowerset(c.Speakers, c.MaxActive)
		if err != nil {
			t.Fatal(err)
		}
		if p.Speakers() != c.Speakers || p.Classes() != c.Classes {
			t.Fatal("geometry")
		}
		original := append([]float32(nil), c.Scores...)
		hard, err := p.Decode(context.Background(), c.Scores, c.Frames, PowersetHard)
		if err != nil {
			t.Fatal(err)
		}
		closePowerset(t, hard, c.Hard, 0)
		soft, err := p.Decode(context.Background(), c.Scores, c.Frames, PowersetSoft)
		if err != nil {
			t.Fatal(err)
		}
		closePowerset(t, soft, c.Soft, 1e-6)
		if !reflect.DeepEqual(original, c.Scores) {
			t.Fatal("mutated input")
		}
		for _, permutation := range c.Permutations {
			mapping, err := p.ClassPermutation(permutation.Slots)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(mapping, permutation.Classes) {
				t.Fatalf("permutation convention reversed: slots%v got%v want%v", permutation.Slots, mapping, permutation.Classes)
			}
			reordered := make([]float32, len(c.Scores))
			for frame := 0; frame < c.Frames; frame++ {
				for column, index := range mapping {
					reordered[frame*c.Classes+column] = c.Scores[frame*c.Classes+index]
				}
			}
			decoded, err := p.Decode(context.Background(), reordered, c.Frames, PowersetSoft)
			if err != nil {
				t.Fatal(err)
			}
			want := make([]float32, len(decoded))
			for frame := 0; frame < c.Frames; frame++ {
				for slot, index := range permutation.Slots {
					want[frame*c.Speakers+slot] = soft[frame*c.Speakers+index]
				}
			}
			closePowerset(t, decoded, want, 1e-6)
		}
	}
}

func TestPowersetClassOrderTiesAndImpossibleClasses(t *testing.T) {
	p, _ := NewPowerset(3, 2)
	// Empty, A, B, C, AB, AC, BC; not binary-mask integer order.
	want := []uint16{0, 1, 2, 4, 3, 5, 6}
	if !reflect.DeepEqual(p.masks, want) {
		t.Fatalf("class order %v", p.masks)
	}
	for winner, mask := range want {
		row := make([]float32, 7)
		for i := range row {
			row[i] = float32(math.Inf(-1))
		}
		row[winner] = 0
		got, err := p.Decode(context.Background(), row, 1, PowersetHard)
		if err != nil {
			t.Fatal(err)
		}
		for slot := 0; slot < 3; slot++ {
			expected := float32(0)
			if mask&(1<<slot) != 0 {
				expected = 1
			}
			if got[slot] != expected {
				t.Fatal("hard membership")
			}
		}
		soft, err := p.Decode(context.Background(), row, 1, PowersetSoft)
		if err != nil {
			t.Fatal(err)
		}
		closePowerset(t, soft, got, 0)
	}
	tied := []float32{-5, 0, 0, -5, -5, -5, -5}
	got, err := p.Decode(context.Background(), tied, 1, PowersetHard)
	if err != nil {
		t.Fatal(err)
	}
	closePowerset(t, got, []float32{1, 0, 0}, 0)
	got, err = p.Decode(context.Background(), make([]float32, 7), 1, PowersetHard)
	if err != nil {
		t.Fatal(err)
	}
	closePowerset(t, got, []float32{0, 0, 0}, 0)
	// Deliberately unnormalised log probabilities: exp(0)=1 for all seven
	// classes; each local speaker occurs in three. No extra softmax allowed.
	got, err = p.Decode(context.Background(), make([]float32, 7), 1, PowersetSoft)
	if err != nil {
		t.Fatal(err)
	}
	closePowerset(t, got, []float32{3, 3, 3}, 0)
}

func TestPowersetRejectsMalformedInput(t *testing.T) {
	for _, args := range [][2]int{{0, 0}, {9, 1}, {3, 0}, {3, -1}, {3, 4}, {int(^uint(0) >> 1), 1}} {
		if p, err := NewPowerset(args[0], args[1]); err == nil || p != nil {
			t.Fatal("bad constructor geometry")
		}
	}
	p, _ := NewPowerset(3, 2)
	for _, tc := range []struct {
		frames int
		mode   PowersetMode
		scores []float32
	}{
		{-1, PowersetHard, nil}, {MaxPowersetFrames + 1, PowersetHard, nil}, {int(^uint(0) >> 1), PowersetHard, nil}, {1, PowersetHard, make([]float32, 6)}, {0, PowersetHard, []float32{0}}, {1, PowersetMode(2), make([]float32, 7)},
	} {
		if out, err := p.Decode(context.Background(), tc.scores, tc.frames, tc.mode); err == nil || out != nil {
			t.Fatal("accepted malformed input")
		}
	}
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		scores := make([]float32, 14)
		scores[13] = value
		for _, mode := range []PowersetMode{PowersetHard, PowersetSoft} {
			if out, err := p.Decode(context.Background(), scores, 2, mode); err == nil || out != nil {
				t.Fatal("nonfinite score")
			}
		}
	}
	scores := make([]float32, 7)
	scores[0] = 1
	if _, err := p.Decode(context.Background(), scores, 1, PowersetSoft); err == nil {
		t.Fatal("positive log probability")
	}
	if _, err := p.Decode(context.Background(), scores, 1, PowersetHard); err != nil {
		t.Fatal("hard logits disallowed")
	}
	for i := range scores {
		scores[i] = float32(math.Inf(-1))
	}
	for _, mode := range []PowersetMode{PowersetHard, PowersetSoft} {
		if out, err := p.Decode(context.Background(), scores, 1, mode); err == nil || out != nil {
			t.Fatal("all impossible classes")
		}
	}
	var zero Powerset
	var missing *Powerset
	for _, p := range []*Powerset{&zero, missing} {
		if out, err := p.Decode(context.Background(), nil, 0, PowersetHard); err == nil || out != nil {
			t.Fatal("zero/nil powerset")
		}
		if _, err := p.ClassPermutation(nil); err == nil {
			t.Fatal("nil permutation")
		}
	}
	if missing.Classes() != 0 || missing.Speakers() != 0 {
		t.Fatal("nil metadata")
	}
	for _, slots := range [][]int{nil, {0, 1}, {0, 1, 1}, {0, 1, 3}, {0, 1, -1}} {
		if _, err := p.ClassPermutation(slots); err == nil {
			t.Fatal("bad permutation")
		}
	}
}

type powersetCancelContext struct {
	context.Context
	cancel    context.CancelFunc
	at, calls int
}

func (c *powersetCancelContext) Err() error {
	c.calls++
	if c.at > 0 && c.calls == c.at {
		c.cancel()
	}
	return c.Context.Err()
}
func newPowersetContext(at int) *powersetCancelContext {
	ctx, cancel := context.WithCancel(context.Background())
	return &powersetCancelContext{Context: ctx, cancel: cancel, at: at}
}

func TestPowersetCancellationBoundsAndOwnership(t *testing.T) {
	p, _ := NewPowerset(3, 2)
	scores := make([]float32, 21)
	for _, mode := range []PowersetMode{PowersetHard, PowersetSoft} {
		count := newPowersetContext(0)
		if _, err := p.Decode(count, scores, 3, mode); err != nil {
			t.Fatal(err)
		}
		count.cancel()
		if count.calls != 8 {
			t.Fatalf("expected pre/post and two checks per row got%d", count.calls)
		}
		for at := 1; at <= count.calls; at++ {
			ctx := newPowersetContext(at)
			out, err := p.Decode(ctx, scores, 3, mode)
			ctx.cancel()
			if !errors.Is(err, context.Canceled) || out != nil {
				t.Fatal("partial output/cancel lost")
			}
		}
	}
	out, err := p.Decode(context.Background(), nil, 0, PowersetSoft)
	if err != nil || len(out) != 0 {
		t.Fatal("empty batch")
	}
	largest, _ := NewPowerset(8, 8)
	if largest.Classes() != 256 {
		t.Fatal("largest class count")
	}
	out, err = largest.Decode(context.Background(), make([]float32, MaxPowersetFrames*256), MaxPowersetFrames, PowersetHard)
	if err != nil || len(out) != MaxPowersetFrames*8 {
		t.Fatal("bounded max batch")
	}
	slots := []int{2, 0, 1}
	mapping, err := p.ClassPermutation(slots)
	if err != nil {
		t.Fatal(err)
	}
	mapping[0] = -1
	slots[0] = -1
	again, err := p.ClassPermutation([]int{2, 0, 1})
	if err != nil || again[0] != 0 {
		t.Fatal("returned map aliases immutable powerset")
	}
}
