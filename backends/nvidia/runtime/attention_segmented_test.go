package nvidia

import (
	"math"
	"testing"
)

func TestSegmentedAttentionMatchesCPU(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const prefix, heads, kvheads, dim = 57, 4, 2, 256
	lengths := []int{1, 24, 7, 33}
	rows := 0
	for _, n := range lengths {
		rows += n
	}
	plan, err := NewSegmentedRows(lengths, prefix)
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	q := make([]float32, rows*heads*dim)
	k, v := make([]float32, rows*kvheads*dim), make([]float32, rows*kvheads*dim)
	pk, pv := make([]float32, prefix*kvheads*dim), make([]float32, prefix*kvheads*dim)
	for i := range q {
		q[i] = float32((i*7)%23-11) * .31
	}
	for i := range k {
		k[i] = float32((i*13)%31-15) * .21
		v[i] = float32((i*17)%29-14) * .2
	}
	for i := range pk {
		pk[i] = float32((i*11)%37-18) * .17
		pv[i] = float32((i*19)%41-20) * .16
	}
	alloc := func(x []float32) *Buffer {
		b, err := Malloc(len(x))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(b.Free)
		if err = b.Upload(x); err != nil {
			t.Fatal(err)
		}
		return b
	}
	qb, kb, vb, pkb, pvb, ob := alloc(q), alloc(k), alloc(v), alloc(pk), alloc(pv), alloc(make([]float32, len(q)))
	for _, window := range []int{0, 8, 64} {
		want := make([]float32, 0, len(q))
		off := 0
		for _, n := range lengths {
			allk := append(append([]float32{}, pk...), k[off*kvheads*dim:(off+n)*kvheads*dim]...)
			allv := append(append([]float32{}, pv...), v[off*kvheads*dim:(off+n)*kvheads*dim]...)
			want = append(want, causalAttentionCPU(q[off*heads*dim:(off+n)*heads*dim], allk, allv, n, prefix, prefix+n, window, heads, kvheads, dim, .0625)...)
			off += n
		}
		for repeat := 0; repeat < 5; repeat++ {
			if err := plan.Attention(ob, qb, kb, vb, pkb, pvb, window, heads, kvheads, dim, .0625); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(q))
			if err := ob.Download(got); err != nil {
				t.Fatal(err)
			}
			for i, x := range got {
				d := math.Abs(float64(x - want[i]))
				if math.IsNaN(d) || d > 2e-5*math.Max(1, math.Abs(float64(want[i]))) {
					t.Fatalf("window=%d row=%d diff=%g", window, i/(heads*dim), d)
				}
			}
		}
	}
}

func TestSegmentedRowsValidation(t *testing.T) {
	for _, lengths := range [][]int{nil, {0}, {-1}, {513}, {256, 257}} {
		if p, err := NewSegmentedRows(lengths, 57); err == nil {
			p.Close()
			t.Fatal("invalid lengths accepted")
		}
	}
	for _, prefix := range []int{0, -1, 2048} {
		if p, err := NewSegmentedRows([]int{1}, prefix); err == nil {
			p.Close()
			t.Fatal("invalid prefix accepted")
		}
	}
	var p *SegmentedRows
	if err := p.RoPE(nil, nil, 1, 32, 16); err == nil {
		t.Fatal("nil plan accepted")
	}
	if err := p.Attention(nil, nil, nil, nil, nil, nil, 0, 1, 1, 32, 1); err == nil {
		t.Fatal("nil plan accepted")
	}
}

func TestBranchedRowsRejectInvalidParents(t *testing.T) {
	for _, s := range [][]AttentionSegment{
		{{Length: 2, ParentStart: 0, ParentLength: 1}},
		{{Length: 2}, {Length: 3, ParentStart: 1, ParentLength: 1}},
		{{Length: 2}, {Length: 3, ParentStart: 0, ParentLength: 3}},
		{{Length: 2}, {Length: 3, ParentStart: 0, ParentLength: 2}, {Length: 1, ParentStart: 2, ParentLength: 3}},
	} {
		if p, err := NewBranchedRows(s, 57); err == nil {
			p.Close()
			t.Fatal("invalid parent accepted")
		}
	}
}
