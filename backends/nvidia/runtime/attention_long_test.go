package nvidia

import (
	"fmt"
	"math"
	"testing"
)

func longBuffer(t *testing.T, x []float32) *Buffer {
	t.Helper()
	b, e := Malloc(len(x))
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(b.Free)
	if e = b.Upload(x); e != nil {
		t.Fatal(e)
	}
	return b
}
func longData(n, seed int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32((i*seed)%37-18) * .07
	}
	return x
}
func compareLong(t *testing.T, out *Buffer, want []float32) {
	t.Helper()
	got := make([]float32, len(want))
	if e := out.Download(got); e != nil {
		t.Fatal(e)
	}
	for i, x := range got {
		d := math.Abs(float64(x - want[i]))
		if math.IsNaN(d) || d > 2e-5*math.Max(1, math.Abs(float64(want[i]))) {
			t.Fatalf("row component %d got=%g want=%g delta=%g", i, x, want[i], d)
		}
	}
}
func TestLongCausalAttentionCPU(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	for _, length := range []int{2048, 2049, 4096, 8193, 32768} {
		for _, window := range []int{0, 1024, 2049} {
			t.Run(fmt.Sprintf("n%d_w%d", length, window), func(t *testing.T) {
				const rows, heads, kvheads, dim = 3, 4, 2, 32
				q, k, v := longData(rows*heads*dim, 7), longData(length*kvheads*dim, 11), longData(length*kvheads*dim, 13)
				qb, kb, vb, out := longBuffer(t, q), longBuffer(t, k), longBuffer(t, v), longBuffer(t, make([]float32, len(q)))
				want := causalAttentionCPU(q, k, v, rows, length-rows, length, window, heads, kvheads, dim, .2)
				if e := CausalBatchAttentionBuffer(out, qb, kb, vb, rows, length-rows, length, window, heads, kvheads, dim, .2); e != nil {
					t.Fatal(e)
				}
				compareLong(t, out, want)
			})
		}
	}
}
func TestLongIndependentAttentionIsolation(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const pre, stride, batch, heads, kh, dim = 4096, 3, 2, 4, 2, 32
	q, pk, pv := longData(batch*heads*dim, 7), longData(pre*kh*dim, 11), longData(pre*kh*dim, 13)
	sk, sv := longData(batch*stride*kh*dim, 17), longData(batch*stride*kh*dim, 19)
	qb, pkb, pvb, skb, svb, out := longBuffer(t, q), longBuffer(t, pk), longBuffer(t, pv), longBuffer(t, sk), longBuffer(t, sv), longBuffer(t, make([]float32, len(q)))
	active := longBuffer(t, make([]float32, batch))
	if e := active.UploadUint32([]uint32{1, 0}); e != nil {
		t.Fatal(e)
	}
	for _, start := range []int{0, 3000} {
		want := []float32{}
		for row, b := range []int{1, 0} {
			k := append(append([]float32{}, pk...), sk[b*stride*kh*dim:(b*stride+2)*kh*dim]...)
			v := append(append([]float32{}, pv...), sv[b*stride*kh*dim:(b*stride+2)*kh*dim]...)
			want = append(want, causalAttentionCPU(q[row*heads*dim:(row+1)*heads*dim], k, v, 1, pre+1, pre+2, pre+2-start, heads, kh, dim, .2)...)
		}
		if e := IndependentBranchAttentionBuffer(out, qb, pkb, pvb, skb, svb, active, batch, batch, pre, stride, start, pre+2-start, heads, kh, dim, .2); e != nil {
			t.Fatal(e)
		}
		compareLong(t, out, want)
	}
}
func TestLongBranchedAttentionCPU(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const pre, heads, kh, dim = 4096, 4, 2, 32
	segments := []AttentionSegment{{Length: 3}, {Length: 2, ParentLength: 3}, {Length: 4, ParentLength: 3}, {Length: 1}}
	p, e := NewBranchedRows(segments, pre)
	if e != nil {
		t.Fatal(e)
	}
	defer p.Close()
	q, k, v, pk, pv := longData(p.rows*heads*dim, 7), longData(p.rows*kh*dim, 11), longData(p.rows*kh*dim, 13), longData(pre*kh*dim, 17), longData(pre*kh*dim, 19)
	qb, kb, vb, pkb, pvb, out := longBuffer(t, q), longBuffer(t, k), longBuffer(t, v), longBuffer(t, pk), longBuffer(t, pv), longBuffer(t, make([]float32, len(q)))
	for _, window := range []int{0, 2048, 3000} {
		var want []float32
		off := 0
		for _, s := range segments {
			allk, allv := append([]float32{}, pk...), append([]float32{}, pv...)
			if s.ParentLength > 0 {
				allk = append(allk, k[s.ParentStart*kh*dim:(s.ParentStart+s.ParentLength)*kh*dim]...)
				allv = append(allv, v[s.ParentStart*kh*dim:(s.ParentStart+s.ParentLength)*kh*dim]...)
			}
			allk = append(allk, k[off*kh*dim:(off+s.Length)*kh*dim]...)
			allv = append(allv, v[off*kh*dim:(off+s.Length)*kh*dim]...)
			want = append(want, causalAttentionCPU(q[off*heads*dim:(off+s.Length)*heads*dim], allk, allv, s.Length, pre+s.ParentLength, pre+s.ParentLength+s.Length, window, heads, kh, dim, .2)...)
			off += s.Length
		}
		for repeat := 0; repeat < 3; repeat++ {
			if e := p.Attention(out, qb, kb, vb, pkb, pvb, window, heads, kh, dim, .2); e != nil {
				t.Fatal(e)
			}
			compareLong(t, out, want)
		}
	}
}

func TestLongAttentionScratchChunkBoundary(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const rows, length, heads, kh, dim = 65, 8193, 4, 2, 32
	q, k, v := longData(rows*heads*dim, 7), longData(length*kh*dim, 11), longData(length*kh*dim, 13)
	qb, kb, vb, out := longBuffer(t, q), longBuffer(t, k), longBuffer(t, v), longBuffer(t, make([]float32, len(q)))
	want := causalAttentionCPU(q, k, v, rows, length-rows, length, 0, heads, kh, dim, .2)
	if err := CausalBatchAttentionBuffer(out, qb, kb, vb, rows, length-rows, length, 0, heads, kh, dim, .2); err != nil {
		t.Fatal(err)
	}
	compareLong(t, out, want)
}

func TestLongAttentionWideHeads(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const rows, length, heads, kh, dim = 2, 2049, 2, 1, 512
	q, k, v := longData(rows*heads*dim, 7), longData(length*kh*dim, 11), longData(length*kh*dim, 13)
	qb, kb, vb, out := longBuffer(t, q), longBuffer(t, k), longBuffer(t, v), longBuffer(t, make([]float32, len(q)))
	want := causalAttentionCPU(q, k, v, rows, length-rows, length, 0, heads, kh, dim, .044194)
	if err := CausalBatchAttentionBuffer(out, qb, kb, vb, rows, length-rows, length, 0, heads, kh, dim, .044194); err != nil {
		t.Fatal(err)
	}
	compareLong(t, out, want)
}
