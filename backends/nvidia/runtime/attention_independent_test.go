package nvidia

import (
	"math"
	"testing"
)

func TestIndependentBranchAttentionMatchesPerBranchGPUOracle(t *testing.T) {
	if !SgemmReady() {
		if Available() {
			t.Fatal("CUDA device is available but PTX runtime is not ready")
		}
		t.Skip("CUDA unavailable")
	}
	const batch, seq, heads, kvHeads, dim = 3, 5, 4, 2, 8
	qDim, kvDim := heads*dim, kvHeads*dim
	q := make([]float32, batch*qDim)
	k := make([]float32, batch*seq*kvDim)
	v := make([]float32, batch*seq*kvDim)
	for i := range q {
		q[i] = float32((i*11)%29-14) * 0.031
	}
	for i := range k {
		k[i] = float32((i*7)%31-15) * 0.027
	}
	for i := range v {
		v[i] = float32((i*13)%37-18) * 0.019
	}
	// All branches share the same immutable two-row trunk; only suffix rows
	// differ. The per-branch oracle below uses this same topology.
	for b := 1; b < batch; b++ {
		copy(k[b*seq*kvDim:b*seq*kvDim+2*kvDim], k[:2*kvDim])
		copy(v[b*seq*kvDim:b*seq*kvDim+2*kvDim], v[:2*kvDim])
	}
	qb, _ := Malloc(len(q))
	kb, _ := Malloc(len(k))
	vb, _ := Malloc(len(v))
	ob, _ := Malloc(len(q))
	defer qb.Free()
	defer kb.Free()
	defer vb.Free()
	defer ob.Free()
	if err := qb.Upload(q); err != nil {
		t.Fatal(err)
	}
	if err := kb.Upload(k); err != nil {
		t.Fatal(err)
	}
	if err := vb.Upload(v); err != nil {
		t.Fatal(err)
	}
	const scale = float32(0.5)
	active, _ := Malloc(batch)
	defer active.Free()
	if err := active.UploadUint32([]uint32{0, 1, 2}); err != nil {
		t.Fatal(err)
	}
	trunkElems := 2 * kvDim
	trunkK, _ := Malloc(trunkElems)
	trunkV, _ := Malloc(trunkElems)
	suffixK, _ := Malloc(batch * (seq - 2) * kvDim)
	suffixV, _ := Malloc(batch * (seq - 2) * kvDim)
	defer trunkK.Free()
	defer trunkV.Free()
	defer suffixK.Free()
	defer suffixV.Free()
	_ = trunkK.Upload(k[:trunkElems])
	_ = trunkV.Upload(v[:trunkElems])
	sk := make([]float32, batch*(seq-2)*kvDim)
	sv := make([]float32, len(sk))
	for b := 0; b < batch; b++ {
		copy(sk[b*(seq-2)*kvDim:], k[b*seq*kvDim+trunkElems:(b+1)*seq*kvDim])
		copy(sv[b*(seq-2)*kvDim:], v[b*seq*kvDim+trunkElems:(b+1)*seq*kvDim])
	}
	_ = suffixK.Upload(sk)
	_ = suffixV.Upload(sv)
	if err := IndependentBranchAttentionBuffer(ob, qb, trunkK, trunkV, suffixK, suffixV, active, batch, batch, 2, seq-2, 0, seq, heads, kvHeads, dim, scale); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(q))
	if err := ob.Download(got); err != nil {
		t.Fatal(err)
	}
	for b := 0; b < batch; b++ {
		want := make([]float32, qDim)
		if err := F32GQAAttention(want, q[b*qDim:(b+1)*qDim], k[b*seq*kvDim:(b+1)*seq*kvDim], v[b*seq*kvDim:(b+1)*seq*kvDim], seq, heads, kvHeads, dim, scale); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if diff := math.Abs(float64(got[b*qDim+i] - want[i])); diff > 2e-6 {
				t.Fatalf("branch=%d value=%d got=%g want=%g diff=%g", b, i, got[b*qDim+i], want[i], diff)
			}
		}
	}
}

func TestIndependentBranchAttentionHasNoCrossBranchVisibility(t *testing.T) {
	if !SgemmReady() {
		if Available() {
			t.Fatal("CUDA device is available but PTX runtime is not ready")
		}
		t.Skip("CUDA unavailable")
	}
	const batch, seq, heads, kvHeads, dim = 2, 3, 2, 1, 4
	qDim, kvDim := heads*dim, kvHeads*dim
	q := make([]float32, batch*qDim)
	k := make([]float32, batch*seq*kvDim)
	v := make([]float32, batch*seq*kvDim)
	for i := range q {
		q[i] = float32(i+1) * 0.03
	}
	for i := range k {
		k[i] = float32(i-5) * 0.02
	}
	for i := range v {
		v[i] = float32(i+2) * 0.04
	}
	run := func(values []float32) []float32 {
		qb, _ := Malloc(len(q))
		kb, _ := Malloc(len(k))
		vb, _ := Malloc(len(values))
		ob, _ := Malloc(len(q))
		defer qb.Free()
		defer kb.Free()
		defer vb.Free()
		defer ob.Free()
		_ = qb.Upload(q)
		_ = kb.Upload(k)
		_ = vb.Upload(values)
		active, _ := Malloc(batch)
		defer active.Free()
		_ = active.UploadUint32([]uint32{0, 1})
		trunkK, _ := Malloc(kvDim)
		trunkV, _ := Malloc(kvDim)
		suffixK, _ := Malloc(batch * (seq - 1) * kvDim)
		suffixV, _ := Malloc(batch * (seq - 1) * kvDim)
		defer trunkK.Free()
		defer trunkV.Free()
		defer suffixK.Free()
		defer suffixV.Free()
		_ = trunkK.Upload(k[:kvDim])
		_ = trunkV.Upload(values[:kvDim])
		sk := make([]float32, batch*(seq-1)*kvDim)
		sv := make([]float32, len(sk))
		for b := 0; b < batch; b++ {
			copy(sk[b*(seq-1)*kvDim:], k[b*seq*kvDim+kvDim:(b+1)*seq*kvDim])
			copy(sv[b*(seq-1)*kvDim:], values[b*seq*kvDim+kvDim:(b+1)*seq*kvDim])
		}
		_ = suffixK.Upload(sk)
		_ = suffixV.Upload(sv)
		if err := IndependentBranchAttentionBuffer(ob, qb, trunkK, trunkV, suffixK, suffixV, active, batch, batch, 1, seq-1, 0, seq, heads, kvHeads, dim, 1); err != nil {
			t.Fatal(err)
		}
		out := make([]float32, len(q))
		_ = ob.Download(out)
		return out
	}
	first := run(v)
	for i := seq * kvDim; i < len(v); i++ {
		v[i] += 1000
	}
	second := run(v)
	for i := 0; i < qDim; i++ {
		if first[i] != second[i] {
			t.Fatalf("branch 0 changed at %d: %g vs %g", i, first[i], second[i])
		}
	}
}
