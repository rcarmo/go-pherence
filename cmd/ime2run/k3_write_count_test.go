package main

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

func TestK3I8I4M1WritesHowMany(t *testing.T) {
	const subs = 1
	a := make([]byte, subs*38)
	b := make([]byte, subs*608)
	*(*float32)(unsafe.Pointer(&a[0])) = 1.0
	*(*int16)(unsafe.Pointer(&a[4])) = int16(-32)
	for i := 0; i < 32; i++ { a[6+i] = 1 }
	for r := 0; r < 32; r++ {
		b[r*2] = 0x00; b[r*2+1] = 0x3C
		b[64+r] = 8
		for n := 0; n < 16; n++ { b[96+r*16+n] = 0x88 }
	}
	sentinel := math.Float32frombits(0x7FC00001)
	out := make([]float32, 64)
	for i := range out { out[i] = sentinel }

	pool := NewAIWorkerPool(1)
	defer pool.Close()
	pool.Run(func(workerID, nWorkers int) {
		k3I8I4M1(&a[0], &b[0], &out[0], subs, 32)
	})

	written := 0
	for i := 0; i < 64; i++ {
		if math.Float32bits(out[i]) != math.Float32bits(sentinel) { written++ }
	}
	fmt.Printf("k3I8I4M1 wrote %d output values\n", written)
	fmt.Printf("out[0:8]  = %v\n", out[:8])
	fmt.Printf("out[8:16] = %v\n", out[8:16])
	if written < 8 { t.Errorf("wrote only %d (< 8)", written) }
}

func TestGoAsmVsRef(t *testing.T) {
	const M, K = 64, 256
	subs := K / 32
	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ { raw[m*K+k] = int8((m*3 + k*7 + 5) & 15) }
		for sb := 0; sb < subs; sb++ {
			scales[m*subs+sb] = 0.01 + float32(m+sb)*0.001
			mins[m*subs+sb] = 0.08 + float32(m+sb)*0.002  // positive dmin
		}
	}
	w := repackQ4KToQ41x32(M, K, raw, scales, mins)
	if !w.Valid { t.Fatal("repack failed") }

	act := make([]float32, K)
	for i := range act { act[i] = float32(i+1) * 0.01 }

	// F32 reference
	outRef := make([]float32, M)
	q4kQ41x32MatVecRef(w, act, outRef)

	// GoAsm path
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	outGoAsm := make([]float32, M)
	q4kQ41x32MatVecGoAsmWithCorrection(w, nil, act, outGoAsm, pool)

	maxDiff := float32(0)
	for i := range outRef {
		d := outRef[i] - outGoAsm[i]
		if d < 0 { d = -d }
		if d > maxDiff { maxDiff = d }
	}
	t.Logf("GoAsm vs Ref maxDiff=%.6f", maxDiff)
	if maxDiff > 0.15 { // 0.08 HP kernel precision error is expected
		t.Errorf("GoAsm diverges from Ref: maxDiff=%.6f", maxDiff)
		t.Logf("ref[0:4]=%v", outRef[:4])
		t.Logf("goasm[0:4]=%v", outGoAsm[:4])
	}
}

func TestBatchVsRef(t *testing.T) {
	const M, K = 64, 256
	subs := K / 32
	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ { raw[m*K+k] = int8((m*3 + k*7 + 5) & 15) }
		for sb := 0; sb < subs; sb++ {
			scales[m*subs+sb] = 0.01 + float32(m+sb)*0.001
			mins[m*subs+sb] = 0.08 + float32(m+sb)*0.002
		}
	}
	w := repackQ4KToQ41x32(M, K, raw, scales, mins)
	act := make([]float32, K)
	for i := range act { act[i] = float32(i+1) * 0.01 }

	outRef := make([]float32, M)
	q4kQ41x32MatVecRef(w, act, outRef)

	pool := NewAIWorkerPool(6)
	defer pool.Close()
	outBatch := make([]float32, M)
	ok := q4kQ41x32MatVecBatchSameAct(act, pool, q4kBatchMatVecSpec{W: w, Out: outBatch})
	if !ok { t.Fatal("batch returned false") }

	maxDiff := float32(0)
	for i := range outRef {
		d := outRef[i] - outBatch[i]
		if d < 0 { d = -d }
		if d > maxDiff { maxDiff = d }
	}
	t.Logf("Batch vs Ref maxDiff=%.6f", maxDiff)
	if maxDiff > 0.5 {
		t.Errorf("Batch diverges: maxDiff=%.6f", maxDiff)
		t.Logf("ref[0:4]=%v", outRef[:4])
		t.Logf("batch[0:4]=%v", outBatch[:4])
	}
}

func TestK3I8I4M1WriteDetail(t *testing.T) {
	const subs = 1
	a := make([]byte, subs*38)
	b := make([]byte, subs*608)
	*(*float32)(unsafe.Pointer(&a[0])) = 1.0
	*(*int16)(unsafe.Pointer(&a[4])) = int16(-32)
	for i := 0; i < 32; i++ { a[6+i] = 1 }
	for r := 0; r < 32; r++ {
		b[r*2] = 0x00; b[r*2+1] = 0x3C
		b[64+r] = 8
		for n := 0; n < 16; n++ { b[96+r*16+n] = 0x88 }
	}
	sentinel := math.Float32frombits(0x7FC00001)
	out := make([]float32, 64)
	for i := range out { out[i] = sentinel }

	pool := NewAIWorkerPool(1)
	defer pool.Close()
	pool.Run(func(workerID, nWorkers int) {
		k3I8I4M1(&a[0], &b[0], &out[0], subs, 32)
	})
	
	t.Logf("out[0:32] = %v", out[:32])
	nonSentinel := 0
	for i := 0; i < 64; i++ {
		if math.Float32bits(out[i]) != math.Float32bits(sentinel) { 
			nonSentinel++
			if nonSentinel <= 10 { t.Logf("  wrote out[%d]=%v", i, out[i]) }
		}
	}
	t.Logf("Total non-sentinel: %d", nonSentinel)
}

func TestK3I8I4M1ExactRange(t *testing.T) {
	const subs = 1
	a := make([]byte, subs*38)
	b := make([]byte, subs*608)
	*(*float32)(unsafe.Pointer(&a[0])) = 1.0
	*(*int16)(unsafe.Pointer(&a[4])) = int16(-32)
	for i := 0; i < 32; i++ { a[6+i] = 1 }
	for r := 0; r < 32; r++ {
		b[r*2] = 0x00; b[r*2+1] = 0x3C
		b[64+r] = 8
		for n := 0; n < 16; n++ { b[96+r*16+n] = 0x88 }
	}

	// Use a large sentinel block, mark different regions
	out := make([]float32, 64)
	for i := range out { out[i] = math.Float32frombits(0x7FC00001) }
	// Mark second half with different sentinel to distinguish
	for i := 8; i < 16; i++ { out[i] = math.Float32frombits(0x7FC00002) }
	for i := 16; i < 32; i++ { out[i] = math.Float32frombits(0x7FC00003) }
	
	pool := NewAIWorkerPool(1)
	defer pool.Close()
	pool.Run(func(workerID, nWorkers int) {
		k3I8I4M1(&a[0], &b[0], &out[0], subs, 32)
	})
	
	// Count values changed from their respective sentinels
	changed0_7 := 0; changed8_15 := 0; changed16_31 := 0
	for i := 0; i < 8; i++ { if math.Float32bits(out[i]) != 0x7FC00001 { changed0_7++ } }
	for i := 8; i < 16; i++ { if math.Float32bits(out[i]) != 0x7FC00002 { changed8_15++ } }
	for i := 16; i < 32; i++ { if math.Float32bits(out[i]) != 0x7FC00003 { changed16_31++ } }
	t.Logf("changed: [0:8]=%d [8:16]=%d [16:32]=%d", changed0_7, changed8_15, changed16_31)
	t.Logf("out[0:16]=%v", out[:16])
}

func TestAICoreVLENViaCGo(t *testing.T) {
	// The C aicore_vlen binary can't be run from AI core via taskset
	// Instead, probe VLEN by running our kernel and measuring output count
	// A 32-element store means VLEN=1024 (e32,m1 gives vl=32)
	// An 8-element store means VLEN=256 (e32,m1 gives vl=8)
	
	pool := NewAIWorkerPool(1)
	defer pool.Close()
	
	// Create minimal Q8 activation (subs=4 to make kBlks manageable)
	const kBlks = 4
	a := make([]byte, kBlks*38)
	b := make([]byte, kBlks*608)
	for k := 0; k < kBlks; k++ {
		*(*float32)(unsafe.Pointer(&a[k*38])) = 1.0
		*(*int16)(unsafe.Pointer(&a[k*38+4])) = int16(-32)
		for i := 0; i < 32; i++ { a[k*38+6+i] = 1 }
		for r := 0; r < 32; r++ {
			b[k*608+r*2] = 0x00; b[k*608+r*2+1] = 0x3C // fp16 1.0
			b[k*608+64+r] = 8
			for n := 0; n < 16; n++ { b[k*608+96+r*16+n] = 0x88 }
		}
	}
	
	sentinel := float32(-99999.0)
	out := make([]float32, 64)
	for i := range out { out[i] = sentinel }
	
	pool.Run(func(workerID, nWorkers int) {
		k3I8I4M1(&a[0], &b[0], &out[0], kBlks, 32)
	})
	
	written := 0
	for i := 0; i < 64; i++ {
		if out[i] != sentinel { written++ }
	}
	
	switch written {
	case 8:
		t.Logf("VLEN=256 (wrote 8 values)")
	case 32:
		t.Logf("VLEN=1024 (wrote 32 values)")
	default:
		t.Logf("Unexpected: wrote %d values", written)
	}
	t.Logf("out[0:8]=%v", out[:8])
	t.Logf("out[8:16]=%v", out[8:16])
}
