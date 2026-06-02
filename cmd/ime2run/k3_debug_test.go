package main

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

// TestK3I8I4M1Simple verifies the kernel with constant/simple inputs
func TestK3I8I4M1Simple(t *testing.T) {
	pool := NewAIWorkerPool(6)
	_ = pool
	// Run directly on AI thread via serial path
	registerAIThread(8)

	// Build a single 32-col subblock with known values
	// B data: 608 bytes = [fp16_d×32:64][zp×32:32][qs×512:512]
	b := make([]byte, 608)

	// B scales: all = 1.0 in fp16 = 0x3C00 LE
	for col := 0; col < 32; col++ {
		b[col*2+0] = 0x00
		b[col*2+1] = 0x3C // fp16(1.0)
	}
	// ZP: all 0 at offset 64
	for col := 0; col < 32; col++ {
		b[64+col] = 0
	}
	// QS at offset 96: col c has 16 bytes for 32 nibbles
	// Set w[i] = 1 for all i, for all cols (constant weight = 1)
	for col := 0; col < 32; col++ {
		for i := 0; i < 16; i++ {
			b[96+col*16+i] = 0x11 // lo4=1, hi4=1 => w[0]=1, w[1]=1
		}
	}

	// A data: 38 bytes = [fp32_scale:4][int16_sum:2][int8_data:32]
	a := make([]byte, 38)
	for i := 0; i < 32; i++ {
		a[6+i] = 1 // all activations = 1
	}
	bits := math.Float32bits(1.0)
	a[0] = byte(bits)
	a[1] = byte(bits >> 8)
	a[2] = byte(bits >> 16)
	a[3] = byte(bits >> 24)
	// int16 sum = 32
	a[4] = 32
	a[5] = 0

	out := make([]float32, 32)
	k3I8I4M1((*byte)(unsafe.Pointer(&a[0])), (*byte)(unsafe.Pointer(&b[0])), &out[0], 1, 32)

	t.Logf("ALL outputs: %v", out)
	//   dot = sum_i(act[i] * w[i]) = sum_i(1 * 1) = 32
	// With ZP=0 correction = 0
	expected := float32(32.0)
	t.Logf("out[0..3] = %v %v %v %v (expected all %.1f)", out[0], out[1], out[2], out[3], expected)
	for col := 0; col < 32; col++ {
		if math.Abs(float64(out[col]-expected)) > 2.0 {
			t.Errorf("col %d: got %.4f expected %.4f", col, out[col], expected)
		}
	}
}

// TestK3I8I4M1Ref compares kernel against scalar reference for one group
func TestK3I8I4M1LargeRef(t *testing.T) {
	pool := NewAIWorkerPool(6)
	defer pool.Close()

	M, K := 64, 1024
	subs := K / 32
	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			raw[m*K+k] = int8((m*3 + k*5 + 7) & 15)
		}
		for sb := 0; sb < subs; sb++ {
			scales[m*subs+sb] = 0.001 + float32((m+sb)%17)*0.0007
			mins[m*subs+sb] = 0.01 + float32((m*2+sb)%13)*0.002
		}
	}
	act := make([]float32, K)
	for k := 0; k < K; k++ {
		act[k] = float32(((k*11+3)%31)-15) / 7.0
	}
	x32 := repackQ4KToQ41x32(M, K, raw, scales, mins)
	if !x32.Valid {
		t.Fatal("repack failed")
	}
	ref := make([]float32, M)
	got := make([]float32, M)
	q4kQ41x32MatVecRef(x32, act, ref)
	q4kQ41x32MatVecCM1(x32, mins, act, got, pool)
	var maxDiff float64
	maxIdx := 0
	for i := range ref {
		d := math.Abs(float64(got[i] - ref[i]))
		if d > maxDiff {
			maxDiff, maxIdx = d, i
		}
	}
	t.Logf("M=%d K=%d maxDiff=%.6f idx=%d got=%.6f ref=%.6f", M, K, maxDiff, maxIdx, got[maxIdx], ref[maxIdx])
	if maxDiff > 0.75 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func TestK3I8I4M1Ref(t *testing.T) {
	pool := NewAIWorkerPool(6)

	// Use a small but real-ish test: M=32, K=32 (1 subblock, 1 group)
	// Fill with simple data
	M, K := 32, 32
	subs := K / 32

	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)

	// Set: weight[m][k] = int8((m + k) % 15) (nibble range 0..14)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			raw[m*K+k] = int8((m + k) % 15)
		}
		scales[m*subs] = 0.1
		mins[m*subs] = 0.05
	}

	// activation = alternating 1, -1
	act := make([]float32, K)
	for k := 0; k < K; k++ {
		if k%2 == 0 {
			act[k] = 1.0
		} else {
			act[k] = -1.0
		}
	}

	// Reference: scalar matmul
	outRef := make([]float32, M)
	for m := 0; m < M; m++ {
		var s float32
		for ks := 0; ks < subs; ks++ {
			sc := scales[m*subs+ks]
			mn := mins[m*subs+ks]
			for i := 0; i < 32; i++ {
				nibble := float32(int(raw[m*K+ks*32+i]))
				s += act[ks*32+i] * (sc*nibble - mn)
			}
		}
		outRef[m] = s
	}

	// Pack and run kernel
	x32 := repackQ4KToQ41x32(M, K, raw, scales, mins)
	if !x32.Valid {
		t.Fatal("repack failed")
	}
	fmt.Printf("x32 BData len=%d (expected %d)\n", len(x32.BData), (M/32)*subs*608)

	outKernel := make([]float32, M)
	q4kQ41x32MatVecCM1(x32, mins, act, outKernel, pool)

	t.Logf("ref[0..3]    = %.4f %.4f %.4f %.4f", outRef[0], outRef[1], outRef[2], outRef[3])
	t.Logf("kernel[0..3] = %.4f %.4f %.4f %.4f", outKernel[0], outKernel[1], outKernel[2], outKernel[3])

	for m := 0; m < M; m++ {
		diff := math.Abs(float64(outKernel[m] - outRef[m]))
		if diff > 0.5 {
			t.Errorf("row %d: kernel=%.4f ref=%.4f diff=%.4f", m, outKernel[m], outRef[m], diff)
		}
	}
}

func TestK3I8I4M1CExactResidualRef(t *testing.T) {
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	M, K := 64, 1024
	subs := K / 32
	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			raw[m*K+k] = int8((m*3 + k*5 + 7) & 15)
		}
		for sb := 0; sb < subs; sb++ {
			scales[m*subs+sb] = 0.001 + float32((m+sb)%17)*0.0007
			mins[m*subs+sb] = 0.01 + float32((m*2+sb)%13)*0.002
		}
	}
	act := make([]float32, K)
	for k := 0; k < K; k++ {
		act[k] = float32(((k*11+3)%31)-15) / 7.0
	}
	x32 := repackQ4KToQ41x32(M, K, raw, scales, mins)
	if !x32.Valid {
		t.Fatal("repack failed")
	}
	ref := make([]float32, M)
	got := make([]float32, M)
	matVecQ4KF32(M, K, raw, scales, mins, act, ref)
	q4kQ41x32MatVecCM1(x32, mins, act, got, pool)
	var maxDiff float64
	maxIdx := 0
	for i := range ref {
		d := math.Abs(float64(got[i] - ref[i]))
		if d > maxDiff {
			maxDiff, maxIdx = d, i
		}
	}
	t.Logf("C-M1 exact-residual M=%d K=%d maxDiff=%.6f idx=%d got=%.6f ref=%.6f", M, K, maxDiff, maxIdx, got[maxIdx], ref[maxIdx])
	if maxDiff > 0.75 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func TestQ41x32ResidualPrecompute(t *testing.T) {
	M, K := 32, 64
	subs := K / 32
	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)
	for r := 0; r < M; r++ {
		for k := 0; k < K; k++ {
			raw[r*K+k] = int8((r + k) & 15)
		}
		for sb := 0; sb < subs; sb++ {
			scales[r*subs+sb] = 0.003 + float32((r+sb)%11)*0.001
			mins[r*subs+sb] = 0.011 + float32((r*3+sb)%17)*0.0017
		}
	}
	x32 := repackQ4KToQ41x32(M, K, raw, scales, mins)
	if !x32.Valid {
		t.Fatal("repack failed")
	}
	if len(x32.Residual) != M*subs {
		t.Fatalf("Residual len=%d want %d", len(x32.Residual), M*subs)
	}
	var maxErr float64
	for r := 0; r < M; r++ {
		for sb := 0; sb < subs; sb++ {
			idx := q41x32MetaIndex(r/32, sb, r%32, subs)
			want := mins[r*subs+sb] - float32(x32.ZP[idx])*x32.D[idx]
			err := math.Abs(float64(x32.Residual[idx] - want))
			if err > maxErr {
				maxErr = err
			}
		}
	}
	t.Logf("residual precompute maxErr=%.9f", maxErr)
	if maxErr > 1e-7 {
		t.Fatalf("residual maxErr %.9f", maxErr)
	}
}

func TestK3I8I4M1CResidualFusedRef(t *testing.T) {
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	M, K := 64, 1024
	subs := K / 32
	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			raw[m*K+k] = int8((m*3 + k*5 + 7) & 15)
		}
		for sb := 0; sb < subs; sb++ {
			scales[m*subs+sb] = 0.001 + float32((m+sb)%17)*0.0007
			mins[m*subs+sb] = 0.01 + float32((m*2+sb)%13)*0.002
		}
	}
	act := make([]float32, K)
	for k := 0; k < K; k++ {
		act[k] = float32(((k*11+3)%31)-15) / 7.0
	}
	x32 := repackQ4KToQ41x32(M, K, raw, scales, mins)
	if !x32.Valid {
		t.Fatal("repack failed")
	}
	q8 := quantizeQ8Blocks32(act)
	quantA := q8Block32ToBytes(q8)
	ref := make([]float32, M)
	got := make([]float32, M)
	matVecQ4KF32(M, K, raw, scales, mins, act, ref)
	pool.Run(func(workerID, nWorkers int) {
		groups := M / 32
		gStart := workerID * groups / nWorkers
		gEnd := (workerID + 1) * groups / nWorkers
		for rg := gStart; rg < gEnd; rg++ {
			k3I8I4M1CResidual(
				(*byte)(unsafe.Pointer(&quantA[0])),
				(*byte)(unsafe.Pointer(&x32.BData[rg*subs*608])),
				&x32.Residual[rg*subs*32],
				&got[rg*32], subs, 32)
		}
	})
	var maxDiff float64
	maxIdx := 0
	for i := range ref {
		d := math.Abs(float64(got[i] - ref[i]))
		if d > maxDiff {
			maxDiff, maxIdx = d, i
		}
	}
	t.Logf("C-M1 fused-residual M=%d K=%d maxDiff=%.6f idx=%d got=%.6f ref=%.6f", M, K, maxDiff, maxIdx, got[maxIdx], ref[maxIdx])
	if maxDiff > 0.75 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func BenchmarkK3I8I4M1CResidualKernel(b *testing.B) {
	registerAIThread(8)
	subs := 32
	a := make([]byte, subs*38)
	bd := make([]byte, subs*608)
	residual := make([]float32, subs*32)
	out := make([]float32, 32)
	for sb := 0; sb < subs; sb++ {
		ao := sb * 38
		bits := math.Float32bits(1.0)
		a[ao+0] = byte(bits)
		a[ao+1] = byte(bits >> 8)
		a[ao+2] = byte(bits >> 16)
		a[ao+3] = byte(bits >> 24)
		negSum := int16(-32)
		a[ao+4] = byte(uint16(negSum))
		a[ao+5] = byte(uint16(negSum) >> 8)
		for i := 0; i < 32; i++ {
			a[ao+6+i] = 1
		}
		bo := sb * 608
		for r := 0; r < 32; r++ {
			bd[bo+r*2] = 0x00
			bd[bo+r*2+1] = 0x3c
			bd[bo+64+r] = 1
			for i := 0; i < 16; i++ {
				bd[bo+96+r*16+i] = 0x11
			}
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		k3I8I4M1CResidual((*byte)(unsafe.Pointer(&a[0])), (*byte)(unsafe.Pointer(&bd[0])), &residual[0], &out[0], subs, 32)
	}
}

func BenchmarkK3I8I4M1CKernel(b *testing.B) {
	registerAIThread(8)
	subs := 32
	a := make([]byte, subs*38)
	bd := make([]byte, subs*608)
	out := make([]float32, 32)
	for sb := 0; sb < subs; sb++ {
		ao := sb * 38
		bits := math.Float32bits(1.0)
		a[ao+0] = byte(bits)
		a[ao+1] = byte(bits >> 8)
		a[ao+2] = byte(bits >> 16)
		a[ao+3] = byte(bits >> 24)
		negSum := int16(-32)
		a[ao+4] = byte(uint16(negSum))
		a[ao+5] = byte(uint16(negSum) >> 8)
		for i := 0; i < 32; i++ {
			a[ao+6+i] = 1
		}
		bo := sb * 608
		for r := 0; r < 32; r++ {
			bd[bo+r*2] = 0x00
			bd[bo+r*2+1] = 0x3c
			bd[bo+64+r] = 1
			for i := 0; i < 16; i++ {
				bd[bo+96+r*16+i] = 0x11
			}
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		k3I8I4M1C((*byte)(unsafe.Pointer(&a[0])), (*byte)(unsafe.Pointer(&bd[0])), &out[0], subs, 32)
	}
}

func TestK3I8I4DispatcherM4MatchesM1(t *testing.T) {
	pool := NewAIWorkerPool(1)
	defer pool.Close()
	M, K := 32, 1024
	subs := K / 32
	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			raw[m*K+k] = int8((m*7 + k*3 + 5) & 15)
		}
		for sb := 0; sb < subs; sb++ {
			scales[m*subs+sb] = 0.002 + float32((m+sb)%13)*0.0005
			mins[m*subs+sb] = 0.01 + float32((m*2+sb)%7)*0.001
		}
	}
	x32 := repackQ4KToQ41x32(M, K, raw, scales, mins)
	if !x32.Valid {
		t.Fatal("repack failed")
	}
	var rows [4][]byte
	for r := 0; r < 4; r++ {
		act := make([]float32, K)
		for k := 0; k < K; k++ {
			act[k] = float32(((r+1)*(k%17)-8)) / 5.0
		}
		rows[r] = quantizeQ8Blocks32Bytes(act)
	}
	want := make([]float32, 4*32)
	got := make([]float32, 4*32)
	var handled int
	packedA := packQ8RowsM4(rows, subs)
	pool.Run(func(workerID, nWorkers int) {
		for r := 0; r < 4; r++ {
			k3I8I4M1((*byte)(unsafe.Pointer(&rows[r][0])), (*byte)(unsafe.Pointer(&x32.BData[0])), &want[r*32], subs, 32)
		}
		handled = k3I8I4((*byte)(unsafe.Pointer(&packedA[0])), (*byte)(unsafe.Pointer(&x32.BData[0])), &got[0], 4, 32, subs, 32)
	})
	if handled != 4 {
		t.Fatalf("dispatcher handled %d rows, want 4", handled)
	}
	var maxDiff float64
	maxIdx := 0
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > maxDiff {
			maxDiff = diff
			maxIdx = i
		}
	}
	t.Logf("maxDiff=%.6f idx=%d got=%.6f want=%.6f", maxDiff, maxIdx, got[maxIdx], want[maxIdx])
	if maxDiff > 0.05 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func TestQ4KMatVec4M4MatchesFourM1(t *testing.T) {
	registerAIThread(8)
	M, K := 64, 1024
	subs := K / 32
	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			raw[m*K+k] = int8((m*5 + k*7 + 3) & 15)
		}
		for sb := 0; sb < subs; sb++ {
			scales[m*subs+sb] = 0.0015 + float32((m+sb)%19)*0.0004
			mins[m*subs+sb] = 0.008 + float32((m*3+sb)%11)*0.0013
		}
	}
	x32 := repackQ4KToQ41x32(M, K, raw, scales, mins)
	if !x32.Valid {
		t.Fatal("repack failed")
	}
	var acts [4][]float32
	var outs [4][]float32
	var want [4][]float32
	for r := 0; r < 4; r++ {
		acts[r] = make([]float32, K)
		outs[r] = make([]float32, M)
		want[r] = make([]float32, M)
		for k := 0; k < K; k++ {
			acts[r][k] = float32(((r+2)*(k%23)-11)) / 6.0
		}
		q := quantizeQ8Blocks32Bytes(acts[r])
		for rg := 0; rg < M/32; rg++ {
			k3I8I4M1((*byte)(unsafe.Pointer(&q[0])), (*byte)(unsafe.Pointer(&x32.BData[rg*subs*608])), &want[r][rg*32], subs, 32)
		}
	}
	if !q4kQ41x32MatVec4GoAsm(x32, acts, outs) {
		t.Fatal("q4kQ41x32MatVec4GoAsm returned false")
	}
	var maxDiff float64
	maxR, maxI := 0, 0
	for r := 0; r < 4; r++ {
		for i := 0; i < M; i++ {
			if diff := math.Abs(float64(outs[r][i] - want[r][i])); diff > maxDiff {
				maxDiff, maxR, maxI = diff, r, i
			}
		}
	}
	t.Logf("maxDiff=%.6f row=%d idx=%d got=%.6f want=%.6f", maxDiff, maxR, maxI, outs[maxR][maxI], want[maxR][maxI])
	if maxDiff > 0.05 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func TestQ4KGateUpSiluFuseMatchesUnfused(t *testing.T) {
	M, K := 64, 1024
	subs := K / 32
	mkWeight := func(seed int) q4kQ41x32 {
		raw := make([]int8, M*K)
		scales := make([]float32, M*subs)
		mins := make([]float32, M*subs)
		for m := 0; m < M; m++ {
			for k := 0; k < K; k++ {
				raw[m*K+k] = int8((m*seed + k*3 + 7) & 15)
			}
			for sb := 0; sb < subs; sb++ {
				scales[m*subs+sb] = 0.001 + float32((m+sb+seed)%17)*0.0006
				mins[m*subs+sb] = 0.007 + float32((m*2+sb+seed)%13)*0.0011
			}
		}
		w := repackQ4KToQ41x32(M, K, raw, scales, mins)
		if !w.Valid {
			t.Fatal("repack failed")
		}
		return w
	}
	gate := mkWeight(5)
	up := mkWeight(11)
	act := make([]float32, K)
	for k := 0; k < K; k++ {
		act[k] = float32((k%29)-14) / 8.0
	}
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	gateA, upA, hiddenA := make([]float32, M), make([]float32, M), make([]float32, M)
	gateB, upB, hiddenB := make([]float32, M), make([]float32, M), make([]float32, M)
	if !q4kQ41x32MatVecBatchSameAct(act, pool, q4kBatchMatVecSpec{W: gate, Out: gateA}, q4kBatchMatVecSpec{W: up, Out: upA}) {
		t.Fatal("batch failed")
	}
	for i := 0; i < M; i++ {
		hiddenA[i] = silu(gateA[i]) * upA[i]
	}
	if !q4kQ41x32GateUpSiluSameAct(act, pool, gate, up, gateB, upB, hiddenB) {
		t.Fatal("fused failed")
	}
	var maxDiff float64
	for i := 0; i < M; i++ {
		for _, d := range []float64{
			math.Abs(float64(gateA[i] - gateB[i])),
			math.Abs(float64(upA[i] - upB[i])),
			math.Abs(float64(hiddenA[i] - hiddenB[i])),
		} {
			if d > maxDiff {
				maxDiff = d
			}
		}
	}
	t.Logf("maxDiff=%.6f", maxDiff)
	if maxDiff > 1e-5 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func TestK3I8I8M1NativeRef(t *testing.T) {
	M, K := 64, 1024
	f32 := make([]float32, M*K)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			f32[m*K+k] = float32(((m*7+k*5)%41)-20) / 13.0
		}
	}
	act := make([]float32, K)
	for k := range act {
		act[k] = float32(((k*3)%29)-14) / 9.0
	}
	w := repackF32ToQ80x32(M, K, f32)
	if !w.Valid {
		t.Fatal("repack failed")
	}
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	got := make([]float32, M)
	if !q8Q80x32MatVecNative(w, act, got, pool) {
		t.Fatal("native matvec failed")
	}
	ref := make([]float32, M)
	matVecF32Direct(M, K, f32, act, ref)
	var maxDiff float64
	maxIdx := 0
	for i := range ref {
		if d := math.Abs(float64(got[i] - ref[i])); d > maxDiff {
			maxDiff, maxIdx = d, i
		}
	}
	t.Logf("maxDiff=%.6f idx=%d got=%.6f ref=%.6f", maxDiff, maxIdx, got[maxIdx], ref[maxIdx])
	if maxDiff > 0.35 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func TestK3I8I8M4DispatcherMatchesM1(t *testing.T) {
	registerAIThread(8)
	M, K := 32, 1024
	f32 := make([]float32, M*K)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			f32[m*K+k] = float32(((m*11+k*3)%47)-23) / 17.0
		}
	}
	w := repackF32ToQ80x32(M, K, f32)
	if !w.Valid {
		t.Fatal("repack failed")
	}
	subs := K / 32
	var acts [4][]float32
	var qRows [4][]byte
	for r := 0; r < 4; r++ {
		acts[r] = make([]float32, K)
		for k := range acts[r] {
			acts[r][k] = float32(((r+1)*(k%31)-15)) / 10.0
		}
		qRows[r] = quantizeQ8Blocks32Bytes(acts[r])
	}
	want := make([]float32, 4*32)
	for r := 0; r < 4; r++ {
		k3I8I8M1((*byte)(unsafe.Pointer(&qRows[r][0])), (*byte)(unsafe.Pointer(&w.BData[0])), &want[r*32], subs, 32)
	}
	packedA := packQ8RowsM4(qRows, subs)
	got := make([]float32, 4*32)
	handled := q8I8Dispatcher((*byte)(unsafe.Pointer(&packedA[0])), (*byte)(unsafe.Pointer(&w.BData[0])), &got[0], 4, 32, subs, 32)
	if handled != 4 {
		t.Fatalf("handled=%d want 4", handled)
	}
	var maxDiff float64
	maxIdx := 0
	for i := range want {
		if d := math.Abs(float64(got[i]-want[i])); d > maxDiff {
			maxDiff, maxIdx = d, i
		}
	}
	t.Logf("maxDiff=%.6f idx=%d got=%.6f want=%.6f", maxDiff, maxIdx, got[maxIdx], want[maxIdx])
	if maxDiff > 0.05 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func TestQ4KGateUpSiluFuseBWaveMatchesDirect(t *testing.T) {
	oldWave := q4kTCMBWaveOn
	defer func() { q4kTCMBWaveOn = oldWave }()
	M, K := 192, 1024
	subs := K / 32
	mkWeight := func(seed int) q4kQ41x32 {
		raw := make([]int8, M*K)
		scales := make([]float32, M*subs)
		mins := make([]float32, M*subs)
		for m := 0; m < M; m++ {
			for k := 0; k < K; k++ {
				raw[m*K+k] = int8((m*seed + k*5 + 9) & 15)
			}
			for sb := 0; sb < subs; sb++ {
				scales[m*subs+sb] = 0.001 + float32((m+sb+seed)%17)*0.0005
				mins[m*subs+sb] = 0.006 + float32((m*3+sb+seed)%13)*0.001
			}
		}
		w := repackQ4KToQ41x32(M, K, raw, scales, mins)
		if !w.Valid {
			t.Fatal("repack failed")
		}
		return w
	}
	gate := mkWeight(7)
	up := mkWeight(13)
	act := make([]float32, K)
	for k := 0; k < K; k++ {
		act[k] = float32((k%31)-15) / 9.0
	}
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	gateA, upA, hiddenA := make([]float32, M), make([]float32, M), make([]float32, M)
	gateB, upB, hiddenB := make([]float32, M), make([]float32, M), make([]float32, M)
	q4kTCMBWaveOn = false
	if !q4kQ41x32MatVecBatchSameAct(act, pool, q4kBatchMatVecSpec{W: gate, Out: gateA}, q4kBatchMatVecSpec{W: up, Out: upA}) {
		t.Fatal("direct batch failed")
	}
	for i := 0; i < M; i++ {
		hiddenA[i] = silu(gateA[i]) * upA[i]
	}
	q4kTCMBWaveOn = true
	if !q4kQ41x32GateUpSiluSameAct(act, pool, gate, up, gateB, upB, hiddenB) {
		t.Fatal("b-wave fused failed")
	}
	var maxDiff float64
	for i := 0; i < M; i++ {
		for _, d := range []float64{
			math.Abs(float64(gateA[i] - gateB[i])),
			math.Abs(float64(upA[i] - upB[i])),
			math.Abs(float64(hiddenA[i] - hiddenB[i])),
		} {
			if d > maxDiff {
				maxDiff = d
			}
		}
	}
	t.Logf("maxDiff=%.6f", maxDiff)
	if maxDiff > 1e-5 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func TestQ4KBWaveMatVecMatchesDirect(t *testing.T) {
	oldWave := q4kTCMBWaveOn
	defer func() { q4kTCMBWaveOn = oldWave }()
	M, K := 1024, 1024
	subs := K / 32
	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			raw[m*K+k] = int8((m*7 + k*11 + 5) & 15)
		}
		for sb := 0; sb < subs; sb++ {
			scales[m*subs+sb] = 0.001 + float32((m+sb)%17)*0.0004
			mins[m*subs+sb] = 0.006 + float32((m*3+sb)%13)*0.001
		}
	}
	w := repackQ4KToQ41x32(M, K, raw, scales, mins)
	if !w.Valid { t.Fatal("repack failed") }
	act := make([]float32, K)
	for k := range act { act[k] = float32((k%31)-15) / 8.0 }
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	direct := make([]float32, M)
	wave := make([]float32, M)
	q4kTCMBWaveOn = false
	q4kQ41x32MatVecGoAsm(w, act, direct, pool)
	q4kTCMBWaveOn = true
	q4kQ41x32MatVecGoAsm(w, act, wave, pool)
	var maxDiff float64
	maxIdx := 0
	for i := range direct {
		if d := math.Abs(float64(direct[i]-wave[i])); d > maxDiff { maxDiff, maxIdx = d, i }
	}
	if maxIdx >= 0 {
		t.Logf("maxDiff=%.6f idx=%d direct=%.6f wave=%.6f", maxDiff, maxIdx, direct[maxIdx], wave[maxIdx])
	} else {
		t.Logf("maxDiff=0")
	}
	if maxDiff > 1e-5 { t.Fatalf("maxDiff %.6f > tolerance", maxDiff) }
}

func TestQ4KBWaveWOLikeMatVecMatchesDirect(t *testing.T) {
	oldWave := q4kTCMBWaveOn
	defer func() { q4kTCMBWaveOn = oldWave }()
	M, K := 1024, 2048
	subs := K / 32
	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ { raw[m*K+k] = int8((m*5 + k*7 + 3) & 15) }
		for sb := 0; sb < subs; sb++ {
			scales[m*subs+sb] = 0.001 + float32((m+sb)%19)*0.0003
			mins[m*subs+sb] = 0.004 + float32((m*2+sb)%11)*0.0008
		}
	}
	w := repackQ4KToQ41x32(M, K, raw, scales, mins)
	if !w.Valid { t.Fatal("repack failed") }
	act := make([]float32, K)
	for k := range act { act[k] = float32((k%37)-18) / 9.0 }
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	direct := make([]float32, M)
	wave := make([]float32, M)
	q4kTCMBWaveOn = false
	q4kQ41x32MatVecGoAsm(w, act, direct, pool)
	q4kTCMBWaveOn = true
	q4kQ41x32MatVecGoAsm(w, act, wave, pool)
	var maxDiff float64
	maxIdx := 0
	for i := range direct { if d := math.Abs(float64(direct[i]-wave[i])); d > maxDiff { maxDiff, maxIdx = d, i } }
	if maxIdx >= 0 { t.Logf("maxDiff=%.6f idx=%d direct=%.6f wave=%.6f", maxDiff, maxIdx, direct[maxIdx], wave[maxIdx]) } else { t.Logf("maxDiff=0") }
	if maxDiff > 1e-5 { t.Fatalf("maxDiff %.6f > tolerance", maxDiff) }
}

func TestQ4KBWaveBatchMixedShapesMatchesDirect(t *testing.T) {
	oldWave := q4kTCMBWaveOn
	oldBatch := q4kTCMBWaveBatchOn
	defer func() { q4kTCMBWaveOn = oldWave; q4kTCMBWaveBatchOn = oldBatch }()
	K := 1024
	subs := K / 32
	mkWeight := func(M int, seed int) q4kQ41x32 {
		raw := make([]int8, M*K)
		scales := make([]float32, M*subs)
		mins := make([]float32, M*subs)
		for m := 0; m < M; m++ {
			for k := 0; k < K; k++ { raw[m*K+k] = int8((m*seed + k*7 + 3) & 15) }
			for sb := 0; sb < subs; sb++ {
				scales[m*subs+sb] = 0.001 + float32((m+sb+seed)%17)*0.0004
				mins[m*subs+sb] = 0.005 + float32((m*3+sb+seed)%13)*0.0009
			}
		}
		w := repackQ4KToQ41x32(M, K, raw, scales, mins)
		if !w.Valid { t.Fatal("repack failed") }
		return w
	}
	wq := mkWeight(2048, 5)
	wk := mkWeight(1024, 11)
	act := make([]float32, K)
	for k := range act { act[k] = float32((k%29)-14) / 7.0 }
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	qDirect, kDirect := make([]float32, wq.M), make([]float32, wk.M)
	qWave, kWave := make([]float32, wq.M), make([]float32, wk.M)
	q4kTCMBWaveOn = false
	q4kTCMBWaveBatchOn = false
	if !q4kQ41x32MatVecBatchSameAct(act, pool, q4kBatchMatVecSpec{W: wq, Out: qDirect}, q4kBatchMatVecSpec{W: wk, Out: kDirect}) {
		t.Fatal("direct batch failed")
	}
	q4kTCMBWaveOn = true
	q4kTCMBWaveBatchOn = true
	if !q4kQ41x32MatVecBatchSameAct(act, pool, q4kBatchMatVecSpec{W: wq, Out: qWave}, q4kBatchMatVecSpec{W: wk, Out: kWave}) {
		t.Fatal("b-wave batch failed")
	}
	var maxDiff float64
	for i := range qDirect { if d := math.Abs(float64(qDirect[i]-qWave[i])); d > maxDiff { maxDiff = d } }
	for i := range kDirect { if d := math.Abs(float64(kDirect[i]-kWave[i])); d > maxDiff { maxDiff = d } }
	t.Logf("maxDiff=%.6f", maxDiff)
	if maxDiff > 1e-5 { t.Fatalf("maxDiff %.6f > tolerance", maxDiff) }
}

// BenchmarkGateUpDecode mirrors the actual gate_up decode operation:
// M=3072, K=1024, batch of 2 (gate+up), 6 workers.
func BenchmarkGateUpDecodeSingleWorkerSerial(b *testing.B) {
	const M, K = 3072, 1024
	w := makeRandQ4KW(M, K)
	act := make([]float32, K)
	for i := range act { act[i] = 0.1 }
	quantA := quantizeQ8Blocks32Bytes(act)
	out := make([]float32, M)
	subs := K / 32
	groups := M / 32
	quantPtr := (*byte)(unsafe.Pointer(&quantA[0]))
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		// Serial version: all groups sequentially (1 worker)
		for rg := 0; rg < groups; rg++ {
			bPtr := (*byte)(unsafe.Pointer(&w.BData[rg*subs*608]))
			k3I8I4M1Groups(quantPtr, bPtr, &out[rg*32], subs, 1)
		}
	}
}

func BenchmarkGateUpDecodePooled6Workers(b *testing.B) {
	const M, K = 3072, 1024
	w := makeRandQ4KW(M, K)
	act := make([]float32, K)
	for i := range act { act[i] = 0.1 }
	out := make([]float32, M)
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		q4kQ41x32MatVecGoAsmWithCorrection(w, nil, act, out, pool)
	}
}

func makeRandQ4KW(M, K int) q4kQ41x32 {
	subs := K / 32
	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			raw[m*K+k] = int8((m*3 + k*7 + 5) & 15)
		}
		for sb := 0; sb < subs; sb++ {
			scales[m*subs+sb] = 0.001 + float32((m+sb)%17)*0.0006
			mins[m*subs+sb] = 0.007 + float32((m*2+sb)%13)*0.0011
		}
	}
	w := repackQ4KToQ41x32(M, K, raw, scales, mins)
	return w
}
