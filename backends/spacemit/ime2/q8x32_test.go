package ime2

import "testing"

func TestQuantizeF32RowsQ8M4Into(t *testing.T) {
	const kBlks = 3
	var rows [4][]float32
	for r := 0; r < 4; r++ {
		rows[r] = make([]float32, kBlks*32)
		for i := range rows[r] {
			rows[r][i] = float32((r+1)*(i%17-8)) / 7
		}
	}
	want := QuantizeF32RowsQ8M4(rows, kBlks)
	got := make([]byte, kBlks*K3I8I8ABlockM4Bytes)
	QuantizeF32RowsQ8M4Into(rows, kBlks, got)
	if len(got) != len(want) {
		t.Fatalf("len got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("byte %d got %02x want %02x", i, got[i], want[i])
		}
	}
}

func TestQuantizeF32RowsQ8M4GELUInto(t *testing.T) {
	const kBlks = 3
	var rows [4][]float32
	var geluRows [4][]float32
	for r := 0; r < 4; r++ {
		rows[r] = make([]float32, kBlks*32)
		geluRows[r] = make([]float32, kBlks*32)
		for i := range rows[r] {
			v := float32((r+1)*(i%19-9)) / 5
			rows[r][i] = v
			geluRows[r][i] = geluQ8(v)
		}
	}
	want := QuantizeF32RowsQ8M4(geluRows, kBlks)
	got := make([]byte, kBlks*K3I8I8ABlockM4Bytes)
	QuantizeF32RowsQ8M4GELUInto(rows, kBlks, got)
	if len(got) != len(want) {
		t.Fatalf("len got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("byte %d got %02x want %02x", i, got[i], want[i])
		}
	}
}

func TestPackF32ToQ80x32RowScaleRepeatsScales(t *testing.T) {
	const M, K = 32, 64
	w := make([]float32, M*K)
	for i := range w {
		w[i] = float32((i%23)-11) / 13
	}
	q := PackF32ToQ80x32RowScale(M, K, w)
	if !q.Valid {
		t.Fatal("row-scale pack invalid")
	}
	for r := 0; r < 32; r++ {
		s0 := q.BData[r*2 : r*2+2]
		s1 := q.BData[K3I8I8BTileBytes+r*2 : K3I8I8BTileBytes+r*2+2]
		if s0[0] != s1[0] || s0[1] != s1[1] {
			t.Fatalf("row %d scale differs across K blocks: %02x%02x vs %02x%02x", r, s0[0], s0[1], s1[0], s1[1])
		}
	}
}
