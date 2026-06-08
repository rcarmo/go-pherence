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
