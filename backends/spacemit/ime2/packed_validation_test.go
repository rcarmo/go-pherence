package ime2

import "testing"

func TestPackedPreflightRejectsBeforeNativeAccess(t *testing.T) {
	for name, fn := range map[string]func(){
		"short A":       func() { GemmINT8Packed(4, 4, 8, []int8{1}, make([]int8, 32), make([]int32, 16)) },
		"short C":       func() { GemmINT8PackedParallel(4, 4, 8, make([]int8, 32), make([]int8, 32), []int32{1}, 2) },
		"pool short":    func() { GemmINT8PackedPool(4, 4, 8, nil, nil, nil, nil) },
		"overflow":      func() { GemmINT8Packed(4, 4, int(^uint(0)>>1)-7, nil, nil, nil) },
		"pack negative": func() { PackTiles(nil, -4, 8) },
		"pack short":    func() { PackTilesInto([]int8{1}, 4, 8, make([]int8, 32)) },
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("invalid buffer accepted")
				}
			}()
			fn()
		})
	}
	GemmINT8Packed(0, 0, 0, nil, nil, nil)
}
