package main

import (
	"fmt"
	"unsafe"
	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

func main() {
	// Simple test: 4 rows × 8 elements (one vmadot pass)
	var wt [32]byte  // 4 rows × 8 cols: all value 2
	var act [32]byte // broadcast: all value 3
	var acc [16]int32

	for i := range wt { wt[i] = 2 }
	for i := range act { act[i] = 3 }

	ime2.VmadotAccSS4x8(&act[0], &wt[0], &acc[0])

	// Expected: each row dot = 8 × (3 × 2) = 48
	// acc[r*4+c]: since act is broadcast (all rows same), 
	// acc[r][c] = dot(act_row_c, wt_row_r) = 48 for all r,c
	fmt.Printf("acc[0][0]=%d acc[1][0]=%d acc[2][0]=%d acc[3][0]=%d\n",
		acc[0], acc[4], acc[8], acc[12])
	fmt.Printf("acc[0][1]=%d acc[0][2]=%d acc[0][3]=%d\n",
		acc[1], acc[2], acc[3])
	
	// Expected: all = 48
	if acc[0] == 48 && acc[4] == 48 {
		fmt.Println("CORRECT: broadcast vmadot works as expected")
	} else {
		fmt.Printf("WRONG: expected 48, got acc[0]=%d\n", acc[0])
		fmt.Println("Check: vmadot(A, B) = A * B^T")
		fmt.Println("A=act (rows=broadcast 3), B=wt (rows=2)")
		fmt.Println("dot(act_row, wt_row) = 8 * 3 * 2 = 48")
	}
}

var _ = unsafe.Pointer(nil)
