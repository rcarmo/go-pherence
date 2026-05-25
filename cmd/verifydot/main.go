package main
import (
	"fmt"; "os"; "runtime"; "strconv"; "syscall"; "time"; "unsafe"
	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)
func main() {
	runtime.LockOSThread()
	f, _ := os.OpenFile("/proc/set_ai_thread", os.O_WRONLY, 0)
	f.Write([]byte(strconv.Itoa(syscall.Gettid()))); f.Close()
	fmt.Println("On AI core (8-15, VLEN=1024). Testing vmadotKLoopAI with vl=32...")

	K := 2048
	M := 8
	wI8 := make([]int8, M*K)
	for i := 0; i < M; i++ { for k := 0; k < K; k++ { wI8[i*K+k] = int8((i+1)*(k%7+1)%127) } }
	wPacked := ime2.PackTiles(wI8, M, K) // standard 32-byte tiles
	actI8 := make([]int8, 4*K)
	for i := range actI8 { actI8[i] = 10 }
	actPacked := ime2.PackTiles(actI8, 4, K)

	// Correctness: compare with scalar
	var accAI [16]int32
	ime2.VmadotKLoopAI((*byte)(unsafe.Pointer(&wPacked[0])), (*byte)(unsafe.Pointer(&actPacked[0])), &accAI[0], K)
	var expected int32
	for k := 0; k < K; k++ { expected += int32(wI8[k]) * 10 }
	fmt.Printf("Row 0: AI=%d expected=%d %s\n", accAI[0], expected, func() string { if accAI[0]==expected { return "✓" }; return "✗" }())

	// Benchmark: AI core vmadot
	tilesPerRow := K / 8
	const iters = 3000
	t0 := time.Now()
	for n := 0; n < iters; n++ {
		for g := 0; g < M/4; g++ {
			var acc [16]int32
			ime2.VmadotKLoopAI((*byte)(unsafe.Pointer(&wPacked[g*tilesPerRow*32])), (*byte)(unsafe.Pointer(&actPacked[0])), &acc[0], K)
		}
	}
	aiTime := time.Since(t0)
	fmt.Printf("AI core: %v per %dx%d matmul (%.1f µs)\n", aiTime/iters, M, K, float64(aiTime.Microseconds())/float64(iters))
	fmt.Printf("GOPS: %.1f\n", float64(M*K*4*2*iters)/float64(aiTime.Seconds())/1e9)
}
