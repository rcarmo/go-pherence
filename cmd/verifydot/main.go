package main
import (
	"fmt"
	"math"
	"os"
	"github.com/rcarmo/go-pherence/loader/gguf"
)
func main() {
	g, _ := gguf.Open(os.Args[1])
	tokT, _ := g.TensorByName("token_embd.weight")
	tokF32, _ := g.DequantF32(tokT)
	nEmbd := int(tokT.Shape[0])
	nVocab := int(tokT.Shape[1])
	
	fmt.Printf("token_embd: %dx%d (type %d)\n", nVocab, nEmbd, tokT.QType)
	
	// Check RMS of various token embeddings
	for _, tok := range []int{0, 1, 9707, 374, 635, 279} {
		if tok >= nVocab { continue }
		emb := tokF32[tok*nEmbd : (tok+1)*nEmbd]
		var ss float64
		for _, v := range emb { ss += float64(v)*float64(v) }
		rms := math.Sqrt(ss / float64(nEmbd))
		fmt.Printf("  tok %6d rms=%.6f first4=[%.4f %.4f %.4f %.4f]\n", tok, rms, emb[0], emb[1], emb[2], emb[3])
	}
}
