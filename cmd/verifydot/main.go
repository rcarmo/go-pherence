package main

import (
	"fmt"
	"os"
	"unsafe"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func fp16dec(h uint16) float32 {
	if h == 0 { return 0 }
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	if exp == 0 { for mant&0x400==0 { mant<<=1; exp-- }; exp++; mant&=0x3ff }
	exp = exp + (127-15)
	bits := (sign<<31)|(exp<<23)|(mant<<13)
	return *(*float32)(unsafe.Pointer(&bits))
}

func dotRow(wqRaw []byte, row, cols int, actI8 []int8, actInv float32) float32 {
	bytesPerBlock := 144
	blocksPerRow := cols / 256
	var result float32
	for blk := 0; blk < blocksPerRow; blk++ {
		b := wqRaw[(row*blocksPerRow+blk)*bytesPerBlock:]
		d := fp16dec(uint16(b[0]) | uint16(b[1])<<8)
		dmin := fp16dec(uint16(b[2]) | uint16(b[3])<<8)
		s := b[4:16]
		var sc, mn [8]float32
		for j := 0; j < 4; j++ { sc[j] = float32(s[j]&63)*d; mn[j] = float32(s[j+4]&63)*dmin }
		for j := 4; j < 8; j++ { k:=j-4; sc[j]=float32((s[j+4]&0xF)|((s[k]>>6)<<4))*d; mn[j]=float32((s[j+4]>>4)|((s[k+4]>>6)<<4))*dmin }
		qs := b[16:144]
		for sb := 0; sb < 8; sb++ {
			var dot, actSum int32
			for i := 0; i < 32; i++ {
				elemInBlock := sb*32 + i
				nib := int8(qs[elemInBlock/2] >> uint(4*(elemInBlock%2)) & 0xf)
				dot += int32(nib) * int32(actI8[blk*256+elemInBlock])
				actSum += int32(actI8[blk*256+elemInBlock])
			}
			result += float32(dot)*sc[sb]*actInv - float32(actSum)*mn[sb]*actInv
		}
	}
	return result
}

func main() {
	g, _ := gguf.Open(os.Args[1])
	wqT, _ := g.TensorByName("blk.0.attn_q.weight")
	wqF32, _ := g.DequantF32(wqT)
	wqRaw, _ := g.Raw(wqT)
	cols := int(wqT.Shape[0])

	act := make([]float32, cols)
	for i := range act { act[i] = 0.01 * float32(i%100-50) }

	var maxAbs float32
	for _, v := range act { a := v; if a < 0 { a = -a }; if a > maxAbs { maxAbs = a } }
	actI8 := make([]int8, cols)
	for i, v := range act {
		q := v * (127.0 / maxAbs)
		if q > 127 { q = 127 } else if q < -128 { q = -128 }
		actI8[i] = int8(q)
	}
	actInv := maxAbs / 127.0

	for row := 0; row < 8; row++ {
		var refDot float32
		for k := 0; k < cols; k++ { refDot += wqF32[row*cols+k] * act[k] }
		ourDot := dotRow(wqRaw, row, cols, actI8, actInv)
		fmt.Printf("row%d: ref=%8.4f our=%8.4f err=%5.2f%%\n", row, refDot, ourDot, (ourDot-refDot)/refDot*100)
	}
}
