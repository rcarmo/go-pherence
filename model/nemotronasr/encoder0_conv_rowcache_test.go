package nemotronasr

import (
	"math"
	"reflect"
	"testing"
)

func TestConvRowMajorMatchesPublicCacheExact(t *testing.T) {
	var public, direct Encoder0ConvCache
	weights := make([]float32, encoderWidth*encoderConvKernel)
	for i := range weights {
		weights[i] = float32(i%17-8) / 31
	}
	for _, frames := range []int{1, 4, 5, 3, 4} {
		rowMajor, channelMajor := make([]float32, frames*encoderWidth), make([]float32, frames*encoderWidth)
		for r := 0; r < frames; r++ {
			for ch := 0; ch < encoderWidth; ch++ {
				rowMajor[r*encoderWidth+ch] = float32((r*encoderWidth+ch)%37-18) / 43
				channelMajor[ch*frames+r] = rowMajor[r*encoderWidth+ch]
			}
		}
		_, want, err := public.Update(channelMajor, frames, weights)
		if err != nil {
			t.Fatal(err)
		}
		dst := make([]float32, len(rowMajor))
		if err := direct.updateRowMajor(rowMajor, frames, weights, dst); err != nil {
			t.Fatal(err)
		}
		for r := 0; r < frames; r++ {
			for ch := 0; ch < encoderWidth; ch++ {
				if math.Float32bits(dst[r*encoderWidth+ch]) != math.Float32bits(want[ch*frames+r]) {
					t.Fatal("depth mismatch")
				}
			}
		}
		if !reflect.DeepEqual(public.Snapshot(), direct.Snapshot()) {
			t.Fatal("history changed")
		}
		before := direct.Snapshot()
		copy := direct
		if err := copy.updateRowMajor(rowMajor, frames, weights, dst); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(before, direct.Snapshot()) {
			t.Fatal("prepared update changed original")
		}
		bad := append([]float32(nil), rowMajor...)
		bad[0] = float32(math.NaN())
		if err := direct.updateRowMajor(bad, frames, weights, dst); err == nil {
			t.Fatal("non-finite input accepted")
		}
		if !reflect.DeepEqual(before, direct.Snapshot()) {
			t.Fatal("validation changed history")
		}
	}
}
