package vulkan

import (
	"context"
	"fmt"
	"math"
	"testing"
)

// Invoked only by the explicitly gated native speech harness.
func nativeSpeechConv(t *testing.T) {
	op, err := NewVkConv1D3F32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, op)
	a := nativeArena(t)
	for _, s := range [][4]int{{1, 1, 1, 1}, {1, 2, 3, 2}, {2, 3, 2, 2}, {15, 5, 17, 1}, {16, 6, 15, 2}, {17, 17, 19, 1}, {33, 31, 7, 2}, {65, 80, 17, 2}, {9, 128, 33, 1}, {3, 2048, 2, 2}, {4096, 1, 1, 1}} {
		length, in, out, stride := s[0], s[1], s[2], s[3]
		rows := (length + stride - 1) / stride
		cf, w, b := nativeData(length*in, 5, .5), nativeData(out*in*3, 6, .5), nativeData(out, 7, .1)
		tm := make([]float32, len(cf))
		for ic := 0; ic < in; ic++ {
			for pos := 0; pos < length; pos++ {
				tm[pos*in+ic] = cf[ic*length+pos]
			}
		}
		var previous []float32
		for _, layout := range []VkConvInputLayout{VkConvChannelsFirst, VkConvTimeMajor} {
			left := nativeGuard(t, a)
			data := cf
			shape := []int{in, length}
			if layout == VkConvTimeMajor {
				data = tm
				shape = []int{length, in}
			}
			x, tw, tb := nativeTensor(t, a, data, shape...), nativeTensor(t, a, w, out, in, 3), nativeTensor(t, a, b, out)
			y := nativeTensor(t, a, nil, rows, out)
			right := nativeGuard(t, a)
			elapsed := nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, y, x, tw, tb, stride, layout) })
			got := nativeDownload(t, y)
			nativeCompare(t, fmt.Sprintf("conv%v/layout=%d", s, layout), got, conv3Reference(data, w, b, length, in, out, stride, layout), 2e-5, 2e-5, elapsed)
			if previous != nil {
				for i := range got {
					if math.Float32bits(previous[i]) != math.Float32bits(got[i]) {
						t.Fatal("nativeconv layout mismatch", s, i)
					}
				}
			}
			previous = got
			left()
			right()
		}
	}
}
func nativeSpeechAdd(t *testing.T) {
	op, err := NewVkAddF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, op)
	a := nativeArena(t)
	for _, n := range []int{1, 3, 255, 256, 257, 513} {
		left := nativeGuard(t, a)
		x, y := nativeData(n, 15, 1), nativeData(n, 16, 1)
		tx, ty, out := nativeTensor(t, a, x, n), nativeTensor(t, a, y, n), nativeTensor(t, a, nil, n)
		right := nativeGuard(t, a)
		for mode := 0; mode < 5; mode++ {
			if err := tx.Upload(context.Background(), x); err != nil {
				t.Fatal(err)
			}
			if err := ty.Upload(context.Background(), y); err != nil {
				t.Fatal(err)
			}
			dst, b := out, ty
			if mode == 1 {
				dst = tx
			}
			if mode == 2 {
				dst = ty
			}
			if mode == 3 {
				dst = tx
				b = tx
			}
			if mode == 4 {
				b = tx
			}
			ref := make([]float64, n)
			for i := range ref {
				other := y[i]
				if mode >= 3 {
					other = x[i]
				}
				ref[i] = float64(x[i] + other)
			}
			elapsed := nativeRun(t, func(ctx context.Context) error { return op.Forward(ctx, dst, tx, b) })
			nativeCompare(t, fmt.Sprintf("add/n=%d/mode=%d", n, mode), nativeDownload(t, dst), ref, 0, 0, elapsed)
			left()
			right()
		}
	}
}
func nativeSpeechStem(t *testing.T) {
	for _, s := range [3][3]int{{17, 2, 3}, {65, 80, 64}, {129, 128, 64}} {
		if !t.Run(fmt.Sprint(s), func(t *testing.T) { nativeSpeechStemShape(t, s[0], s[1], s[2]) }) {
			return
		}
	}
}
func nativeSpeechStemShape(t *testing.T, length, mel, width int) {
	conv, err := NewVkConv1D3F32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, conv)
	gelu, err := NewVkGELUErfF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, gelu)
	add, err := NewVkAddF32(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, add)
	a := nativeArena(t)
	left := nativeGuard(t, a)
	rows := (length + 1) / 2
	x, w1, b1, w2, b2, pos := nativeData(mel*length, 41, 1), nativeData(width*mel*3, 42, .05), nativeData(width, 43, .1), nativeData(width*width*3, 44, .05), nativeData(width, 45, .1), nativeData(rows*width, 46, .1)
	tx, tw1, tb1, tw2, tb2, tp := nativeTensor(t, a, x, mel, length), nativeTensor(t, a, w1, width, mel, 3), nativeTensor(t, a, b1, width), nativeTensor(t, a, w2, width, width, 3), nativeTensor(t, a, b2, width), nativeTensor(t, a, pos, rows, width)
	c1, c2 := nativeTensor(t, a, nil, length, width), nativeTensor(t, a, nil, rows, width)
	right := nativeGuard(t, a)
	stages := []VkF32Stage{}
	appendStage := func(s VkF32Stage, e error) {
		if e != nil {
			t.Fatal(e)
		}
		stages = append(stages, s)
	}
	appendStage(conv.Stage(context.Background(), c1, tx, tw1, tb1, 1, VkConvChannelsFirst))
	appendStage(gelu.Stage(context.Background(), c1, c1))
	appendStage(conv.Stage(context.Background(), c2, c1, tw2, tb2, 2, VkConvTimeMajor))
	appendStage(gelu.Stage(context.Background(), c2, c2))
	appendStage(add.Stage(context.Background(), c2, c2, tp))
	plan, err := NewVkF32Plan(context.Background(), stages)
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, plan)
	ref := nativeF32(conv3Reference(x, w1, b1, length, mel, width, 1, VkConvChannelsFirst))
	ref = nativeF32(nativeGELURef(ref))
	ref = nativeF32(conv3Reference(ref, w2, b2, length, width, width, 2, VkConvTimeMajor))
	ref = nativeF32(nativeGELURef(ref))
	want := make([]float64, len(ref))
	for i := range want {
		want[i] = float64(ref[i] + pos[i])
	}
	for i := 0; i < 3; i++ {
		elapsed := nativeRun(t, plan.Run)
		nativeCompare(t, fmt.Sprintf("stem/%dx%dx%d/repeat=%d", length, mel, width, i), nativeDownload(t, c2), want, 5e-5, 5e-5, elapsed)
		left()
		right()
	}
	separate := func(ctx context.Context) error {
		if err := conv.Forward(ctx, c1, tx, tw1, tb1, 1, VkConvChannelsFirst); err != nil {
			return err
		}
		if err := gelu.Forward(ctx, c1, c1); err != nil {
			return err
		}
		if err := conv.Forward(ctx, c2, c1, tw2, tb2, 2, VkConvTimeMajor); err != nil {
			return err
		}
		if err := gelu.Forward(ctx, c2, c2); err != nil {
			return err
		}
		return add.Forward(ctx, c2, c2, tp)
	}
	nativeRun(t, separate)
	before := nativeDownload(t, c2)
	nativeRun(t, plan.Run)
	after := nativeDownload(t, c2)
	for i, v := range after {
		if math.Float32bits(v) != math.Float32bits(before[i]) {
			t.Fatal("stem plan vs separate", i, v, before[i])
		}
	}
	t.Log("native stem plan vs five separate dispatches: bit-exact")
	left()
	right()
}
