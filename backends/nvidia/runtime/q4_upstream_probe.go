package nvidia

import (
	"fmt"
	"unsafe"
)

// ProbeQ4UpstreamMMQ runs a pinned upstream Q4_K tile for validation.
func ProbeQ4UpstreamMMQ(out, x, raw *Buffer, batch, inDim, outDim int) error {
	var fn CUfunction
	switch {
	case batch <= 8:
		fn = fnQ4UpstreamMMQJ8
	case batch <= 16:
		fn = fnQ4UpstreamMMQJ16
	case batch <= 24:
		fn = fnQ4UpstreamMMQJ24
	case batch <= 32:
		fn = fnQ4UpstreamMMQJ32
	case batch <= 64:
		fn = fnQ4UpstreamMMQJ64
	default:
		fn = fnQ4UpstreamMMQJ64
	}
	if fn == 0 || out == nil || x == nil || raw == nil || fnQuantizeQ81MMQ == 0 {
		return fmt.Errorf("invalid upstream Q4 MMQ probe")
	}
	blocks := inDim / 128
	tileRows := upstreamQ4TileRows(batch)
	stride := (batch + tileRows - 1) / tileRows * tileRows
	q8, err := MallocBytes(stride * blocks * 144)
	if err != nil {
		return err
	}
	defer q8.Free()
	kk, bb, ss := uint32(inDim), uint32(batch), uint32(stride)
	if err = LaunchKernel(fnQuantizeQ81MMQ, uint32(inDim/128), uint32(batch), 1, 32, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&q8.Ptr), unsafe.Pointer(&kk), unsafe.Pointer(&bb), unsafe.Pointer(&ss)); err != nil {
		return err
	}
	nn := uint32(outDim)
	return LaunchKernel(fn, uint32((outDim+127)/128), uint32((batch+upstreamQ4TileRows(batch)-1)/upstreamQ4TileRows(batch)), 1, 32, 8, 1, uint32(upstreamQ4SharedBytes(batch)), unsafe.Pointer(&raw.Ptr), unsafe.Pointer(&q8.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&kk), unsafe.Pointer(&bb))
}

func upstreamQ4TileRows(batch int) int {
	j := 8
	if batch > 8 {
		j = 16
	}
	if batch > 16 {
		j = 24
	}
	if batch > 24 {
		j = 32
	}
	if batch > 32 {
		j = 64
	}
	return j
}
func upstreamQ4SharedBytes(batch int) int {
	j := upstreamQ4TileRows(batch)
	pad := func(n, a int) int { return (n + a - 1) / a * a }
	return j*4 + 128*76*4 + pad(j*144, 1024)
}
