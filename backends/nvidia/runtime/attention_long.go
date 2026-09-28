package nvidia

import (
	"fmt"
	"sync"
	"unsafe"
)

// Long attention is bounded by the public decision context contract. Its
// scratch is <=8 MiB regardless of query rows; rows are launched in chunks.
const MaxDecisionAttentionTokens = 32768
const longAttentionScratchBytes = 8 << 20

var attnLongFn CUfunction
var longAttentionMu sync.Mutex

func longAttention(out, q, k, v, pk, pv, sk, sv, index *Buffer, rows, pos0, kvlen, window, heads, kvheads, dim, prefix, stride, seqstart, seqlen, mode, scorestride int, scale float32) error {
	if attnLongFn == 0 || scorestride < 1 || scorestride > MaxDecisionAttentionTokens || heads < 1 || heads > 64 || rows < 1 {
		return fmt.Errorf("invalid long attention launch")
	}
	// The lock fences scratch ownership across any concurrent low-level callers.
	longAttentionMu.Lock()
	defer longAttentionMu.Unlock()
	chunk := min(rows, max(1, longAttentionScratchBytes/(heads*scorestride*4)))
	scratch, err := Malloc(chunk * heads * scorestride)
	if err != nil {
		return err
	}
	defer scratch.Free()
	defer SyncAll()
	ptr := func(b *Buffer) CUdeviceptr {
		if b == nil {
			return 0
		}
		return b.Ptr
	}
	qp, kp, vp, pkp, pvp, skp, svp, ip, op := ptr(q), ptr(k), ptr(v), ptr(pk), ptr(pv), ptr(sk), ptr(sv), ptr(index), ptr(out)
	rr, p0, kl, ww, hh, kk, dd := uint32(rows), uint32(pos0), uint32(kvlen), uint32(window), uint32(heads), uint32(kvheads), uint32(dim)
	pre, st, ss, sl, mo, sc := uint32(prefix), uint32(stride), uint32(seqstart), uint32(seqlen), uint32(mode), uint32(scorestride)
	for offset := 0; offset < rows; offset += chunk {
		off := uint32(offset)
		if err := LaunchKernel(attnLongFn, hh, uint32(min(chunk, rows-offset)), 1, 256, 1, 1, 0,
			unsafe.Pointer(&qp), unsafe.Pointer(&kp), unsafe.Pointer(&vp), unsafe.Pointer(&pkp), unsafe.Pointer(&pvp), unsafe.Pointer(&skp), unsafe.Pointer(&svp), unsafe.Pointer(&ip), unsafe.Pointer(&op), unsafe.Pointer(&scratch.Ptr),
			unsafe.Pointer(&rr), unsafe.Pointer(&p0), unsafe.Pointer(&kl), unsafe.Pointer(&ww), unsafe.Pointer(&hh), unsafe.Pointer(&kk), unsafe.Pointer(&dd), unsafe.Pointer(&pre), unsafe.Pointer(&st), unsafe.Pointer(&ss), unsafe.Pointer(&sl), unsafe.Pointer(&mo), unsafe.Pointer(&off), unsafe.Pointer(&sc), unsafe.Pointer(&scale)); err != nil {
			return err
		}
	}
	return SyncErr()
}
