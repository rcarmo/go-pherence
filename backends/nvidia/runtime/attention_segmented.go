package nvidia

import (
	"fmt"
	"unsafe"
)

var attnSegmentedFn, ropeSegmentedFn, selectedBatchFn CUfunction

// SegmentedRows owns immutable per-row sequence starts, uploaded once and reused
// by every layer. Close only after all work using this plan has completed.
type SegmentedRows struct {
	starts                *Buffer
	rows, prefix, longest int
}

// AttentionSegment optionally reads one earlier context before its own suffix.
// Parents must be complete, root-level segments; sibling branches stay isolated.
type AttentionSegment struct{ Length, ParentStart, ParentLength int }

func NewSegmentedRows(lengths []int, prefix int) (*SegmentedRows, error) {
	segments := make([]AttentionSegment, len(lengths))
	for i, n := range lengths {
		segments[i].Length = n
	}
	return NewBranchedRows(segments, prefix)
}

func NewBranchedRows(segments []AttentionSegment, prefix int) (*SegmentedRows, error) {
	if len(segments) == 0 || len(segments) > 512 || prefix < 1 || prefix >= MaxDecisionAttentionTokens {
		return nil, fmt.Errorf("invalid segmented prefix/lengths")
	}
	var ids []uint32
	rows, longest := 0, 0
	roots := map[int]int{}
	for _, segment := range segments {
		n, parent, plen := segment.Length, segment.ParentStart, segment.ParentLength
		if n < 1 || n > 512-rows || plen < 0 || plen > MaxDecisionAttentionTokens-prefix-n || parent < 0 || (plen > 0 && roots[parent] != plen) || (plen == 0 && parent != 0) {
			return nil, fmt.Errorf("invalid segment length/parent")
		}
		if plen == 0 {
			roots[rows] = n
		}
		for i := 0; i < n; i++ {
			ids = append(ids, uint32(rows), uint32(parent), uint32(plen))
		}
		rows += n
		longest = max(longest, plen+n)
	}
	buf, err := Malloc(len(ids))
	if err != nil {
		return nil, err
	}
	if err = buf.UploadUint32(ids); err != nil {
		buf.Free()
		return nil, err
	}
	return &SegmentedRows{buf, rows, prefix, longest}, nil
}
func (p *SegmentedRows) Close() {
	if p != nil && p.starts != nil {
		p.starts.Free()
		p.starts = nil
	}
}
func (p *SegmentedRows) valid() bool  { return p != nil && p.starts != nil && p.starts.Ptr != 0 }
func hasFloats(b *Buffer, n int) bool { return b != nil && b.Ptr != 0 && n > 0 && n <= b.Size/4 }

func (p *SegmentedRows) RoPE(x, table *Buffer, heads, dim, rot int) error {
	if !p.valid() || ropeSegmentedFn == 0 || heads < 1 || heads > 256 || dim < 1 || dim > 2048 || rot < 1 || rot > dim/2 || !hasFloats(x, p.rows*heads*dim) || !hasFloats(table, (p.prefix+p.longest)*rot*2) {
		return fmt.Errorf("invalid segmented RoPE")
	}
	r, pre, h, d, ro := uint32(p.rows), uint32(p.prefix), uint32(heads), uint32(dim), uint32(rot)
	return LaunchKernel(ropeSegmentedFn, uint32((p.rows*heads*rot+255)/256), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&table.Ptr), unsafe.Pointer(&p.starts.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&pre), unsafe.Pointer(&h), unsafe.Pointer(&d), unsafe.Pointer(&ro))
}

// AttentionFromPrefix reads a compacted sliding prefix while RoPE still uses
// the original absolute positions from the immutable plan.
func (p *SegmentedRows) AttentionFromPrefix(out, q, k, v, pk, pv *Buffer, base, window, heads, kvHeads, dim int, scale float32) error {
	if !p.valid() || base < 0 || base >= p.prefix || base > 0 && (window <= 0 || p.prefix-base < window-1) {
		return fmt.Errorf("invalid compacted segmented prefix")
	}
	view := *p
	view.prefix -= base
	return view.Attention(out, q, k, v, pk, pv, window, heads, kvHeads, dim, scale)
}

func (p *SegmentedRows) Attention(out, q, k, v, pk, pv *Buffer, window, heads, kvHeads, dim int, scale float32) error {
	if !p.valid() || attnSegmentedFn == 0 || window < 0 || heads < 1 || heads > 256 || kvHeads < 1 || heads%kvHeads != 0 || dim < 1 || dim > 2048 || !hasFloats(out, p.rows*heads*dim) || !hasFloats(q, p.rows*heads*dim) || !hasFloats(k, p.rows*kvHeads*dim) || !hasFloats(v, p.rows*kvHeads*dim) || !hasFloats(pk, p.prefix*kvHeads*dim) || !hasFloats(pv, p.prefix*kvHeads*dim) {
		return fmt.Errorf("invalid segmented attention")
	}
	visible := p.prefix + p.longest
	if window > 0 {
		visible = min(visible, window)
	}
	if visible > 2048 {
		return longAttention(out, q, k, v, pk, pv, nil, nil, p.starts, p.rows, 0, 0, window, heads, kvHeads, dim, p.prefix, 0, 0, 0, 2, visible, scale)
	}
	r, pre, w, h, kh, d := uint32(p.rows), uint32(p.prefix), uint32(window), uint32(heads), uint32(kvHeads), uint32(dim)
	return LaunchKernel(attnSegmentedFn, h, r, 1, 256, 1, 1, 0, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&k.Ptr), unsafe.Pointer(&v.Ptr), unsafe.Pointer(&pk.Ptr), unsafe.Pointer(&pv.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&p.starts.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&pre), unsafe.Pointer(&w), unsafe.Pointer(&h), unsafe.Pointer(&kh), unsafe.Pointer(&d), unsafe.Pointer(&scale))
}

// ProjectSelectedBatch computes only named vocabulary entries for each hidden
// row. IDs are validated on the host; scratch lives through the download fence.
func (m *GPUGGUFMatrix) ProjectSelectedBatch(x *Buffer, candidates [][]int) ([][]float32, error) {
	if m == nil || m.qk == nil || m.qk.kind != qkKindQ5K || selectedBatchFn == 0 || len(candidates) < 1 || len(candidates) > 256 || !hasFloats(x, len(candidates)*m.InDim) {
		return nil, fmt.Errorf("invalid selected batch matrix")
	}
	var ids, owners []uint32
	for row, list := range candidates {
		if len(list) < 1 || len(list) > 255 {
			return nil, fmt.Errorf("invalid selected batch count")
		}
		for _, id := range list {
			if id < 0 || id >= m.OutDim {
				return nil, fmt.Errorf("invalid candidate ID")
			}
			ids = append(ids, uint32(id))
			owners = append(owners, uint32(row))
		}
	}
	i, err := Malloc(len(ids))
	if err != nil {
		return nil, err
	}
	defer i.Free()
	o, err := Malloc(len(ids))
	if err != nil {
		return nil, err
	}
	defer o.Free()
	y, err := Malloc(len(ids))
	if err != nil {
		return nil, err
	}
	defer y.Free()
	if err = i.UploadUint32(ids); err != nil {
		return nil, err
	}
	if err = o.UploadUint32(owners); err != nil {
		return nil, err
	}
	k, n := uint32(m.InDim), uint32(len(ids))
	if err = LaunchKernel(selectedBatchFn, n, 1, 1, 32, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&m.qk.PackedQ.Ptr), unsafe.Pointer(&m.qk.PackedScale.Ptr), unsafe.Pointer(&m.qk.PackedMin.Ptr), unsafe.Pointer(&i.Ptr), unsafe.Pointer(&o.Ptr), unsafe.Pointer(&y.Ptr), unsafe.Pointer(&k), unsafe.Pointer(&n)); err != nil {
		return nil, err
	}
	flat := make([]float32, len(ids))
	if err = y.Download(flat); err != nil {
		return nil, err
	}
	out := make([][]float32, len(candidates))
	at := 0
	for j, list := range candidates {
		end := at + len(list)
		out[j] = flat[at:end:end]
		at = end
	}
	return out, nil
}
