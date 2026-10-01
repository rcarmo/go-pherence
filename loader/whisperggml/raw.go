package whisperggml

import (
	"context"
	"fmt"
)

// Q5Blocks returns owned original Q5_0 storage from this pinned open file.
// No dequantisation, mapping or conversion. Reads and Close are serialised.
func (f *File) Q5Blocks(ctx context.Context, name string) ([]byte, []int, error) {
	if ctx == nil || f == nil {
		return nil, nil, fmt.Errorf("WhisperGGML: nil raw context/file")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if f.file == nil {
		return nil, nil, fmt.Errorf("WhisperGGML: closed")
	}
	t, ok := f.tensors[name]
	if !ok || t.Type != 6 || len(t.Shape) != 2 || t.Shape[0]%32 != 0 || t.Bytes < 22 || t.Bytes > 64<<20 {
		return nil, nil, fmt.Errorf("WhisperGGML: Q5 matrix shape/type/budget")
	}
	raw := make([]byte, int(t.Bytes))
	for pos := 0; pos < len(raw); {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		end := min(pos+65536, len(raw))
		n, err := f.file.ReadAt(raw[pos:end], t.Offset+int64(pos))
		if err != nil || n != end-pos {
			return nil, nil, fmt.Errorf("WhisperGGML: raw read %w", err)
		}
		pos = end
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	return raw, append([]int(nil), t.Shape...), nil
}
