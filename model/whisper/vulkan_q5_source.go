package whisper

import (
	"context"
	"encoding/binary"
	"fmt"
	"github.com/rcarmo/go-pherence/half"
	"math"
)

// Check original stored values against the supplied widened encoder. This is
// construction-time admission, not requantisation; no changed checkpoint slips
// through an otherwise matching geometry. Bound temporary work to one block.
func checkOriginalQ5Values(ctx context.Context, raw []byte, values []float32) error {
	if ctx == nil {
		return fmt.Errorf("whisper Q5: nil context")
	}
	if len(raw) == 0 || len(raw)%22 != 0 || len(values) != len(raw)/22*32 {
		return fmt.Errorf("whisper Q5: stored-value extent")
	}
	for b := 0; b < len(raw)/22; b++ {
		if b%1024 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		block := raw[b*22:][:22]
		bits := binary.LittleEndian.Uint16(block)
		if bits&0x7c00 == 0x7c00 {
			return fmt.Errorf("whisper Q5: nonfinite scale")
		}
		d := half.F16ToF32(bits)
		high := binary.LittleEndian.Uint32(block[2:])
		for i := 0; i < 32; i++ {
			q := uint32(block[6+i%16])
			if i < 16 {
				q &= 15
			} else {
				q >>= 4
			}
			q |= ((high >> uint(i)) & 1) << 4
			v := d * float32(int(q)-16)
			if math.Float32bits(v) != math.Float32bits(values[b*32+i]) {
				return fmt.Errorf("whisper Q5: source weight differs at%d", b*32+i)
			}
		}
	}
	return ctx.Err()
}
