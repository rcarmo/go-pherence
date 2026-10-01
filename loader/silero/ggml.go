// Package silero reads the pinned legacy GGML Silero 16-kHz VAD format.
// It does not initialise a backend or download assets. This format is not GGUF.
package silero

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/rcarmo/go-pherence/half"
)

const MaxFileBytes = 2 << 20
const RevisionSHA256 = "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987"

// Tensor values are owned, contiguous F32, in source output/channel/kernel
// order. Shape retains the GGML dimension order (fastest dimension first).
// Callers must keep weights immutable while models or streams use them.
type Tensor struct {
	Shape  []int
	Values []float32
}
type File struct {
	Version         [3]int
	Window, Context int
	Tensors         map[string]Tensor
}

type tensorSpec struct {
	name  string
	dtype int
	shape []int
}

func specifications() []tensorSpec {
	specs := make([]tensorSpec, 0, 15)
	channels := [5]int{129, 128, 64, 64, 128}
	for layer := 0; layer < 4; layer++ {
		prefix := fmt.Sprintf("_model.encoder.%d.reparam_conv", layer)
		specs = append(specs, tensorSpec{prefix + ".weight", 1, []int{3, channels[layer], channels[layer+1]}}, tensorSpec{prefix + ".bias", 0, []int{channels[layer+1]}})
	}
	return append(specs,
		tensorSpec{"_model.decoder.rnn.weight_ih", 0, []int{128, 512}},
		tensorSpec{"_model.decoder.rnn.weight_hh", 0, []int{128, 512}},
		tensorSpec{"_model.decoder.rnn.bias_ih", 0, []int{512}},
		tensorSpec{"_model.decoder.rnn.bias_hh", 0, []int{512}},
		tensorSpec{"_model.decoder.decoder.2.weight", 1, []int{128}},
		tensorSpec{"_model.decoder.decoder.2.bias", 0, []int{}},
		tensorSpec{"_model.stft.forward_basis_buffer", 1, []int{256, 1, 258}})
}

// LoadPath requires a full SHA-256 pin and decodes only after checking it.
// File closure is joined into failures; a cancelled/malformed file returns no
// partial model. Default tests use synthetic bytes rather than a local cache.
func LoadPath(ctx context.Context, path, pin string) (*File, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Silero: nil context")
	}
	raw, err := hex.DecodeString(pin)
	if err != nil || len(raw) != 32 || pin != strings.ToLower(pin) {
		return nil, fmt.Errorf("Silero: require lowercase SHA256 pin")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	data, err := io.ReadAll(io.LimitReader(file, MaxFileBytes+1))
	closeErr := file.Close()
	if err := errors.Join(err, closeErr); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(data) > MaxFileBytes {
		return nil, fmt.Errorf("Silero: file exceeds %d bytes", MaxFileBytes)
	}
	digest := sha256.Sum256(data)
	if hex.EncodeToString(digest[:]) != pin {
		return nil, fmt.Errorf("Silero: SHA256 mismatch")
	}
	return Parse(ctx, data)
}

type cursor struct {
	data   []byte
	offset int
}

func (r *cursor) take(n int) ([]byte, error) {
	if n < 0 || n > len(r.data)-r.offset {
		return nil, io.ErrUnexpectedEOF
	}
	b := r.data[r.offset : r.offset+n]
	r.offset += n
	return b, nil
}
func (r *cursor) integer() (int, error) {
	b, err := r.take(4)
	if err != nil {
		return 0, err
	}
	return int(int32(binary.LittleEndian.Uint32(b))), nil
}
func (r *cursor) expect(want int) error {
	v, err := r.integer()
	if err != nil {
		return err
	}
	if v != want {
		return fmt.Errorf("Silero: header field=%d want=%d", v, want)
	}
	return nil
}

// Parse admits the 6.2.0 16-kHz geometry and exact tensor inventory, including
// the final rank-zero scalar bias. It bounds all lengths before allocation,
// rejects unsupported dtypes/nonfinite weights, and rejects trailing bytes.
// Parse verifies structure; LoadPath additionally verifies asset identity.
func Parse(ctx context.Context, data []byte) (*File, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Silero: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(data) == 0 || len(data) > MaxFileBytes {
		return nil, fmt.Errorf("Silero: invalid file size")
	}
	r := cursor{data: data}
	if err := r.expect(0x67676d6c); err != nil {
		return nil, err
	}
	if err := r.expect(len("silero-16k")); err != nil {
		return nil, err
	}
	name, err := r.take(len("silero-16k"))
	if err != nil {
		return nil, err
	}
	if string(name) != "silero-16k" {
		return nil, fmt.Errorf("Silero: unsupported model type")
	}
	// Header layout is fixed and interleaves each layer's input/output/kernel.
	for _, want := range []int{6, 2, 0, 512, 64, 4, 129, 128, 3, 128, 64, 3, 64, 64, 3, 64, 128, 3, 128, 128, 128, 1} {
		if err := r.expect(want); err != nil {
			return nil, err
		}
	}
	specs := specifications()
	expected := make(map[string]tensorSpec, len(specs))
	for _, s := range specs {
		expected[s.name] = s
	}
	model := &File{Version: [3]int{6, 2, 0}, Window: 512, Context: 64, Tensors: make(map[string]Tensor, len(specs))}
	for range specs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		rank, err := r.integer()
		if err != nil {
			return nil, err
		}
		nameLen, err := r.integer()
		if err != nil {
			return nil, err
		}
		dtype, err := r.integer()
		if err != nil {
			return nil, err
		}
		if rank < 0 || rank > 3 || nameLen < 1 || nameLen > 128 || (dtype != 0 && dtype != 1) {
			return nil, fmt.Errorf("Silero: tensor header rejected")
		}
		shape := make([]int, rank)
		elements := 1
		for i := range shape {
			d, err := r.integer()
			if err != nil {
				return nil, err
			}
			if d < 1 || d > 512 || elements > MaxFileBytes/4/d {
				return nil, fmt.Errorf("Silero: tensor dimension/size rejected")
			}
			shape[i] = d
			elements *= d
		}
		encodedName, err := r.take(nameLen)
		if err != nil {
			return nil, err
		}
		tensorName := string(encodedName)
		spec, ok := expected[tensorName]
		if !ok || spec.dtype != dtype || len(spec.shape) != rank {
			return nil, fmt.Errorf("Silero: unexpected tensor %q", tensorName)
		}
		if _, duplicate := model.Tensors[tensorName]; duplicate {
			return nil, fmt.Errorf("Silero: duplicate tensor %q", tensorName)
		}
		for i, d := range shape {
			if d != spec.shape[i] {
				return nil, fmt.Errorf("Silero: tensor %q shape rejected", tensorName)
			}
		}
		bytesPerElement := 4
		if dtype == 1 {
			bytesPerElement = 2
		}
		payload, err := r.take(elements * bytesPerElement)
		if err != nil {
			return nil, err
		}
		values := make([]float32, elements)
		for i := range values {
			if i%4096 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			if dtype == 1 {
				values[i] = half.F16ToF32(binary.LittleEndian.Uint16(payload[i*2:]))
			} else {
				values[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[i*4:]))
			}
			if math.IsNaN(float64(values[i])) || math.IsInf(float64(values[i]), 0) {
				return nil, fmt.Errorf("Silero: tensor %q has nonfinite weights", tensorName)
			}
		}
		model.Tensors[tensorName] = Tensor{Shape: shape, Values: values}
	}
	if r.offset != len(data) {
		return nil, fmt.Errorf("Silero: trailing data")
	}
	return model, nil
}
