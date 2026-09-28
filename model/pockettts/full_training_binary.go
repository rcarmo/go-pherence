package pockettts

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// The binary checkpoint keeps the versioned JSON format available for small
// fixtures, but does not encode released-size F32 arrays as decimal text.
// All integers and F32 bits use little endian. The final SHA-256 covers every
// preceding byte, including the header, names, lengths and floating values.
var fullTrainingBinaryMagic = [16]byte{'P', 'T', 'T', 'S', 'F', '3', '2', 0, 1}

const maxBinaryTrainingTensors = 4096
const maxBinaryTrainingNameBytes = 256

// SaveFullTrainingStateBinary atomically publishes a checked, non-executable
// binary checkpoint. It does not mutate or consume state.
func SaveFullTrainingStateBinary(path string, state FullTrainingState) error {
	if err := state.Validate(); err != nil {
		return err
	}
	dir := filepath.Dir(path)
	info, err := os.Stat(dir)
	if err != nil || !info.IsDir() {
		return fmt.Errorf("Pocket TTS binary checkpoint requires an existing directory %q: %v", dir, err)
	}
	f, err := os.CreateTemp(dir, ".pockettts-full-binary-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	buffer := bufio.NewWriterSize(f, 256<<10)
	if err = writeFullTrainingBinary(buffer, state); err != nil {
		f.Close()
		return err
	}
	if err = buffer.Flush(); err != nil {
		f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	if err = os.Rename(f.Name(), path); err != nil {
		return err
	}
	// As with the JSON saver, a failure after rename reports uncertainty:
	// the destination is published but its directory entry may not be durable.
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	if err = d.Sync(); err != nil {
		d.Close()
		return err
	}
	return d.Close()
}

// LoadFullTrainingStateBinary requires a positive caller-owned file-size
// ceiling. The ceiling must include the digest and metadata, not only F32 data.
func LoadFullTrainingStateBinary(path string, maxBytes int64) (FullTrainingState, error) {
	if maxBytes <= 0 {
		return FullTrainingState{}, fmt.Errorf("Pocket TTS binary checkpoint limit must be positive")
	}
	f, err := os.Open(path)
	if err != nil {
		return FullTrainingState{}, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return FullTrainingState{}, err
	}
	if !info.Mode().IsRegular() || info.Size() > maxBytes {
		return FullTrainingState{}, fmt.Errorf("Pocket TTS binary checkpoint is not a regular file within the byte limit")
	}
	// A forged length in a small file must not allocate against a much larger
	// caller ceiling before the read encounters EOF.
	return ReadFullTrainingStateBinary(f, info.Size())
}

// ReadFullTrainingStateBinary bounds bytes and element counts before allocating
// tensor storage. It rejects truncation, corruption, trailing data and invalid
// state. LoadState must only be called on its successful result.
func ReadFullTrainingStateBinary(reader io.Reader, maxBytes int64) (FullTrainingState, error) {
	var state FullTrainingState
	if reader == nil || maxBytes <= 0 || maxBytes == math.MaxInt64 {
		return state, fmt.Errorf("invalid Pocket TTS binary checkpoint reader or limit")
	}
	limited := &io.LimitedReader{R: reader, N: maxBytes + 1}
	checksum := sha256.New()
	input := io.TeeReader(limited, checksum)
	var magic [16]byte
	if _, err := io.ReadFull(input, magic[:]); err != nil {
		return state, err
	}
	if magic != fullTrainingBinaryMagic {
		return state, fmt.Errorf("invalid Pocket TTS binary checkpoint header")
	}
	var header [36]byte
	if _, err := io.ReadFull(input, header[:]); err != nil {
		return state, err
	}
	state.Version = int(binary.LittleEndian.Uint32(header[0:4]))
	step := binary.LittleEndian.Uint64(header[4:12])
	if step > uint64(math.MaxInt) {
		return state, fmt.Errorf("Pocket TTS binary checkpoint step overflow")
	}
	state.Step = int(step)
	state.AdamW = AdamWConfig{
		LearningRate: math.Float32frombits(binary.LittleEndian.Uint32(header[12:16])),
		Beta1:        math.Float32frombits(binary.LittleEndian.Uint32(header[16:20])),
		Beta2:        math.Float32frombits(binary.LittleEndian.Uint32(header[20:24])),
		Epsilon:      math.Float32frombits(binary.LittleEndian.Uint32(header[24:28])),
		WeightDecay:  math.Float32frombits(binary.LittleEndian.Uint32(header[28:32])),
	}
	state.EMADecay = math.Float32frombits(binary.LittleEndian.Uint32(header[32:36]))
	// Charge every tensor element against the explicit file ceiling. This
	// prevents hostile length fields from driving unbounded allocations.
	remainingElements := uint64(maxBytes / 4)
	readFloat32 := func(length uint64) ([]float32, error) {
		if length == 0 || length > remainingElements || length > uint64(math.MaxInt) {
			return nil, fmt.Errorf("Pocket TTS binary checkpoint tensor length exceeds limit")
		}
		remainingElements -= length
		values := make([]float32, int(length))
		var bytes [32 << 10]byte
		for i := 0; i < len(values); {
			count := min(len(values)-i, len(bytes)/4)
			if _, err := io.ReadFull(input, bytes[:count*4]); err != nil {
				return nil, err
			}
			for j := 0; j < count; j++ {
				values[i+j] = math.Float32frombits(binary.LittleEndian.Uint32(bytes[j*4:]))
				if !isFinite(values[i+j]) {
					return nil, fmt.Errorf("Pocket TTS binary checkpoint has non-finite tensor value")
				}
			}
			i += count
		}
		return values, nil
	}
	readName := func() (string, error) {
		var length [2]byte
		if _, err := io.ReadFull(input, length[:]); err != nil {
			return "", err
		}
		n := int(binary.LittleEndian.Uint16(length[:]))
		if n == 0 || n > maxBinaryTrainingNameBytes {
			return "", fmt.Errorf("Pocket TTS binary checkpoint name length exceeds limit")
		}
		name := make([]byte, n)
		_, err := io.ReadFull(input, name)
		return string(name), err
	}
	readGroup := func(adam bool) ([]NamedTrainingTensor, []NamedAdamState, error) {
		var countBytes [4]byte
		if _, err := io.ReadFull(input, countBytes[:]); err != nil {
			return nil, nil, err
		}
		count := binary.LittleEndian.Uint32(countBytes[:])
		if count > maxBinaryTrainingTensors {
			return nil, nil, fmt.Errorf("Pocket TTS binary checkpoint tensor count exceeds limit")
		}
		var tensors []NamedTrainingTensor
		var moments []NamedAdamState
		for i := uint32(0); i < count; i++ {
			name, err := readName()
			if err != nil {
				return nil, nil, err
			}
			var size [8]byte
			if _, err = io.ReadFull(input, size[:]); err != nil {
				return nil, nil, err
			}
			length := binary.LittleEndian.Uint64(size[:])
			values, err := readFloat32(length)
			if err != nil {
				return nil, nil, err
			}
			if adam {
				second, err := readFloat32(length)
				if err != nil {
					return nil, nil, err
				}
				moments = append(moments, NamedAdamState{Name: name, M: values, V: second})
			} else {
				tensors = append(tensors, NamedTrainingTensor{Name: name, Values: values})
			}
		}
		return tensors, moments, nil
	}
	var err error
	if state.Params, _, err = readGroup(false); err != nil {
		return FullTrainingState{}, err
	}
	if state.Buffers, _, err = readGroup(false); err != nil {
		return FullTrainingState{}, err
	}
	if _, state.Adam, err = readGroup(true); err != nil {
		return FullTrainingState{}, err
	}
	if state.EMA, _, err = readGroup(false); err != nil {
		return FullTrainingState{}, err
	}
	var digest [sha256.Size]byte
	if _, err = io.ReadFull(limited, digest[:]); err != nil {
		return FullTrainingState{}, err
	}
	var actual [sha256.Size]byte
	copy(actual[:], checksum.Sum(nil))
	if digest != actual {
		return FullTrainingState{}, fmt.Errorf("Pocket TTS binary checkpoint checksum mismatch")
	}
	var extra [1]byte
	if n, readErr := limited.Read(extra[:]); n != 0 || (readErr != nil && !errors.Is(readErr, io.EOF)) {
		return FullTrainingState{}, fmt.Errorf("Pocket TTS binary checkpoint contains trailing data")
	}
	if limited.N < 1 {
		return FullTrainingState{}, fmt.Errorf("Pocket TTS binary checkpoint exceeds byte limit")
	}
	return state, state.Validate()
}

func writeFullTrainingBinary(writer io.Writer, state FullTrainingState) error {
	checksum := sha256.New()
	output := io.MultiWriter(writer, checksum)
	if _, err := output.Write(fullTrainingBinaryMagic[:]); err != nil {
		return err
	}
	var header [36]byte
	binary.LittleEndian.PutUint32(header[0:4], uint32(state.Version))
	binary.LittleEndian.PutUint64(header[4:12], uint64(state.Step))
	for i, v := range [...]float32{state.AdamW.LearningRate, state.AdamW.Beta1, state.AdamW.Beta2, state.AdamW.Epsilon, state.AdamW.WeightDecay, state.EMADecay} {
		binary.LittleEndian.PutUint32(header[12+i*4:], math.Float32bits(v))
	}
	if _, err := output.Write(header[:]); err != nil {
		return err
	}
	writeGroup := func(tensors []NamedTrainingTensor, adam []NamedAdamState) error {
		count := len(tensors) + len(adam)
		if count > maxBinaryTrainingTensors {
			return fmt.Errorf("Pocket TTS binary checkpoint tensor count exceeds limit")
		}
		var number [8]byte
		binary.LittleEndian.PutUint32(number[:4], uint32(count))
		if _, err := output.Write(number[:4]); err != nil {
			return err
		}
		writeValues := func(values []float32) error {
			var bytes [32 << 10]byte
			for i := 0; i < len(values); {
				count := min(len(values)-i, len(bytes)/4)
				for j, v := range values[i : i+count] {
					binary.LittleEndian.PutUint32(bytes[j*4:], math.Float32bits(v))
				}
				if _, err := output.Write(bytes[:count*4]); err != nil {
					return err
				}
				i += count
			}
			return nil
		}
		writeTensor := func(name string, first, second []float32) error {
			if len(name) == 0 || len(name) > maxBinaryTrainingNameBytes || len(first) == 0 || (second != nil && len(second) != len(first)) {
				return fmt.Errorf("invalid Pocket TTS binary checkpoint tensor %q", name)
			}
			binary.LittleEndian.PutUint16(number[:2], uint16(len(name)))
			if _, err := output.Write(number[:2]); err != nil {
				return err
			}
			if _, err := io.WriteString(output, name); err != nil {
				return err
			}
			binary.LittleEndian.PutUint64(number[:8], uint64(len(first)))
			if _, err := output.Write(number[:8]); err != nil {
				return err
			}
			if err := writeValues(first); err != nil {
				return err
			}
			if second != nil {
				return writeValues(second)
			}
			return nil
		}
		for _, tensor := range tensors {
			if err := writeTensor(tensor.Name, tensor.Values, nil); err != nil {
				return err
			}
		}
		for _, moment := range adam {
			if err := writeTensor(moment.Name, moment.M, moment.V); err != nil {
				return err
			}
		}
		return nil
	}
	for _, tensors := range [][]NamedTrainingTensor{state.Params, state.Buffers} {
		if err := writeGroup(tensors, nil); err != nil {
			return err
		}
	}
	if err := writeGroup(nil, state.Adam); err != nil {
		return err
	}
	if err := writeGroup(state.EMA, nil); err != nil {
		return err
	}
	digest := checksum.Sum(nil)
	n, err := writer.Write(digest)
	if err == nil && n != len(digest) {
		return io.ErrShortWrite
	}
	return err
}
