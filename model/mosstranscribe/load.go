package mosstranscribe

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/rcarmo/go-pherence/loader/weights"
	"github.com/rcarmo/go-pherence/models/whisper"
)

const modelPrefix = "model."

// AudioBackbone owns the mapped checkpoint source and the native audio graph.
// Close must be called after inference; adaptor BF16 weights alias the mmap.
type AudioBackbone struct {
	Config     Config
	Encoder    *whisper.Encoder
	GPUEncoder *whisper.GPUEncoder
	GPUAdaptor *GPUAdaptor
	Adaptor    AdaptorWeights
	source     weights.Source
}

// LoadAudioBackbone loads the exact Whisper encoder and VQ adaptor from a MOSS
// Hugging Face model directory. The encoder currently widens BF16 parameters to
// F32; adaptor matrices remain zero-copy BF16 and use SIMD dot kernels.
func LoadAudioBackbone(modelDir string) (*AudioBackbone, error) {
	cfg, err := LoadConfig(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return nil, err
	}
	source, err := weights.OpenSafetensors(modelDir)
	if err != nil {
		return nil, fmt.Errorf("MOSS open weights: %w", err)
	}
	fail := func(err error) (*AudioBackbone, error) {
		_ = source.Close()
		return nil, err
	}
	encoder, err := whisper.LoadEncoderSource(source, modelPrefix+"whisper_encoder", cfg.WhisperConfig())
	if err != nil {
		return fail(err)
	}
	adaptor, err := loadAdaptorWeights(source)
	if err != nil {
		return fail(err)
	}
	return &AudioBackbone{Config: cfg, Encoder: encoder, Adaptor: adaptor, source: source}, nil
}

// EnableGPU uploads the Whisper encoder's reusable weights to the existing
// runtime-loaded NVIDIA backend. It leaves the CPU encoder intact for fallback.
func (m *AudioBackbone) EnableGPU() bool {
	if m == nil || m.Encoder == nil {
		return false
	}
	if m.GPUEncoder == nil {
		m.GPUEncoder = whisper.NewGPUEncoder(m.Encoder, m.Config.WhisperConfig())
	}
	if m.GPUAdaptor == nil {
		m.GPUAdaptor = NewGPUAdaptor(m.Adaptor)
	}
	return m.GPUEncoder.Ready() || m.GPUAdaptor.Ready()
}

func (m *AudioBackbone) Close() error {
	if m == nil {
		return nil
	}
	if m.GPUEncoder != nil {
		m.GPUEncoder.Close()
		m.GPUEncoder = nil
	}
	if m.GPUAdaptor != nil {
		m.GPUAdaptor.Close()
		m.GPUAdaptor = nil
	}
	if m.source == nil {
		return nil
	}
	err := m.source.Close()
	m.source = nil
	return err
}

func loadAdaptorWeights(source weights.Source) (AdaptorWeights, error) {
	if source == nil {
		return AdaptorWeights{}, fmt.Errorf("MOSS: nil adaptor tensor source")
	}
	bf16 := func(suffix string, shape ...int) ([]uint16, error) {
		name := modelPrefix + "vq_adaptor." + suffix
		raw, dtype, gotShape, err := source.GetRaw(name)
		if err != nil {
			return nil, fmt.Errorf("MOSS load %s: %w", name, err)
		}
		if dtype != "BF16" || !shapeEqual(gotShape, shape) {
			return nil, fmt.Errorf("MOSS tensor %s dtype=%s shape=%v, want BF16 %v", name, dtype, gotShape, shape)
		}
		count := shapeProduct(shape)
		if count < 0 || len(raw) != count*2 {
			return nil, fmt.Errorf("MOSS tensor %s has %d bytes, want %d", name, len(raw), count*2)
		}
		if count == 0 {
			return nil, nil
		}
		if uintptr(unsafe.Pointer(&raw[0]))%unsafe.Alignof(uint16(0)) != 0 || binary.NativeEndian.Uint16([]byte{1, 0}) != 1 {
			copyValues := make([]uint16, count)
			for i := range copyValues {
				copyValues[i] = binary.LittleEndian.Uint16(raw[i*2:])
			}
			return copyValues, nil
		}
		return unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), count), nil
	}
	f32 := func(suffix string, shape ...int) ([]float32, error) {
		name := modelPrefix + "vq_adaptor." + suffix
		values, gotShape, err := source.GetFloat32(name)
		if err != nil {
			return nil, fmt.Errorf("MOSS load %s: %w", name, err)
		}
		if !shapeEqual(gotShape, shape) {
			return nil, fmt.Errorf("MOSS tensor %s shape=%v, want %v", name, gotShape, shape)
		}
		return values, nil
	}

	var out AdaptorWeights
	var err error
	if out.Linear1Weight, err = bf16("layers.0.weight", AdaptorHiddenDim, AdaptorInputDim); err != nil {
		return AdaptorWeights{}, err
	}
	if out.Linear1Bias, err = f32("layers.0.bias", AdaptorHiddenDim); err != nil {
		return AdaptorWeights{}, err
	}
	if out.Linear2Weight, err = bf16("layers.2.weight", AdaptorHiddenDim, AdaptorHiddenDim); err != nil {
		return AdaptorWeights{}, err
	}
	if out.Linear2Bias, err = f32("layers.2.bias", AdaptorHiddenDim); err != nil {
		return AdaptorWeights{}, err
	}
	if out.NormWeight, err = f32("layers.3.weight", AdaptorHiddenDim); err != nil {
		return AdaptorWeights{}, err
	}
	if out.NormBias, err = f32("layers.3.bias", AdaptorHiddenDim); err != nil {
		return AdaptorWeights{}, err
	}
	return out, nil
}

func shapeEqual(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func shapeProduct(shape []int) int {
	product := 1
	for _, dim := range shape {
		if dim <= 0 || product > int(^uint(0)>>1)/dim {
			return -1
		}
		product *= dim
	}
	return product
}

// CheckModelDirectory reports whether the minimum native files exist without
// mapping the checkpoint. It is useful for CLI capability diagnostics.
func CheckModelDirectory(modelDir string) error {
	for _, name := range []string{"config.json", "model.safetensors.index.json"} {
		if _, err := os.Stat(filepath.Join(modelDir, name)); err != nil {
			return fmt.Errorf("MOSS model directory missing %s: %w", name, err)
		}
	}
	return nil
}
