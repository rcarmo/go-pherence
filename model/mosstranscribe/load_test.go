package mosstranscribe

import (
	"encoding/binary"
	"errors"
	"testing"
)

type fakeMOSSRawTensor struct {
	raw   []byte
	dtype string
	shape []int
	f32   []float32
}

type fakeMOSSSource struct {
	tensors map[string]fakeMOSSRawTensor
}

func (s *fakeMOSSSource) GetRaw(name string) ([]byte, string, []int, error) {
	tensor, ok := s.tensors[name]
	if !ok {
		return nil, "", nil, errors.New("missing")
	}
	return tensor.raw, tensor.dtype, tensor.shape, nil
}
func (s *fakeMOSSSource) GetFloat32(name string) ([]float32, []int, error) {
	tensor, ok := s.tensors[name]
	if !ok {
		return nil, nil, errors.New("missing")
	}
	return tensor.f32, tensor.shape, nil
}
func (*fakeMOSSSource) GetInt32(string) ([]int32, []int, error) {
	return nil, nil, errors.New("unsupported")
}
func (*fakeMOSSSource) Close() error { return nil }

func TestLoadAdaptorWeightsExactTensorContract(t *testing.T) {
	source := completeAdaptorSource()
	weights, err := loadAdaptorWeights(source)
	if err != nil {
		t.Fatal(err)
	}
	if !weights.valid() {
		t.Fatal("loaded invalid adaptor")
	}
	binary.LittleEndian.PutUint16(source.tensors[modelPrefix+"vq_adaptor.layers.0.weight"].raw, 0x3f80)
	if weights.Linear1Weight[0] != 0x3f80 {
		t.Fatal("BF16 adaptor matrix does not alias mapped source")
	}
}

func TestLoadAdaptorWeightsRejectsWrongDTypeAndShape(t *testing.T) {
	t.Run("dtype", func(t *testing.T) {
		source := completeAdaptorSource()
		tensor := source.tensors[modelPrefix+"vq_adaptor.layers.2.weight"]
		tensor.dtype = "F16"
		source.tensors[modelPrefix+"vq_adaptor.layers.2.weight"] = tensor
		if _, err := loadAdaptorWeights(source); err == nil {
			t.Fatal("accepted F16 adaptor weight")
		}
	})
	t.Run("shape", func(t *testing.T) {
		source := completeAdaptorSource()
		tensor := source.tensors[modelPrefix+"vq_adaptor.layers.3.bias"]
		tensor.shape = []int{512}
		source.tensors[modelPrefix+"vq_adaptor.layers.3.bias"] = tensor
		if _, err := loadAdaptorWeights(source); err == nil {
			t.Fatal("accepted incorrect adaptor shape")
		}
	})
}

func completeAdaptorSource() *fakeMOSSSource {
	s := &fakeMOSSSource{tensors: map[string]fakeMOSSRawTensor{}}
	addBF16 := func(name string, shape ...int) {
		n := shapeProduct(shape)
		s.tensors[modelPrefix+"vq_adaptor."+name] = fakeMOSSRawTensor{raw: make([]byte, n*2), dtype: "BF16", shape: shape}
	}
	addF32 := func(name string, shape ...int) {
		n := shapeProduct(shape)
		s.tensors[modelPrefix+"vq_adaptor."+name] = fakeMOSSRawTensor{dtype: "BF16", shape: shape, f32: make([]float32, n)}
	}
	addBF16("layers.0.weight", AdaptorHiddenDim, AdaptorInputDim)
	addF32("layers.0.bias", AdaptorHiddenDim)
	addBF16("layers.2.weight", AdaptorHiddenDim, AdaptorHiddenDim)
	addF32("layers.2.bias", AdaptorHiddenDim)
	addF32("layers.3.weight", AdaptorHiddenDim)
	addF32("layers.3.bias", AdaptorHiddenDim)
	return s
}
