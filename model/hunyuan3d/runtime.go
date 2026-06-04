package hunyuan3d

import (
	"errors"
	"fmt"
	"image"
	"sort"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/tensor"
)

var ErrKernelNotImplemented = errors.New("hunyuan3d native kernel not implemented")

type LoadOptions struct {
	ConfigPath     string
	CheckpointPath string
	EagerLoad      bool
}

type Model struct {
	Config     loaderconfig.Hunyuan3DConfig
	Shape      ShapeConfig
	DiT        DiTConfig
	Condition  ConditionerConfig
	Coverage   TensorCoverage
	checkpoint *safetensors.File
	Tensors    TensorGroups
}

type TensorGroups struct {
	Model       map[string]TensorRef
	VAE         map[string]TensorRef
	Conditioner map[string]TensorRef
	Other       map[string]TensorRef
}

type TensorRef struct {
	Name  string
	DType string
	Shape []int
	file  *safetensors.File
}

type RunOptions struct {
	Steps         int
	GuidanceScale float32
	Seed          int64
	Image         ImagePreprocessConfig
}

type RunState struct {
	Image      ImagePreprocessResult
	Latents    []float32
	LatentDims []int
	Schedule   loaderconfig.Hunyuan3DFlowMatchSchedule
}

func Load(opts LoadOptions) (*Model, error) {
	if opts.ConfigPath == "" || opts.CheckpointPath == "" {
		return nil, fmt.Errorf("hunyuan3d load: config and checkpoint paths are required")
	}
	cfg, _, err := loaderconfig.ReadHunyuan3DConfig(opts.ConfigPath)
	if err != nil {
		return nil, err
	}
	shape, err := FromLoaderConfig(cfg)
	if err != nil {
		return nil, err
	}
	dit, err := DiTFromShapeConfig(shape)
	if err != nil {
		return nil, err
	}
	cond, err := ConditionerFromShapeConfig(shape)
	if err != nil {
		return nil, err
	}
	ckpt, err := safetensors.Open(opts.CheckpointPath)
	if err != nil {
		return nil, err
	}
	if opts.EagerLoad {
		if _, err := ckpt.EagerLoad(); err != nil {
			_ = ckpt.Close()
			return nil, err
		}
	}
	coverage, err := ValidateTensorCoverage(ckpt.Names())
	if err != nil {
		_ = ckpt.Close()
		return nil, err
	}
	m := &Model{Config: cfg, Shape: shape, DiT: dit, Condition: cond, Coverage: coverage, checkpoint: ckpt}
	m.Tensors = bindTensorGroups(ckpt)
	return m, nil
}

func (m *Model) Close() error {
	if m == nil || m.checkpoint == nil {
		return nil
	}
	err := m.checkpoint.Close()
	m.checkpoint = nil
	return err
}

func (m *Model) Tensor(name string) (TensorRef, bool) {
	if m == nil {
		return TensorRef{}, false
	}
	for _, group := range []map[string]TensorRef{m.Tensors.Model, m.Tensors.VAE, m.Tensors.Conditioner, m.Tensors.Other} {
		if ref, ok := group[name]; ok {
			return ref, true
		}
	}
	return TensorRef{}, false
}

func (r TensorRef) Float32() ([]float32, []int, error) {
	if r.file == nil {
		return nil, nil, fmt.Errorf("hunyuan3d tensor %q: checkpoint closed", r.Name)
	}
	return r.file.GetFloat32(r.Name)
}

func (r TensorRef) Tensor() (*tensor.Tensor, error) {
	data, shape, err := r.Float32()
	if err != nil {
		return nil, err
	}
	return tensor.FromFloat32(data, shape), nil
}

// Linear applies x @ weight^T + bias using the shared tensor/SIMD matmul path.
func Linear(x *tensor.Tensor, weight TensorRef, bias *TensorRef) (*tensor.Tensor, error) {
	w, err := weight.Tensor()
	if err != nil {
		return nil, err
	}
	var b *tensor.Tensor
	if bias != nil {
		b, err = bias.Tensor()
		if err != nil {
			return nil, err
		}
	}
	return x.Linear(w, b), nil
}

// RunImageToShape prepares the image, scheduler, and initial latents using the
// native Go code that exists today. It then returns ErrKernelNotImplemented at
// the conditioner boundary until DINO/CLIP, Hunyuan3DDiT, ShapeVAE, and mesh
// kernels are implemented on top of the existing tensor/SIMD backends.
func (m *Model) RunImageToShape(img image.Image, opts RunOptions) (RunState, error) {
	if m == nil {
		return RunState{}, fmt.Errorf("hunyuan3d run: nil model")
	}
	if opts.Steps <= 0 {
		opts.Steps = 50
	}
	if opts.GuidanceScale == 0 {
		opts.GuidanceScale = 5
	}
	if opts.Image.Size == 0 {
		opts.Image = DefaultImagePreprocessConfig()
	}
	pre, err := PreprocessImageV2(img, opts.Image)
	if err != nil {
		return RunState{}, err
	}
	latentShape, err := m.Shape.LatentShape(1)
	if err != nil {
		return RunState{}, err
	}
	latents, err := DeterministicLatents(latentShape, opts.Seed)
	if err != nil {
		return RunState{}, err
	}
	schedule, err := m.Config.FlowMatchSchedule(opts.Steps)
	if err != nil {
		return RunState{}, err
	}
	state := RunState{Image: pre, Latents: latents, LatentDims: latentShape, Schedule: schedule}
	return state, fmt.Errorf("%w: conditioner encoder, DiT denoiser, ShapeVAE decode, and mesh export", ErrKernelNotImplemented)
}

func bindTensorGroups(f *safetensors.File) TensorGroups {
	groups := TensorGroups{Model: map[string]TensorRef{}, VAE: map[string]TensorRef{}, Conditioner: map[string]TensorRef{}, Other: map[string]TensorRef{}}
	infos := f.TensorInfos()
	names := make([]string, 0, len(infos))
	for name := range infos {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		info := infos[name]
		ref := TensorRef{Name: name, DType: info.DType, Shape: append([]int(nil), info.Shape...), file: f}
		switch loaderconfig.ClassifyHunyuan3DTensorName(name) {
		case loaderconfig.Hunyuan3DTensorModel:
			groups.Model[name] = ref
		case loaderconfig.Hunyuan3DTensorVAE:
			groups.VAE[name] = ref
		case loaderconfig.Hunyuan3DTensorConditioner:
			groups.Conditioner[name] = ref
		default:
			groups.Other[name] = ref
		}
	}
	return groups
}
