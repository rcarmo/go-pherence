package hunyuan3d

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestLoadRuntimeAndTensorAccess(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	ckptPath := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(cfgPath, []byte(sampleConfig), 0o644); err != nil {
		t.Fatal(err)
	}
	writeRuntimeSafetensors(t, ckptPath)
	m, err := Load(LoadOptions{ConfigPath: cfgPath, CheckpointPath: ckptPath, EagerLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Shape.InChannels != 64 || m.DiT.HiddenSize != 1024 || m.Condition.Type != "DinoImageEncoder" {
		t.Fatalf("metadata shape=%+v dit=%+v conditioner=%+v", m.Shape, m.DiT, m.Condition)
	}
	if m.Coverage.Model != 2 || m.Coverage.VAE != 1 || m.Coverage.Conditioner != 1 {
		t.Fatalf("coverage=%+v", m.Coverage)
	}
	ref, ok := m.Tensor("model.linear.weight")
	if !ok {
		t.Fatal("missing tensor ref")
	}
	data, shape, err := ref.Float32()
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(shape, []int{2, 3}) || len(data) != 6 || data[5] != 6 {
		t.Fatalf("tensor data=%v shape=%v", data, shape)
	}
}

func TestRuntimeLinearUsesTensorKernel(t *testing.T) {
	dir := t.TempDir()
	ckptPath := filepath.Join(dir, "model.safetensors")
	writeRuntimeSafetensors(t, ckptPath)
	f, err := openRuntimeCheckpointForTest(ckptPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	groups := bindTensorGroups(f)
	w := groups.Model["model.linear.weight"]
	b := groups.Model["model.linear.bias"]
	x := tensor.FromFloat32([]float32{1, 2, 3}, []int{1, 3})
	y, err := Linear(x, w, &b)
	if err != nil {
		t.Fatal(err)
	}
	y.Realize()
	got := y.Data()
	want := []float32{15, 34}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("linear=%v want %v", got, want)
		}
	}
}

func TestRunImageToShapeStopsAtKernelBoundary(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	ckptPath := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(cfgPath, []byte(sampleConfig), 0o644); err != nil {
		t.Fatal(err)
	}
	writeRuntimeSafetensors(t, ckptPath)
	m, err := Load(LoadOptions{ConfigPath: cfgPath, CheckpointPath: ckptPath})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	img := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	for y := 2; y < 6; y++ {
		for x := 2; x < 6; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: 180, G: 90, B: 40, A: 255})
		}
	}
	state, err := m.RunImageToShape(img, RunOptions{Steps: 3, Seed: 42, Image: ImagePreprocessConfig{Size: 16, BorderRatio: 0.25}})
	if !errors.Is(err, ErrKernelNotImplemented) {
		t.Fatalf("err=%v want ErrKernelNotImplemented", err)
	}
	if len(state.Image.Image) != 3*16*16 || len(state.Latents) != 1*512*64 || len(state.Schedule.Sigmas) != 3 {
		t.Fatalf("state image=%d latents=%d sigmas=%d", len(state.Image.Image), len(state.Latents), len(state.Schedule.Sigmas))
	}
}

func TestLoadRejectsMissingTensorGroups(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	ckptPath := filepath.Join(dir, "bad.safetensors")
	if err := os.WriteFile(cfgPath, []byte(sampleConfig), 0o644); err != nil {
		t.Fatal(err)
	}
	writeSafetensors(t, ckptPath, map[string]testTensor{"model.only": {Shape: []int{1}, Data: []float32{1}}})
	if _, err := Load(LoadOptions{ConfigPath: cfgPath, CheckpointPath: ckptPath}); err == nil {
		t.Fatal("missing groups accepted")
	}
}

type testTensor struct {
	Shape []int
	Data  []float32
}

func openRuntimeCheckpointForTest(path string) (*safetensors.File, error) {
	return safetensors.Open(path)
}

func writeRuntimeSafetensors(t *testing.T, path string) {
	t.Helper()
	writeSafetensors(t, path, map[string]testTensor{
		"model.linear.weight":        {Shape: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}},
		"model.linear.bias":          {Shape: []int{2}, Data: []float32{1, 2}},
		"vae.decoder.weight":         {Shape: []int{1}, Data: []float32{7}},
		"conditioner.encoder.weight": {Shape: []int{1}, Data: []float32{8}},
	})
}

func writeSafetensors(t *testing.T, path string, tensors map[string]testTensor) {
	t.Helper()
	type headerTensor struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets [2]int `json:"data_offsets"`
	}
	header := map[string]headerTensor{}
	data := make([]byte, 0)
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		tt := tensors[name]
		start := len(data)
		for _, v := range tt.Data {
			var buf [4]byte
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			data = append(data, buf[:]...)
		}
		header[name] = headerTensor{DType: "F32", Shape: tt.Shape, DataOffsets: [2]int{start, len(data)}}
	}
	headerBytes, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]byte, 8, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	out = append(out, headerBytes...)
	out = append(out, data...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}
