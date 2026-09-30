package nemotronasr

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/backends/vulkan"
)

// This deterministic mock owns persistent input/output storage. Its projection
// is an independent single-nonzero-column oracle, not DenseNTTo or a GPU kernel.
type scopedProjectionMock struct {
	input, output          []float32
	produceErr, consumeErr error
	malformed              int
	calls                  int
}

func (m *scopedProjectionMock) ScopedProjectionEnabled() bool { return true }
func (m *scopedProjectionMock) Project(context.Context, []float32, []float32, []float32, int) ([]float32, error) {
	panic("copied projection called")
}
func (m *scopedProjectionMock) ProjectScoped(ctx context.Context, w, b []float32, rows int, produce, consume func([]float32) error) error {
	m.calls++
	if m.input == nil {
		m.input = make([]float32, rows*4352)
		m.output = make([]float32, rows*1024)
	}
	input := m.input
	if m.malformed == 1 {
		input = input[:len(input)-1]
	}
	if err := produce(input); err != nil {
		return err
	}
	if m.produceErr != nil {
		return m.produceErr
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < 1024; col++ {
			m.output[row*1024+col] = m.input[row*4352+col]*w[col*4352+col] + b[col]
		}
	}
	output := m.output
	if m.malformed == 2 {
		output = output[:len(output)-1]
	}
	if m.malformed == 3 {
		output[0] = float32(math.NaN())
	}
	if m.consumeErr != nil {
		return m.consumeErr
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return consume(output)
}
func syntheticSharedSubsampling() *Subsampling {
	m := &Subsampling{Stem: &StemConv2D{Weight: make([]float32, 256*9), Bias: make([]float32, 256)}, linearWeight: make([]float32, 1024*4352), linearBias: make([]float32, 1024)}
	for ch := 0; ch < 256; ch++ {
		m.Stem.Weight[ch*9+4] = 0.125
		m.Stem.Bias[ch] = float32(ch%7) * 0.01
	}
	for l := range m.layers {
		stage := &m.layers[l]
		stage.depthWeight = make([]float32, 256*9)
		stage.depthBias = make([]float32, 256)
		stage.pointWeight = make([]float32, 256*256)
		stage.pointBias = make([]float32, 256)
		for ch := 0; ch < 256; ch++ {
			stage.depthWeight[ch*9+4] = 0.5
			stage.pointWeight[ch*256+ch] = 0.25
			stage.pointBias[ch] = float32(ch%5) * 0.005
		}
	}
	for col := 0; col < 1024; col++ {
		m.linearWeight[col*4352+col] = 0.25
		m.linearBias[col] = float32(col%13) * 0.001
	}
	return m
}
func syntheticSharedFeatures(rows int) []float32 {
	v := make([]float32, rows*128)
	for i := range v {
		v[i] = float32(i%31-15) * 0.03125
	}
	return v
}
func assertSharedParity(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d/%d", len(got), len(want))
	}
	for i, v := range got {
		if math.Abs(float64(v-want[i])) > 1e-7 {
			t.Fatalf("projection[%d] %g/%g", i, v, want[i])
		}
	}
}
func TestSubsamplingSharedDirectProducerConsumerParity(t *testing.T) {
	model := syntheticSharedSubsampling()
	cpu := SubsamplingStream{Model: model}
	mock := &scopedProjectionMock{}
	shared := SubsamplingStream{Model: model, Projector: mock}
	for _, rows := range []int{25, 32, 32} {
		features := syntheticSharedFeatures(rows)
		before := append([]float32(nil), features...)
		want, err := cpu.ForwardUnmaskedChunk(features, rows)
		if err != nil {
			t.Fatal(err)
		}
		consumed := false
		err = shared.WithUnmaskedChunkContext(context.Background(), features, rows, func(v []float32) error {
			consumed = true
			if &v[0] != &mock.output[0] {
				t.Fatal("consumer received copied output")
			}
			assertSharedParity(t, v, want)
			return nil
		})
		if err != nil || !consumed {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(features, before) || !reflect.DeepEqual(shared.last, cpu.last) {
			t.Fatal("input/cache parity")
		}
	}
	if mock.calls != 3 {
		t.Fatal("wrong shared call count")
	}
	// Public owned-return API must never leak mapped scratch.
	got, err := shared.ForwardUnmaskedChunk(syntheticSharedFeatures(32), 32)
	if err != nil {
		t.Fatal(err)
	}
	before := append([]float32(nil), got...)
	if &got[0] == &mock.output[0] {
		t.Fatal("owned output aliases mapping")
	}
	clear(mock.output)
	if !reflect.DeepEqual(got, before) {
		t.Fatal("owned output changed on reuse")
	}
}
func TestSubsamplingSharedFailuresLeaveCachesUnchanged(t *testing.T) {
	sentinel := errors.New("injected failure")
	model := syntheticSharedSubsampling()
	for _, kind := range []string{"producer", "projection", "consumer", "input-length", "output-length", "nonfinite", "cancel"} {
		t.Run(kind, func(t *testing.T) {
			mock := &scopedProjectionMock{}
			s := SubsamplingStream{Model: model, Projector: mock}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch kind {
			case "producer":
				mock.produceErr = sentinel
			case "projection":
				mock.consumeErr = sentinel
			case "input-length":
				mock.malformed = 1
			case "output-length":
				mock.malformed = 2
			case "nonfinite":
				mock.malformed = 3
			}
			err := s.WithUnmaskedChunkContext(ctx, syntheticSharedFeatures(25), 25, func([]float32) error {
				if kind == "consumer" {
					return sentinel
				}
				if kind == "cancel" {
					cancel()
				}
				return nil
			})
			if err == nil || s.started || s.mode != 0 || !reflect.DeepEqual(s.last, [3][]float32{}) {
				t.Fatalf("failure advanced state: %v", err)
			}
			if kind == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			mock.produceErr = nil
			mock.consumeErr = nil
			mock.malformed = 0
			if _, err := s.ForwardUnmaskedChunk(syntheticSharedFeatures(25), 25); err != nil {
				t.Fatal("fresh retry", err)
			}
		})
	}
}
func TestSubsamplingScopedCPUConsumerAndValidation(t *testing.T) {
	model := syntheticSharedSubsampling()
	s := SubsamplingStream{Model: model}
	sentinel := errors.New("consumer")
	if err := s.WithUnmaskedChunkContext(context.Background(), syntheticSharedFeatures(25), 25, func([]float32) error { return sentinel }); !errors.Is(err, sentinel) || s.started {
		t.Fatal(err)
	}
	for _, test := range []struct {
		ctx     context.Context
		frames  int
		consume func([]float32) error
	}{
		{nil, 25, func([]float32) error { return nil }},
		{context.Background(), 25, nil},
		{context.Background(), 32, func([]float32) error { return nil }},
	} {
		if err := s.WithUnmaskedChunkContext(test.ctx, syntheticSharedFeatures(test.frames), test.frames, test.consume); err == nil {
			t.Fatal("accepted invalid scoped call")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := s.WithUnmaskedChunkContext(ctx, syntheticSharedFeatures(25), 25, func([]float32) error { return nil }); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := s.WithUnmaskedChunkContext(context.Background(), syntheticSharedFeatures(25), 25, func(v []float32) error {
		if len(v) != 4096 {
			t.Fatal("shape")
		}
		return nil
	}); err != nil || !s.started {
		t.Fatal(err)
	}
}
func TestDeviceSharedProjectionValidationWithoutDevice(t *testing.T) {
	p := &DeviceSubsamplingProjector{Backend: "vulkan", SharedMemory: true}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	w, b := make([]float32, 1024*4352), make([]float32, 1024)
	noop := func([]float32) error { return nil }
	if err := p.ProjectScoped(ctx, w, b, 4, noop, noop); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		p                *DeviceSubsamplingProjector
		ctx              context.Context
		rows             int
		produce, consume func([]float32) error
	}{
		{nil, context.Background(), 4, noop, noop},
		{p, nil, 4, noop, noop},
		{p, context.Background(), 3, noop, noop},
		{p, context.Background(), 4, nil, noop},
		{p, context.Background(), 4, noop, nil},
		{&DeviceSubsamplingProjector{Backend: "ptx", SharedMemory: true}, context.Background(), 4, noop, noop},
		{&DeviceSubsamplingProjector{Backend: "vulkan"}, context.Background(), 4, noop, noop},
		{&DeviceSubsamplingProjector{Backend: "vulkan", SharedMemory: true, closed: true}, context.Background(), 4, noop, noop},
	} {
		if err := tc.p.ProjectScoped(tc.ctx, w, b, tc.rows, tc.produce, tc.consume); err == nil {
			t.Fatal("invalid shared call")
		}
	}
	w[0] = float32(math.Inf(1))
	if err := p.ProjectScoped(context.Background(), w, b, 4, noop, noop); err == nil {
		t.Fatal("nonfinite weight")
	}
	w[0] = 0
	b[0] = float32(math.NaN())
	if err := p.ProjectScoped(context.Background(), w, b, 4, noop, noop); err == nil {
		t.Fatal("nonfinite bias")
	}
	// Prepared tensors with invalid owners exercise errors without native calls.
	p.ready = true
	p.vkX = &vulkan.VkTensorF32{}
	if err := p.ProjectScoped(context.Background(), w, b, 4, noop, noop); err == nil {
		t.Fatal("invalid storage")
	}
	if p.dispatches != 0 {
		t.Fatal("published unsuccessful dispatch")
	}
	if _, err := p.Project(ctx, make([]float32, 4*4352), w, b, 4); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := p.Project(context.Background(), nil, w, b, 4); err == nil {
		t.Fatal("invalid owned input")
	}
}

func TestDeviceScopedProjectionSelection(t *testing.T) {
	var nilProjector *DeviceSubsamplingProjector
	if nilProjector.ScopedProjectionEnabled() {
		t.Fatal("nil enabled")
	}
	if (&DeviceSubsamplingProjector{Backend: "vulkan"}).ScopedProjectionEnabled() {
		t.Fatal("default shared")
	}
	if !(&DeviceSubsamplingProjector{Backend: "vulkan", SharedMemory: true}).ScopedProjectionEnabled() {
		t.Fatal("explicit opt-in disabled")
	}
}

func TestSubsamplingSharedFailurePreservesAcknowledgedChunk(t *testing.T) {
	mock := &scopedProjectionMock{}
	s := SubsamplingStream{Model: syntheticSharedSubsampling(), Projector: mock}
	if _, err := s.ForwardUnmaskedChunk(syntheticSharedFeatures(25), 25); err != nil {
		t.Fatal(err)
	}
	before := s
	mock.consumeErr = errors.New("projection failed")
	if _, err := s.ForwardUnmaskedChunk(syntheticSharedFeatures(32), 32); err == nil {
		t.Fatal("failure ignored")
	}
	if !reflect.DeepEqual(s.last, before.last) || s.started != before.started || s.mode != before.mode {
		t.Fatal("acknowledged caches changed")
	}
	mock.consumeErr = nil
	if _, err := s.ForwardUnmaskedChunk(syntheticSharedFeatures(32), 32); err != nil {
		t.Fatal(err)
	}
}
