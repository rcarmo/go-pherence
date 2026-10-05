package silero

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"

	assets "github.com/rcarmo/go-pherence/loader/silero"
)

// Synthetic inventory, independent of the file parser. Nonzero gate biases
// supply a closed-form recurrent oracle without calling the model under test.
func fixture() *assets.File {
	f := &assets.File{Version: [3]int{6, 2, 0}, Window: 512, Context: 64, Tensors: map[string]assets.Tensor{}}
	put := func(name string, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		f.Tensors[name] = assets.Tensor{Shape: append([]int(nil), shape...), Values: make([]float32, n)}
	}
	put("_model.stft.forward_basis_buffer", 256, 1, 258)
	channels := [5]int{129, 128, 64, 64, 128}
	for i := 0; i < 4; i++ {
		prefix := fmt.Sprintf("_model.encoder.%d.reparam_conv", i)
		put(prefix+".weight", 3, channels[i], channels[i+1])
		put(prefix+".bias", channels[i+1])
	}
	put("_model.decoder.rnn.weight_ih", 128, 512)
	put("_model.decoder.rnn.weight_hh", 128, 512)
	put("_model.decoder.rnn.bias_ih", 512)
	put("_model.decoder.rnn.bias_hh", 512)
	put("_model.decoder.decoder.2.weight", 128)
	put("_model.decoder.decoder.2.bias")
	for i := 256; i < 384; i++ {
		f.Tensors["_model.decoder.rnn.bias_ih"].Values[i] = 1
	}
	f.Tensors["_model.decoder.decoder.2.weight"].Values[0] = 1
	return f
}
func mustModel(t *testing.T) *Model {
	t.Helper()
	m, err := New(fixture())
	if err != nil {
		t.Fatal(err)
	}
	return m
}
func TestSyntheticRecurrentOracleAndFreshState(t *testing.T) {
	file := fixture()
	m, err := New(file)
	if err != nil {
		t.Fatal(err)
	}
	// Ownership: mutations to the loaded file must not alter model weights.
	file.Tensors["_model.decoder.rnn.bias_ih"].Values[256] = 9
	s, err := m.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]float32, 512)
	pcm[4] = 0.1234
	before := append([]float32(nil), pcm...)
	var cell float64
	for step := 0; step < 4; step++ {
		got, err := s.Probability(context.Background(), pcm)
		if err != nil {
			t.Fatal(err)
		}
		cell = 0.5*cell + 0.5*math.Tanh(1)
		hidden := float32(0.5 * math.Tanh(cell))
		// Source final conv rounds its activation to F16. Independent binary16
		// nearby value gives a format-derived error budget of <=2e-4 probability.
		want := 1 / (1 + math.Exp(-float64(hidden)))
		if math.Abs(float64(got)-want) > 2e-4 {
			t.Fatalf("step%d probability%g oracle%g", step, got, want)
		}
	}
	if !reflect.DeepEqual(pcm, before) {
		t.Fatal("PCM mutated")
	}
	s.Reset()
	fresh, _ := m.NewStream()
	a, err := s.Probability(context.Background(), pcm)
	if err != nil {
		t.Fatal(err)
	}
	b, err := fresh.Probability(context.Background(), pcm)
	if err != nil || a != b {
		t.Fatal("fresh/reset state differs", a, b, err)
	}
}
func TestReflectSTFTAndChannelMajorConvolution(t *testing.T) {
	m := mustModel(t)
	// Delta bases: real frequency0 selects padded index0, imaginary selects1.
	m.basis[0] = 1
	m.basis[129*256+1] = 1
	s, _ := m.NewStream()
	pcm := make([]float32, 512)
	for i := range pcm {
		pcm[i] = float32(i) / 512
	}
	if _, err := s.Probability(context.Background(), pcm); err != nil {
		t.Fatal(err)
	}
	want := float32(math.Sqrt(float64(pcm[64]*pcm[64] + pcm[63]*pcm[63])))
	if math.Abs(float64(s.magnitude[0]-want)) > 1e-7 {
		t.Fatal("reflect/STFT", s.magnitude[0], want)
	}
	if s.padded[0] != pcm[64] || s.padded[63] != pcm[1] || s.padded[576] != pcm[510] || s.padded[639] != pcm[447] {
		t.Fatal("reflection boundary")
	}
	layer := convLayer{input: 2, output: 2, stride: 2, weight: []float32{0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1}, bias: []float32{0.5, -0.5}}
	input := []float32{1, 2, 3, 4, 5, 6}
	out := make([]float32, 4)
	if err := s.convolution(context.Background(), out, input, 3, 2, layer); err != nil {
		t.Fatal(err)
	}
	// row-major weights with channel-major input. stride2 padding gives frames0,2.
	wantOut := []float32{9.5, 15.5, 4.5, 1.5}
	if !reflect.DeepEqual(out, wantOut) {
		t.Fatal("convolution layout", out, wantOut)
	}
}

type cancelAtCheck struct{ calls, at int }

func (c *cancelAtCheck) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *cancelAtCheck) Done() <-chan struct{}       { return nil }
func (c *cancelAtCheck) Value(any) any               { return nil }
func (c *cancelAtCheck) Err() error {
	c.calls++
	if c.calls >= c.at {
		return context.Canceled
	}
	return nil
}
func TestFrameFailureIsTransactional(t *testing.T) {
	m := mustModel(t)
	s, _ := m.NewStream()
	pcm := make([]float32, 512)
	if _, err := s.Probability(context.Background(), pcm); err != nil {
		t.Fatal(err)
	}
	h, c := s.hidden, s.cell
	for _, at := range []int{1, 2, 3, 4, 5, 6, 7} {
		ctx := &cancelAtCheck{at: at}
		if _, err := s.Probability(ctx, pcm); !errors.Is(err, context.Canceled) {
			t.Fatal("cancel checkpoint", at, err)
		}
		if s.hidden != h || s.cell != c {
			t.Fatal("cancellation committed state")
		}
	}
	for _, bad := range [][]float32{nil, pcm[:511], append(make([]float32, 511), float32(math.NaN())), append(make([]float32, 511), float32(math.Inf(1)))} {
		if _, err := s.Probability(context.Background(), bad); err == nil {
			t.Fatal("bad PCM")
		}
		if s.hidden != h || s.cell != c {
			t.Fatal("bad PCM committed")
		}
	}
	if _, err := s.Probability(nil, pcm); err == nil {
		t.Fatal("nil context")
	}
	if _, err := (*Stream)(nil).Probability(context.Background(), pcm); err == nil {
		t.Fatal("nil stream")
	}
	if _, err := (&Stream{}).Probability(context.Background(), pcm); err == nil {
		t.Fatal("nil model")
	}
	m.layers[0].bias[0] = float32(math.Inf(1))
	if _, err := s.Probability(context.Background(), pcm); err == nil {
		t.Fatal("nonfinite convolution")
	}
	if s.hidden != h {
		t.Fatal("failure committed")
	}
	m.layers[0].bias[0] = 0
	m.inputBias[0] = float32(math.NaN())
	if _, err := s.Probability(context.Background(), pcm); err == nil {
		t.Fatal("nonfinite recurrent")
	}
	m.inputBias[0] = 0
	m.finalBias = float32(math.NaN())
	if _, err := s.Probability(context.Background(), pcm); err == nil {
		t.Fatal("nonfinite probability")
	}
	(*Stream)(nil).Reset()
	if _, err := (*Model)(nil).NewStream(); err == nil {
		t.Fatal("nil model stream")
	}
}
func TestNewRejectsMalformedInventory(t *testing.T) {
	if _, err := New(nil); err == nil {
		t.Fatal("nil model")
	}
	for _, mutate := range []func(*assets.File){
		func(f *assets.File) { f.Version[0] = 7 }, func(f *assets.File) { f.Window = 256 }, func(f *assets.File) { f.Context = 0 }, func(f *assets.File) { delete(f.Tensors, "_model.decoder.decoder.2.bias") },
		func(f *assets.File) {
			x := f.Tensors["_model.stft.forward_basis_buffer"]
			x.Shape[0] = 255
			f.Tensors["_model.stft.forward_basis_buffer"] = x
		},
		func(f *assets.File) {
			x := f.Tensors["_model.decoder.rnn.weight_ih"]
			x.Values = x.Values[:1]
			f.Tensors["_model.decoder.rnn.weight_ih"] = x
		},
		func(f *assets.File) { f.Tensors["_model.encoder.1.reparam_conv.bias"].Values[0] = float32(math.Inf(1)) },
		func(f *assets.File) {
			x := f.Tensors["_model.decoder.decoder.2.bias"]
			x.Shape = []int{1}
			f.Tensors["_model.decoder.decoder.2.bias"] = x
		},
	} {
		f := fixture()
		mutate(f)
		if m, err := New(f); err == nil || m != nil {
			t.Fatal("bad inventory")
		}
	}
	for _, name := range []string{"_model.stft.forward_basis_buffer", "_model.encoder.0.reparam_conv.weight", "_model.encoder.0.reparam_conv.bias", "_model.decoder.rnn.weight_ih", "_model.decoder.decoder.2.bias"} {
		f := fixture()
		delete(f.Tensors, name)
		f.Tensors["unexpected"] = assets.Tensor{}
		if _, err := New(f); err == nil {
			t.Fatal("missing tensor")
		}
	}
}
func TestFrameSteadyStateNoAllocations(t *testing.T) {
	m := mustModel(t)
	s, _ := m.NewStream()
	pcm := make([]float32, 512)
	ctx := context.Background()
	if a := testing.AllocsPerRun(10, func() {
		if _, err := s.Probability(ctx, pcm); err != nil {
			panic(err)
		}
	}); a != 0 {
		t.Fatal("allocations/frame", a)
	}
}
