package whisper

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
	"time"
)

func vulkanToyEncoder(t *testing.T, c Config) *Encoder {
	t.Helper()
	enc, err := LoadEncoderSource(completeEncoderSource("audio", c), "audio", c)
	if err != nil {
		t.Fatal(err)
	}
	// Deterministic, non-symmetric trained-shape data, independent of map order.
	layout, err := describeVulkanEncoder(context.Background(), enc, c.MaxLength)
	if err != nil {
		t.Fatal(err)
	}
	for _, group := range layout.weights {
		for _, s := range group {
			for i := range s.data {
				v := float32(math.Sin(float64(i+len(s.name)*17)) * .04)
				if len(s.shape) == 1 && len(s.name) >= 2 && s.name[len(s.name)-2:] == ".w" {
					v += 1
				}
				s.data[i] = v
			}
		}
	}
	return enc
}
func vulkanToyConfig() Config {
	return Config{NumMelBins: 3, MaxLength: 18, EncoderLayers: 2, EncoderDModel: 8, EncoderHeads: 2, HeadDim: 4, EncoderFFNDim: 13}
}
func TestVulkanEncoderLayout(t *testing.T) {
	c := vulkanToyConfig()
	enc := vulkanToyEncoder(t, c)
	l, err := describeVulkanEncoder(context.Background(), enc, 17)
	if err != nil {
		t.Fatal(err)
	}
	if l.rows != 9 || len(l.weights) != 3 || len(l.scratch) != 9 || len(l.plans) != 4 {
		t.Fatal("layout", l.rows, len(l.weights), len(l.scratch), len(l.plans))
	}
	if len(l.plans[0]) != 5 || len(l.plans[1]) != 12 || len(l.plans[2]) != 12 || len(l.plans[3]) != 1 {
		t.Fatal("stagecounts")
	}
	defined := map[string]bool{}
	shapes := map[string][]int{}
	for _, group := range l.weights {
		for _, s := range group {
			if defined[s.name] {
				t.Fatal("duplicate", s.name)
			}
			defined[s.name] = true
			shapes[s.name] = s.shape
		}
	}
	for _, s := range l.scratch {
		if defined[s.name] {
			t.Fatal("duplicate scratch")
		}
		shapes[s.name] = s.shape
	}
	defined["mel"] = true
	for _, plan := range l.plans {
		for _, s := range plan {
			for _, input := range s.in {
				if !defined[input] {
					t.Fatal("read before write", s.op, input)
				}
			}
			if shapes[s.out] == nil {
				t.Fatal("unknown output")
			}
			defined[s.out] = true
		}
	}
	if !reflect.DeepEqual(shapes["pos"], []int{9, 8}) || !reflect.DeepEqual(shapes["stem"], []int{17, 8}) || !reflect.DeepEqual(shapes["ff"], []int{9, 13}) {
		t.Fatal(shapes)
	}
	for i := 1; i < len(l.weights); i++ {
		found := false
		for _, s := range l.weights[i] {
			if s.zero {
				found = true
				if s.name[len(s.name)-3:] != "k.b" {
					t.Fatal("unexpected zero")
				}
			}
		}
		if !found {
			t.Fatal("missing zero Kbias")
		}
	}
	for _, a := range []uint64{1, 4, 16, 256} {
		n, err := vkEncoderArenaBytes(l.scratch, a)
		if err != nil || n == 0 {
			t.Fatal(n, err)
		}
	}
	if _, err := vkEncoderArenaBytes(l.scratch, 7); err == nil {
		t.Fatal("unaligned")
	}
	if _, err := vkEncoderArenaBytes([]vkEncoderTensor{{shape: []int{int(^uint(0) >> 1), 2}}}, 256); err == nil {
		t.Fatal("overflow")
	}
	// Full-size config arithmetic and positional prefix metadata without weights.
	for _, cfg := range []Config{Tiny(), LargeV3Turbo()} {
		if cfg.HeadDim > 64 || cfg.EncoderLayers > 32 || cfg.MaxLength > 4096 {
			t.Fatal("unexpected production envelope")
		}
	}
}

func TestVulkanEncoderLayoutFinalCancellationReturnsNil(t *testing.T) {
	c := vulkanToyConfig()
	enc := vulkanToyEncoder(t, c)
	count := newCheckpointContext(0)
	layout, err := describeVulkanEncoder(count, enc, 17)
	count.cancel()
	if err != nil || layout == nil || count.calls < 1 {
		t.Fatal(layout, err, count.calls)
	}
	ctx := newCheckpointContext(count.calls)
	layout, err = describeVulkanEncoder(ctx, enc, 17)
	ctx.cancel()
	if layout != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("late cancellation returned layout", layout, err)
	}
}

func TestVulkanEncoderDefaultLinearMode(t *testing.T) {
	if vulkanDefaultLinearMode != vulkanLinearF32RegTile {
		t.Fatal("Vulkan encoder default must use qualified F32 register tile")
	}
}

func TestVulkanEncoderQ8PlacementSelection(t *testing.T) {
	for _, tc := range []struct {
		mode vulkanLinearMode
		name string
		want bool
	}{
		{vulkanLinearF32, "layer0.fc1.w", false},
		{vulkanLinearF32RegTile, "layer0.fc2.w", false},
		{vulkanLinearQ8Weight, "layer0.q.w", true},
		{vulkanLinearQ8Weight, "layer0.fc1.w", true},
		{vulkanLinearQ8MLPWeight, "layer0.fc1.w", true},
		{vulkanLinearQ8MLPWeight, "layer0.fc2.w", true},
		{vulkanLinearQ8MLPWeight, "layer0.q.w", false},
		{vulkanLinearQ8MLPWeight, "layer0.o.w", false},
		{vulkanLinearQ8KVMLPWeight, "layer0.fc1.w", true},
		{vulkanLinearQ8KVMLPWeight, "layer0.k.w", true},
		{vulkanLinearQ8KVMLPWeight, "layer0.v.w", true},
		{vulkanLinearQ8KVMLPWeight, "layer0.q.w", false},
		{vulkanLinearQ8KVMLPWeight, "layer0.o.w", false},
	} {
		if got := vulkanQ8WeightSelected(tc.mode, tc.name); got != tc.want {
			t.Fatal(tc, got)
		}
	}
}

func TestVulkanEncoderQ8RejectsBeforeDevice(t *testing.T) {
	c := vulkanToyConfig()
	enc := vulkanToyEncoder(t, c)
	if result, err := NewVulkanEncoderQ8Weight(context.Background(), enc, 0); err == nil || result != nil {
		t.Fatal("Q8 accepted invalid frames")
	}
	if result, err := NewVulkanEncoderQ8Weight(nil, enc, 17); err == nil || result != nil {
		t.Fatal("Q8 accepted nil context")
	}
	if result, err := NewVulkanEncoderQ8Weight(context.Background(), nil, 17); err == nil || result != nil {
		t.Fatal("Q8 accepted nil source")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if result, err := NewVulkanEncoderQ8Weight(ctx, enc, 17); !errors.Is(err, context.Canceled) || result != nil {
		t.Fatal("Q8 precancel", result, err)
	}
}

func TestVulkanEncoderRejectsBeforeDevice(t *testing.T) {
	c := vulkanToyConfig()
	enc := vulkanToyEncoder(t, c)
	cases := []struct {
		name  string
		alter func(*Encoder)
	}{
		{"mel", func(e *Encoder) { e.cfg.NumMelBins = 0 }}, {"layers", func(e *Encoder) { e.cfg.EncoderLayers = 33 }}, {"head", func(e *Encoder) { e.cfg.HeadDim = 65 }}, {"heads", func(e *Encoder) { e.cfg.EncoderHeads = 0 }}, {"width", func(e *Encoder) { e.cfg.EncoderDModel = 2049 }}, {"ffn", func(e *Encoder) { e.cfg.EncoderFFNDim = 16385 }}, {"oddmax", func(e *Encoder) { e.cfg.MaxLength = 17 }},
		{"missingconv", func(e *Encoder) { e.Conv1Weight = nil }}, {"convsize", func(e *Encoder) { e.Conv2Weight = e.Conv2Weight[:2] }}, {"pos", func(e *Encoder) { e.PosEmbed = e.PosEmbed[:2] }}, {"final", func(e *Encoder) { e.FinalLNWeight = nil }}, {"layercount", func(e *Encoder) { e.Layers = e.Layers[:1] }},
		{"qbias", func(e *Encoder) { e.Layers[0].QBias = nil }}, {"kbias", func(e *Encoder) { e.Layers[0].KBias = []float32{1} }}, {"fc", func(e *Encoder) { e.Layers[1].FC1Weight = nil }}, {"nan", func(e *Encoder) {
			e.Layers[0].VBias = append([]float32(nil), e.Layers[0].VBias...)
			e.Layers[0].VBias[0] = float32(math.NaN())
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := *enc
			e.Layers = append([]EncoderLayer(nil), enc.Layers...)
			tc.alter(&e)
			if result, err := NewVulkanEncoder(context.Background(), &e, 17); err == nil || result != nil {
				t.Fatal("accepted invalid source before device", result, err)
			}
		})
	}
	for _, n := range []int{0, 19} {
		if _, err := NewVulkanEncoder(context.Background(), enc, n); err == nil {
			t.Fatal("frames")
		}
	}
	if _, err := NewVulkanEncoder(nil, enc, 17); err == nil {
		t.Fatal("nil context")
	}
	if _, err := NewVulkanEncoder(context.Background(), nil, 17); err == nil {
		t.Fatal("nil source")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewVulkanEncoder(ctx, enc, 17); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	for failAt := 1; failAt < 20; failAt++ {
		checkpoint := newCheckpointContext(failAt)
		_, err := describeVulkanEncoder(checkpoint, enc, 17)
		checkpoint.cancel()
		if !errors.Is(err, context.Canceled) {
			t.Fatal("metadata scan cancellation", failAt, err)
		}
	}
}

type vkTestCloser struct {
	fail  bool
	calls int
}

func (c *vkTestCloser) Close() error {
	c.calls++
	if c.fail {
		return errors.New("retained")
	}
	return nil
}
func TestVulkanEncoderOwnership(t *testing.T) {
	var zero *VulkanEncoder
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
	if zero.Stats() != (VulkanEncoderStats{}) {
		t.Fatal("zero stats")
	}
	if _, err := zero.Forward(context.Background(), nil); err == nil {
		t.Fatal("zero forward")
	}
	c1, c2 := &vkTestCloser{}, &vkTestCloser{fail: true}
	s := &vulkanEncoderState{gate: make(chan struct{}, 1), resources: []vkEncoderCloser{c1, c2}}
	e := &VulkanEncoder{s: s}
	copy := *e
	if err := e.Close(); err == nil || !s.stopping || s.closed {
		t.Fatal("retained close")
	}
	if _, err := copy.Forward(context.Background(), nil); err == nil {
		t.Fatal("forward after partial close")
	}
	c2.fail = false
	if err := copy.Close(); err != nil {
		t.Fatal(err)
	}
	if !s.closed || c1.calls != 1 || c2.calls != 2 {
		t.Fatal("retry ownership", c1, c2)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	s = &vulkanEncoderState{gate: make(chan struct{}, 1)}
	e = &VulkanEncoder{s: s}
	s.gate <- struct{}{}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if _, err := e.acquire(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	<-s.gate
}
