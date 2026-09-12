package community1

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type resnetSource struct {
	infos       map[string]safetensors.TensorInfo
	values      map[string][]float32
	seen        []string
	after       func(string)
	fail        error
	reuse       []float32
	reuseValues bool
}

func (s *resnetSource) TensorInfos() map[string]safetensors.TensorInfo { return s.infos }
func (s *resnetSource) GetFloat32(name string) ([]float32, []int, error) {
	s.seen = append(s.seen, name)
	if s.after != nil {
		s.after(name)
	}
	if s.fail != nil {
		return nil, nil, s.fail
	}
	values, ok := s.values[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing floating payload %s", name)
	}
	if s.reuseValues {
		if cap(s.reuse) < len(values) {
			s.reuse = make([]float32, len(values))
		}
		s.reuse = s.reuse[:len(values)]
		copy(s.reuse, values)
		return s.reuse, s.infos[name].Shape, nil
	}
	return values, s.infos[name].Shape, nil
}

// Independent schema from actual synthetic Torch state_dict, with independent
// value wiring from the graph fixture (not the loader's binding function).
func resnetLoadFixture(t *testing.T, index int, prefix string, counters bool) (*resnetSource, resnetCase) {
	t.Helper()
	data, err := os.ReadFile("testdata/resnet-schema.json")
	if err != nil {
		t.Fatal(err)
	}
	var schema struct {
		Schema int
		SHA    string `json:"resnet_sha256"`
		Cases  []struct {
			Config  WeSpeakerResNetConfig
			Tensors map[string]safetensors.TensorInfo
		}
	}
	if err := json.Unmarshal(data, &schema); err != nil {
		t.Fatal(err)
	}
	if schema.Schema != 1 || len(schema.Cases) != 3 || schema.SHA != "2de7673e14e8c74d6c430e0b6e6cc844157f47ac35b67ff6238ca2d2c559eba9" {
		t.Fatal("schema contract")
	}
	c := loadResNetFixtures(t)[index]
	if c.Config != schema.Cases[index].Config {
		t.Fatal("schema/graph config mismatch")
	}
	s := &resnetSource{infos: make(map[string]safetensors.TensorInfo), values: make(map[string][]float32)}
	pre := prefix
	if pre != "" {
		pre += "."
	}
	add := func(name string, values []float32) { s.values[pre+name] = values }
	bn := func(name string, b WeSpeakerBN) {
		add(name+".weight", b.Weight)
		add(name+".bias", b.Bias)
		add(name+".running_mean", b.RunningMean)
		add(name+".running_var", b.RunningVariance)
	}
	add("conv1.weight", c.Weights.Stem)
	bn("bn1", c.Weights.StemBN)
	for stage, blocks := range c.Weights.Stages {
		for i, b := range blocks {
			name := fmt.Sprintf("layer%d.%d", stage+1, i)
			add(name+".conv1.weight", b.Conv1)
			add(name+".conv2.weight", b.Conv2)
			bn(name+".bn1", b.BN1)
			bn(name+".bn2", b.BN2)
			if len(b.Shortcut) > 0 {
				add(name+".shortcut.0.weight", b.Shortcut)
				bn(name+".shortcut.1", b.ShortcutBN)
			}
		}
	}
	add("seg_1.weight", c.Weights.Projection.Weight)
	add("seg_1.bias", c.Weights.Projection.Bias)
	names := make([]string, 0)
	for name := range schema.Cases[index].Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	offset := 0
	for _, name := range names {
		info := schema.Cases[index].Tensors[name]
		if info.DType == "I64" && !counters {
			continue
		}
		count := 1
		for _, dim := range info.Shape {
			count *= dim
		}
		bytes := 4
		if info.DType == "I64" {
			bytes = 8
		}
		info.DataOffsets = [2]int{offset, offset + bytes*count}
		offset += bytes * count
		s.infos[pre+name] = info
	}
	return s, c
}

func TestWeSpeakerLoadSchemaAndOwnedGraphParity(t *testing.T) {
	for _, prefix := range []string{"", "resnet"} {
		for _, counters := range []bool{false, true} {
			source, c := resnetLoadFixture(t, 0, prefix, counters)
			m, err := LoadWeSpeakerResNetSource(context.Background(), source, c.Config, prefix)
			if err != nil {
				t.Fatal(err)
			}
			if len(source.seen) != 182 {
				t.Fatalf("floating tensor reads got%d want182", len(source.seen))
			}
			for _, name := range source.seen {
				if strings.HasSuffix(name, "num_batches_tracked") {
					t.Fatal("training-only payload read")
				}
			}
			if counters && len(source.infos) != 218 {
				t.Fatalf("Torch full schema %d want218", len(source.infos))
			}
			// Every value is observable in the reconstructed inference graph; compare
			// against independent constructor and all source arrays before mutation.
			direct, err := NewWeSpeakerResNet34(context.Background(), c.Config, c.Weights)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(m, direct) {
				t.Fatal("loader bound a tensor to the wrong field")
			}
			for _, values := range source.values {
				for i := range values {
					values[i] = 99
				}
			}
			for _, mc := range c.MaskCases {
				out, err := m.Forward(context.Background(), c.Input, c.Frames, mc.Masks, mc.Speakers, mc.MaskFrames, WeSpeakerBlockSIMD)
				if err != nil {
					t.Fatal(err)
				}
				checkEmbedding(t, out, mc)
			}
		}
	}
	source, c := resnetLoadFixture(t, 1, "resnet", true)
	source.reuseValues = true
	m, err := LoadWeSpeakerResNetSource(context.Background(), source, c.Config, "resnet")
	if err != nil {
		t.Fatal(err)
	}
	for i := range source.reuse {
		source.reuse[i] = 999
	}
	out, err := m.Forward(context.Background(), c.Input, c.Frames, nil, 0, 0, WeSpeakerBlockSIMD)
	if err != nil {
		t.Fatal(err)
	}
	checkEmbedding(t, out, c.MaskCases[0])
}

func TestWeSpeakerLoadMetadataRejectionBeforePayload(t *testing.T) {
	for _, kind := range []string{"missing", "unknown", "transpose", "dtype", "counter_rank", "counter_dtype", "extent", "negative_offset", "overlap", "counter_overlap", "second_projection", "wrong_prefix"} {
		source, c := resnetLoadFixture(t, 0, "resnet", true)
		key := "resnet.seg_1.weight"
		info := source.infos[key]
		switch kind {
		case "missing":
			delete(source.infos, "resnet.layer4.2.bn2.bias")
		case "unknown":
			source.infos["resnet.unknown.weight"] = info
		case "transpose":
			info.Shape = []int{info.Shape[1], info.Shape[0]}
			source.infos[key] = info
		case "dtype":
			info.DType = "I32"
			source.infos[key] = info
		case "counter_rank":
			key = "resnet.bn1.num_batches_tracked"
			info = source.infos[key]
			info.Shape = []int{1}
			source.infos[key] = info
		case "counter_dtype":
			key = "resnet.bn1.num_batches_tracked"
			info = source.infos[key]
			info.DType = "F32"
			source.infos[key] = info
		case "extent":
			info.DataOffsets[1]--
			source.infos[key] = info
		case "negative_offset":
			info.DataOffsets[0] = -1
			source.infos[key] = info
		case "overlap":
			key = "resnet.bn1.weight"
			info = source.infos[key]
			n := info.DataOffsets[1] - info.DataOffsets[0]
			start := source.infos["resnet.bn1.bias"].DataOffsets[0]
			info.DataOffsets = [2]int{start, start + n}
			source.infos[key] = info
		case "counter_overlap":
			key = "resnet.bn1.num_batches_tracked"
			info = source.infos[key]
			info.DataOffsets = [2]int{0, 8}
			source.infos[key] = info
		case "second_projection":
			source.infos["resnet.seg_2.weight"] = info
		case "wrong_prefix":
			source.infos["seg_1.weight"] = info
			delete(source.infos, key)
		}
		model, err := LoadWeSpeakerResNetSource(context.Background(), source, c.Config, "resnet")
		if err == nil || model != nil || len(source.seen) != 0 {
			t.Fatalf("%s accessed weights/accepted metadata: %v %d", kind, err, len(source.seen))
		}
	}
	source, c := resnetLoadFixture(t, 0, "", false)
	for _, prefix := range []string{"module", "resnet.", "/", "../resnet"} {
		if _, err := LoadWeSpeakerResNetSource(context.Background(), source, c.Config, prefix); err == nil || len(source.seen) > 0 {
			t.Fatal("bad prefix")
		}
	}
	bad := c.Config
	bad.BaseChannels = int(^uint(0) >> 1)
	if _, err := LoadWeSpeakerResNetSource(context.Background(), source, bad, ""); err == nil || len(source.seen) > 0 {
		t.Fatal("bad config")
	}
	if _, err := LoadWeSpeakerResNetSource(context.Background(), nil, c.Config, ""); err == nil {
		t.Fatal("nil source")
	}
}

func TestWeSpeakerLoadPayloadFailureCancellation(t *testing.T) {
	for _, kind := range []string{"short", "nan", "inf", "negative_variance", "changed_shape", "source_error", "cancel"} {
		source, c := resnetLoadFixture(t, 0, "", true)
		key := "conv1.weight"
		sentinel := errors.New("payload failure")
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		switch kind {
		case "short":
			source.values[key] = source.values[key][:2]
		case "nan":
			source.values[key][0] = float32(math.NaN())
		case "inf":
			source.values[key][0] = float32(math.Inf(1))
		case "negative_variance":
			source.values["bn1.running_var"][0] = -1
		case "changed_shape":
			source.after = func(name string) {
				if name == key {
					i := source.infos[name]
					i.Shape = []int{len(source.values[name])}
					source.infos[name] = i
				}
			}
		case "source_error":
			source.fail = sentinel
		case "cancel":
			source.after = func(string) { cancel() }
		}
		model, err := LoadWeSpeakerResNetSource(ctx, source, c.Config, "")
		if model != nil || err == nil {
			t.Fatal("accepted bad payload", kind)
		}
		if kind == "source_error" && !errors.Is(err, sentinel) {
			t.Fatal("lost source cause")
		}
		if kind == "cancel" && !errors.Is(err, context.Canceled) {
			t.Fatal("lost cancellation")
		}
	}
	source, c := resnetLoadFixture(t, 0, "resnet", true)
	count := newPowersetContext(0)
	_, err := LoadWeSpeakerResNetSource(count, source, c.Config, "resnet")
	count.cancel()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("WeSpeaker loader checkpoints: %d", count.calls)
	for at := 1; at <= count.calls; at += max(1, count.calls/25) {
		ctx := newPowersetContext(at)
		m, err := LoadWeSpeakerResNetSource(ctx, source, c.Config, "resnet")
		ctx.cancel()
		if m != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("checkpoint cancellation", at, err)
		}
	}
}

func writeWeSpeakerSafetensors(t *testing.T, source *resnetSource, dtype string) string {
	t.Helper()
	infos := make(map[string]safetensors.TensorInfo, len(source.infos))
	names := make([]string, 0, len(source.infos))
	for name := range source.infos {
		names = append(names, name)
	}
	sort.Strings(names)
	payload := []byte{}
	for _, name := range names {
		info := source.infos[name]
		start := len(payload)
		if info.DType == "I64" {
			payload = append(payload, make([]byte, 8)...)
		} else {
			info.DType = dtype
			for _, value := range source.values[name] {
				if dtype == "F32" {
					var b [4]byte
					binary.LittleEndian.PutUint32(b[:], math.Float32bits(value))
					payload = append(payload, b[:]...)
				} else {
					// Fixed 1.0 is exact in both narrowed dtypes; this tests widening,
					// construction and lifetime, not quantisation/model quality.
					bits := uint16(0x3c00)
					if dtype == "BF16" {
						bits = 0x3f80
					}
					var b [2]byte
					binary.LittleEndian.PutUint16(b[:], bits)
					payload = append(payload, b[:]...)
				}
			}
		}
		info.DataOffsets = [2]int{start, len(payload)}
		infos[name] = info
	}
	header, err := json.Marshal(infos)
	if err != nil {
		t.Fatal(err)
	}
	for len(header)%8 != 0 {
		header = append(header, ' ')
	}
	data := make([]byte, 8+len(header)+len(payload))
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	copy(data[8:], header)
	copy(data[8+len(header):], payload)
	path := filepath.Join(t.TempDir(), "synthetic.safetensors")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestWeSpeakerLoadSafetensorsDTypeAndSourceClose(t *testing.T) {
	for _, dtype := range []string{"F32", "F16", "BF16"} {
		source, c := resnetLoadFixture(t, 0, "resnet", true)
		path := writeWeSpeakerSafetensors(t, source, dtype)
		file, err := safetensors.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		model, err := LoadWeSpeakerResNetSource(context.Background(), file, c.Config, "resnet")
		_ = file.Close()
		if err != nil {
			t.Fatal(err)
		}
		if dtype == "F32" {
			out, err := model.Forward(context.Background(), c.Input, c.Frames, nil, 0, 0, WeSpeakerBlockSIMD)
			if err != nil {
				t.Fatal(err)
			}
			checkEmbedding(t, out, c.MaskCases[0])
		} else {
			// Extract fields through exact deep equality with an independently assembled
			// all-one model; no inference on untrained constant weights needed.
			for name, values := range source.values {
				for i := range values {
					values[i] = 1
				}
				source.values[name] = values
			}
			expected, err := NewWeSpeakerResNet34(context.Background(), c.Config, c.Weights)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(model, expected) {
				t.Fatal("dtype widening/owned lifetime mismatch", dtype)
			}
		}
	}
}
