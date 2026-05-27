package speaker

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type testSTEntry struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

func TestLoadECAPASafetensors(t *testing.T) {
	cfg := Config{SampleRate: 16000, NumMels: 3, Channels: []int{4, 5}, KernelSize: 3, EmbedDim: 2, SEBottleneck: 2, AttentionDim: 3}
	path := writeECAPATestSafetensors(t, map[string][]int{
		"conv0.weight":            {4, 3, 3},
		"conv0.bias":              {4},
		"blocks.0.conv.weight":    {5, 4, 3},
		"blocks.0.conv.bias":      {5},
		"blocks.0.se_down.weight": {2, 5},
		"blocks.0.se_up.weight":   {5, 2},
		"pool.attn.weight":        {3, 5},
		"pool.attn.bias":          {3},
		"pool.out.weight":         {3},
		"pool.out.bias":           {1},
		"embed.weight":            {2, 10},
		"embed.bias":              {2},
	})
	m, err := LoadECAPASafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadECAPASafetensors: %v", err)
	}
	if len(m.Conv0Weight) != 4*3*3 || len(m.Blocks) != 1 || len(m.EmbedWeight) != 20 {
		t.Fatalf("unexpected loaded sizes conv0=%d blocks=%d embed=%d", len(m.Conv0Weight), len(m.Blocks), len(m.EmbedWeight))
	}
}

func TestLoadECAPASafetensorsRejectsShapeMismatch(t *testing.T) {
	cfg := Config{SampleRate: 16000, NumMels: 3, Channels: []int{4, 5}, KernelSize: 3, EmbedDim: 2, SEBottleneck: 2, AttentionDim: 3}
	path := writeECAPATestSafetensors(t, map[string][]int{
		"conv0.weight":            {4, 3, 3},
		"conv0.bias":              {4},
		"blocks.0.conv.weight":    {5, 4, 3},
		"blocks.0.conv.bias":      {5},
		"blocks.0.se_down.weight": {2, 5},
		"blocks.0.se_up.weight":   {5, 2},
		"pool.attn.weight":        {3, 5},
		"pool.attn.bias":          {3},
		"pool.out.weight":         {3},
		"pool.out.bias":           {1},
		"embed.weight":            {2, 9},
		"embed.bias":              {2},
	})
	_, err := LoadECAPASafetensors(path, cfg)
	if err == nil || !strings.Contains(err.Error(), "embed.weight shape") {
		t.Fatalf("err=%v, want embed.weight shape mismatch", err)
	}
}

func writeECAPATestSafetensors(t *testing.T, shapes map[string][]int) string {
	t.Helper()
	header := make(map[string]testSTEntry, len(shapes))
	var data bytes.Buffer
	for name, shape := range shapes {
		count := 1
		for _, dim := range shape {
			count *= dim
		}
		start := data.Len()
		for i := 0; i < count; i++ {
			if err := binary.Write(&data, binary.LittleEndian, float32(i%7)/7); err != nil {
				t.Fatalf("write float: %v", err)
			}
		}
		header[name] = testSTEntry{DType: "F32", Shape: shape, DataOffsets: [2]int{start, data.Len()}}
	}
	h, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	var out bytes.Buffer
	if err := binary.Write(&out, binary.LittleEndian, uint64(len(h))); err != nil {
		t.Fatalf("write len: %v", err)
	}
	out.Write(h)
	out.Write(data.Bytes())
	path := filepath.Join(t.TempDir(), "ecapa.safetensors")
	if err := os.WriteFile(path, out.Bytes(), 0644); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
	return path
}
