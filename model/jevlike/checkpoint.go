package jevlike

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// Checkpoint is the versioned, non-executable interchange format. Frozen encoder
// weights are external; EncoderReference identifies the required backbone.
type Checkpoint struct {
	Version          int              `json:"version"`
	Encoder          string           `json:"encoder"`
	EncoderReference string           `json:"encoder_reference,omitempty"`
	Config           Config           `json:"config"`
	Parameters       []NamedParameter `json:"parameters"`
}

func TinyCheckpoint(m *TinyScorer) (Checkpoint, error) {
	if err := m.Validate(); err != nil {
		return Checkpoint{}, err
	}
	return Checkpoint{Version: 1, Encoder: "tiny", Config: m.Config, Parameters: m.NamedParameters()}, nil
}

func (c Checkpoint) Validate() error {
	if c.Version != 1 {
		return fmt.Errorf("unsupported jevlike checkpoint version %d", c.Version)
	}
	if err := c.Config.Validate(); err != nil {
		return err
	}
	var expected []NamedParameter
	switch c.Encoder {
	case "tiny":
		m, err := NewTinyScorer(c.Config)
		if err != nil {
			return err
		}
		expected = m.NamedParameters()
	case "frozen":
		if c.EncoderReference == "" {
			return fmt.Errorf("frozen checkpoint requires encoder_reference")
		}
		h, err := NewAttentionHead(c.Config.Width, c.Config.Rank)
		if err != nil {
			return err
		}
		expected = h.NamedParameters("head")
	default:
		return fmt.Errorf("unsupported encoder %q", c.Encoder)
	}
	if len(c.Parameters) != len(expected) {
		return fmt.Errorf("checkpoint has %d tensors, want %d", len(c.Parameters), len(expected))
	}
	specs := make(map[string]NamedParameter, len(expected))
	for _, p := range expected {
		specs[p.Name] = p
	}
	for _, p := range c.Parameters {
		spec, ok := specs[p.Name]
		if !ok {
			return fmt.Errorf("unexpected or duplicate tensor %q", p.Name)
		}
		delete(specs, p.Name)
		if len(p.Shape) != len(spec.Shape) || len(p.Values) != len(spec.Values) {
			return fmt.Errorf("tensor %q shape/length mismatch", p.Name)
		}
		for i, n := range p.Shape {
			if n != spec.Shape[i] {
				return fmt.Errorf("tensor %q shape mismatch", p.Name)
			}
		}
		for _, v := range p.Values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("tensor %q contains non-finite value", p.Name)
			}
		}
	}
	return nil
}

func (c Checkpoint) Tiny() (*TinyScorer, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if c.Encoder != "tiny" {
		return nil, fmt.Errorf("checkpoint encoder %q is not tiny", c.Encoder)
	}
	m, err := NewTinyScorer(c.Config)
	if err != nil {
		return nil, err
	}
	params := make(map[string][]float32, len(c.Parameters))
	for _, p := range c.Parameters {
		params[p.Name] = p.Values
	}
	if err = m.LoadNamedParameters(params); err != nil {
		return nil, err
	}
	return m, nil
}

func ReadCheckpoint(r io.Reader) (Checkpoint, error) {
	var c Checkpoint
	d := json.NewDecoder(r)
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return c, err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return c, fmt.Errorf("checkpoint contains trailing data")
	}
	return c, c.Validate()
}
func LoadCheckpoint(path string) (Checkpoint, error) {
	f, err := os.Open(path)
	if err != nil {
		return Checkpoint{}, err
	}
	defer f.Close()
	return ReadCheckpoint(f)
}

// SaveCheckpoint writes atomically so an interrupted save cannot corrupt the
// previous best-validation checkpoint.
func SaveCheckpoint(path string, c Checkpoint) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".jevlike-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	e := json.NewEncoder(f)
	if err = e.Encode(c); err != nil {
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
	return os.Rename(f.Name(), path)
}
