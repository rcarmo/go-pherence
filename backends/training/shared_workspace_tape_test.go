package training

import (
	"context"
	"errors"
	"math"
	"testing"
)

type sharedHostTape struct {
	driver  *sharedHostDriver
	tape    Tape
	bound   map[int]ResidentBuffer
	outputs map[int]*residentHostBuffer
	closed  bool
}

func (d *sharedHostDriver) ValidateAccumulate(ctx context.Context, o, a, b ResidentBuffer, n int, scale float32) error {
	return d.ValidateSharedAccumulate(ctx, o, a, b, n, scale)
}
func (d *sharedHostDriver) NewBoundTape(ctx context.Context, t Tape, bound map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	if err := t.Validate(maxBytes); err != nil {
		return nil, err
	}
	for id, b := range bound {
		n, _ := t.Tensors[id].Elements()
		x, e := d.raw(ctx, b, false)
		if e != nil || len(x.values) != n {
			return nil, errors.New("tape binding")
		}
	}
	return &sharedHostTape{driver: d, tape: t, bound: bound}, d.event("tape-new")
}
func (t *sharedHostTape) RunDevice(ctx context.Context) error {
	d := t.driver
	if err := d.event("tape-run"); err != nil {
		return err
	}
	values := make([][]float32, len(t.tape.Tensors))
	for i, s := range t.tape.Tensors {
		values[i] = append([]float32(nil), s.Values...)
	}
	for id, b := range t.bound {
		x, e := d.raw(ctx, b, false)
		if e != nil {
			return e
		}
		values[id] = x.values
	}
	for _, c := range t.tape.Commands {
		n, _ := t.tape.Tensors[c.Output].Elements()
		out := make([]float32, n)
		if c.Op == "vector" {
			for i := range out {
				out[i] = values[c.Inputs[0]][i] + values[c.Inputs[1]][i]*c.Scale
			}
		} else if c.Op == "embedding" && c.Mode == 3 {
			index := values[c.Inputs[1]]
			for slot := 0; slot < c.Rows; slot++ {
				start, end := int(index[c.Rows+slot]), int(index[c.Rows+slot+1])
				for ch := 0; ch < c.Width; ch++ {
					for q := start; q < end; q++ {
						row := int(index[2*c.Rows+1+q])
						out[slot*c.Width+ch] += values[c.Inputs[0]][row*c.Width+ch]
					}
				}
			}
		} else {
			return errors.New("shared mock opcode")
		}
		values[c.Output] = out
	}
	t.outputs = map[int]*residentHostBuffer{}
	for _, id := range t.tape.Outputs {
		t.outputs[id] = &residentHostBuffer{values: values[id], driver: &d.residentHostDriver}
	}
	return ctx.Err()
}
func (t *sharedHostTape) OutputBuffer(id int) (ResidentBuffer, error) {
	if t.closed {
		return nil, errors.New("closed")
	}
	if err := t.driver.event("tape-output"); err != nil {
		return nil, err
	}
	b := t.outputs[id]
	if b == nil {
		return nil, errors.New("output")
	}
	return b, nil
}
func (t *sharedHostTape) DownloadOutputs(ctx context.Context, ids []int) ([][]float32, error) {
	if err := t.driver.event("tape-download"); err != nil {
		return nil, err
	}
	out := make([][]float32, len(ids))
	for i, id := range ids {
		b, e := t.OutputBuffer(id)
		if e != nil {
			return nil, e
		}
		out[i] = append([]float32(nil), b.(*residentHostBuffer).values...)
	}
	return out, ctx.Err()
}
func (t *sharedHostTape) Close() error {
	if err := t.driver.event("tape-close"); err != nil {
		return err
	}
	t.closed = true
	return nil
}
func (d *sharedHostDriver) ValidateCompactEmbeddingAccumulate(ctx context.Context, o, a, src, index ResidentBuffer, rows, width, vocab int, scale float32) error {
	dense, compact, ix, err := CompactEmbeddingAccumulateShape(rows, width, vocab, scale)
	if err != nil {
		return err
	}
	x, e := d.raw(ctx, o, true)
	if e != nil {
		return e
	}
	p, e := d.raw(ctx, a, false)
	if e != nil {
		return e
	}
	v, e := d.raw(ctx, src, false)
	if e != nil {
		return e
	}
	i, e := d.raw(ctx, index, false)
	if e != nil {
		return e
	}
	if len(x.values) != dense || len(p.values) != dense || len(v.values) != compact || len(i.values) != ix || sharedHostOverlap(x, p) || sharedHostOverlap(x, v) || sharedHostOverlap(x, i) || sharedHostOverlap(p, v) || sharedHostOverlap(p, i) || sharedHostOverlap(v, i) {
		return errors.New("compact ranges")
	}
	if err := ValidateCompactEmbeddingIndex(i.values, rows, vocab); err != nil {
		return err
	}
	return d.event("compact-admit")
}
func (d *sharedHostDriver) CompactEmbeddingAccumulate(ctx context.Context, o, a, src, index ResidentBuffer, rows, width, vocab int, scale float32) error {
	if err := d.ValidateCompactEmbeddingAccumulate(ctx, o, a, src, index, rows, width, vocab, scale); err != nil {
		return err
	}
	if err := d.event("compact"); err != nil {
		return err
	}
	x, _ := d.raw(ctx, o, true)
	p, _ := d.raw(ctx, a, false)
	v, _ := d.raw(ctx, src, false)
	ix, _ := d.raw(ctx, index, false)
	for token := 0; token < vocab; token++ {
		slot := -1
		for j := 0; j < rows; j++ {
			if int(ix.values[j]) == token {
				slot = j
				break
			}
		}
		for ch := 0; ch < width; ch++ {
			z := float32(0)
			if slot >= 0 {
				z = scale * v.values[slot*width+ch]
			}
			pos := token*width + ch
			x.values[pos] = p.values[pos] + z
		}
	}
	return ctx.Err()
}
func TestTrainingSharedWorkspaceDenseCompactTapes(t *testing.T) {
	ctx := context.Background()
	t.Run("dense", func(t *testing.T) {
		s, d := requireShared(t, residentFixture())
		tape, b := residentBindingFixture()
		stable := s.entries[0].gradient
		for _, scale := range []float32{0, .25, 1} {
			values, e := s.RunPrunedTape(ctx, tape, b, []int{2}, true, scale, 1024)
			if e != nil || len(values) != 1 {
				t.Fatal(e)
			}
		}
		if s.entries[0].gradient != stable || d.events["copy"] != 3 {
			t.Fatal("dense stable copy")
		}
		want := []float64{1.25 * (.3 + 1), 1.25 * (-.7 + 2)}
		for i, v := range stable.(*residentHostBuffer).values {
			if math.Abs(float64(v)-want[i]) > 2e-6 {
				t.Fatal("denseF64", v, want[i])
			}
		}
	})
	t.Run("compact", func(t *testing.T) {
		s, d := requireShared(t, sharedCompactFixture())
		tape, bindings, compact := compactBindingFixture()
		identities := []ResidentBuffer{s.entries[0].gradient, s.entries[1].gradient}
		for _, scale := range []float32{0, .25, 1} {
			if _, err := s.RunCompactZeroPrunedTape(ctx, tape, bindings, compact, []int{6}, true, scale, 4096); err != nil {
				t.Fatal(err)
			}
		}
		if s.entries[0].gradient != identities[0] || s.entries[1].gradient != identities[1] || d.events["compact"] != 3 || d.events["copy"] != 6 {
			t.Fatal("compact count")
		}
		expected := [][]float32{{3.75, 5, 0, 0, 7.5, 10, 0, 0}, {3.75, 6.25}}
		for j, e := range s.entries {
			for i, v := range e.gradient.(*residentHostBuffer).values {
				if math.Abs(float64(v)-float64(expected[j][i])) > 2e-6 {
					t.Fatal("compact numerical", j, i, v, expected[j][i])
				}
			}
		}
	})
}
func TestTrainingSharedWorkspaceTapeFailures(t *testing.T) {
	for _, event := range []string{"tape-new", "tape-run", "tape-output", "shared-admit", "copy-admit", "compact-admit", "accumulate", "compact", "copy", "tape-download", "tape-close"} {
		t.Run(event, func(t *testing.T) {
			ctx := context.Background()
			fixture := residentFixture()
			if event == "compact" || event == "compact-admit" {
				fixture = sharedCompactFixture()
			}
			s, d := requireShared(t, fixture)
			d.fail = event
			d.at = 1
			var err error
			if event == "compact" || event == "compact-admit" {
				tape, b, c := compactBindingFixture()
				_, err = s.RunCompactZeroPrunedTape(ctx, tape, b, c, []int{6}, true, .25, 4096)
			} else {
				tape, b := residentBindingFixture()
				_, err = s.RunPrunedTape(ctx, tape, b, []int{2}, true, .25, 1024)
			}
			if err == nil {
				t.Fatal("fault accepted")
			}
			// Source has executed for validation failures, so retain whole consumer and
			// state even when no first persistent copy occurred.
			if s.terminal == nil || len(s.retained) == 0 || s.Close() == nil {
				t.Fatal("tape uncertain retention", event)
			}
		})
	}
}
func TestTrainingSharedWorkspaceCopyTransportAndReentry(t *testing.T) {
	ctx := context.Background()
	s, d := requireShared(t, residentFixture())
	w := s.shared
	if err := w.begin(ctx); err != nil {
		t.Fatal(err)
	}
	opctx := w.ctx
	src, err := w.view(w.scratch, 2, true)
	if err != nil {
		t.Fatal(err)
	}
	vals := []float32{float32(math.Copysign(0, -1)), math.SmallestNonzeroFloat32}
	if err = src.Upload(opctx, vals); err != nil {
		t.Fatal(err)
	}
	dst := s.entries[0].gradient
	if err = d.Copy(opctx, dst, src, 2); err != nil {
		t.Fatal(err)
	}
	// Exact bits are the raw copy transport contract, not an arithmetic gate.
	for i, v := range dst.(*residentHostBuffer).values {
		if math.Float32bits(v) != math.Float32bits(vals[i]) {
			t.Fatal("copy transport bits")
		}
	}
	w.end()
	d.hook = func(event string) {
		if s.Accumulate(ctx, map[string][]float32{"b": {1}}, 1) == nil || s.Close() == nil {
			t.Fatal("reentry accepted")
		}
	}
	if err = s.Accumulate(ctx, map[string][]float32{"a": {1, 2}, "b": {3}}, .25); err != nil {
		t.Fatal(err)
	}
}
func TestTrainingSharedWorkspaceCopyCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	s, d := requireShared(t, residentFixture())
	d.hook = func(event string) {
		if event == "copy" {
			cancel()
		}
	}
	if err := s.Accumulate(ctx, map[string][]float32{"a": {1, 2}, "b": {3}}, 1); err == nil || s.terminal == nil || s.entries[0].ready || s.Close() == nil {
		t.Fatal("copy completion cancellation")
	}
}

func (d *sharedHostDriver) NewReuseBoundTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	return d.NewBoundTape(ctx, t, b, maxBytes)
}
func (d *sharedHostDriver) NewZeroBoundTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	return d.NewBoundTape(ctx, t, b, maxBytes)
}
func sharedCompactFixture() ResidentSnapshot {
	f := residentFixture()
	f.Parameters = map[string]ResidentParameter{"a": {Parameter: []float32{1, 2, 3, 4, 5, 6, 7, 8}, M: make([]float32, 8), V: make([]float32, 8), EMA: make([]float32, 8)}, "b": {Parameter: []float32{2, 3}, M: make([]float32, 2), V: make([]float32, 2), EMA: make([]float32, 2)}}
	return f
}
