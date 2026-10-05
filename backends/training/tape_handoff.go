package training

import (
	"context"
	"fmt"
)

// ResidentHandoffTapeDriver preflights a complete graph and bound read-only
// buffers without allocating or submitting work. The normal constructor must
// repeat admission when future producer outputs are finally bound. Returned
// tapes must complete and synchronise device work before RunDevice succeeds;
// drivers that only enqueue work cannot implement this interface safely.
type ResidentHandoffTapeDriver interface {
	ResidentTapeDriver
	ValidateHandoffTape(context.Context, Tape, map[int]ResidentBuffer, int64) error
}

// TapeHandoffEdge connects a declared producer output to an explicit consumer
// constant placeholder of equal extent. No gradients implicitly cross this edge.
type TapeHandoffEdge struct{ Output, Input int }

// ResidentTapeHandoff owns both tapes. Producer storage stays live while the
// consumer borrows it, including caller accumulation from consumer outputs.
// All methods and borrowed-output uses require caller serialisation: concurrent
// Close versus RunDevice (or accumulation/download) is unsupported. External
// bindings remain caller-owned, live and immutable until successful Close;
// producer bindings are read at its run and consumer bindings at its later run.
// Returned buffers are read-only borrows valid only until successful Close.
// Finish and synchronise external work before Close, or call RetainFailure on
// error and retain this owner. A failure after producer execution retains both
// owners, blocks retries and refuses ordinary Close; no recovery is implied.
type ResidentTapeHandoff struct {
	driver             ResidentHandoffTapeDriver
	producer, consumer ResidentBoundTape
	source, target     Tape
	edges              []TapeHandoffEdge
	targetBindings     map[int]ResidentBuffer
	maxBytes           int64
	ran, closed        bool
	terminal           error
}

// NewResidentTapeHandoff admits both complete graphs before constructing the
// producer. A non-nil owner returned with an error retains partial resources;
// callers must keep that owner for coordinated recovery, never blindly unwind
// uncertain native work. Kernel modules and external buffer owners must outlive it.
func NewResidentTapeHandoff(ctx context.Context, driver ResidentHandoffTapeDriver, source, target Tape, edges []TapeHandoffEdge, sourceBindings, targetBindings map[int]ResidentBuffer, maxBytes int64) (*ResidentTapeHandoff, error) {
	if ctx == nil || driver == nil {
		return nil, fmt.Errorf("nil handoff context/driver")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if maxBytes <= 0 {
		return nil, fmt.Errorf("handoff byte ceiling required")
	}
	if err := source.Validate(maxBytes); err != nil {
		return nil, err
	}
	if err := target.Validate(maxBytes); err != nil {
		return nil, err
	}
	// Both tapes coexist. Charge complete logical extents, including borrowed
	// buffers/placeholders, to one combined bound before any allocation.
	var combined int64
	for _, tape := range []Tape{source, target} {
		for _, spec := range tape.Tensors {
			n, _ := spec.Elements()
			bytes := int64(n) * 4
			if bytes > maxBytes-combined {
				return nil, fmt.Errorf("handoff combined byte ceiling")
			}
			combined += bytes
		}
	}
	if len(edges) == 0 || len(edges) > 4096 {
		return nil, fmt.Errorf("handoff edge count")
	}
	declared := map[int]bool{}
	for _, id := range source.Outputs {
		declared[id] = true
	}
	used := map[int]bool{}
	producerUsed := map[int]bool{}
	producerFamily := make([]int, len(source.Tensors))
	for id := range producerFamily {
		producerFamily[id] = id
	}
	for _, command := range source.Commands {
		if command.Op == "reshape" {
			producerFamily[command.Output] = producerFamily[command.Inputs[0]]
		}
	}
	for _, e := range edges {
		if e.Output < 0 || e.Output >= len(source.Tensors) || e.Input < 0 || e.Input >= len(target.Tensors) || !declared[e.Output] || used[e.Input] || producerUsed[producerFamily[e.Output]] || target.Tensors[e.Input].Values == nil {
			return nil, fmt.Errorf("handoff edge ID/constant/duplicate")
		}
		if _, bound := targetBindings[e.Input]; bound {
			return nil, fmt.Errorf("handoff edge conflicts with binding")
		}
		a, _ := source.Tensors[e.Output].Elements()
		b, _ := target.Tensors[e.Input].Elements()
		if a != b {
			return nil, fmt.Errorf("handoff edge extent")
		}
		used[e.Input] = true
		producerUsed[producerFamily[e.Output]] = true
	}
	if err := ValidateExternalTapeBindings(source, sourceBindings); err != nil {
		return nil, err
	}
	if err := ValidateExternalTapeBindings(target, targetBindings); err != nil {
		return nil, err
	}
	if err := driver.ValidateHandoffTape(ctx, source, sourceBindings, maxBytes); err != nil {
		return nil, err
	}
	if err := driver.ValidateHandoffTape(ctx, target, targetBindings, maxBytes); err != nil {
		return nil, err
	}
	// Freeze graph metadata and constants before deferring consumer construction.
	// External bound buffer owners remain borrowed and must outlive this owner.
	p := &ResidentTapeHandoff{driver: driver, source: copyHandoffTape(source), target: copyHandoffTape(target), edges: append([]TapeHandoffEdge(nil), edges...), targetBindings: make(map[int]ResidentBuffer, len(targetBindings)+len(edges)), maxBytes: maxBytes}
	for id, b := range targetBindings {
		p.targetBindings[id] = b
	}
	var err error
	p.producer, err = driver.NewBoundTape(ctx, p.source, sourceBindings, maxBytes)
	if err != nil {
		if p.producer != nil {
			p.terminal = err
			return p, err
		}
		return nil, err
	}
	if p.producer == nil {
		p.terminal = fmt.Errorf("handoff nil producer")
		return p, p.terminal
	}
	return p, nil
}
func copyHandoffTape(t Tape) Tape {
	return copyTapeStorage(t, nil, true)
}

// Lease sources retain only frozen output/reshape/extent provenance after
// synchronous native construction. They are not executable deferred tapes.
func copyLeaseMetadata(t Tape) Tape { return copyTapeStorage(t, nil, false) }

func copyTapeStorage(t Tape, removed map[int]bool, values bool) Tape {
	t.Tensors = append([]TensorSpec(nil), t.Tensors...)
	for i, s := range t.Tensors {
		s.Shape = append([]int(nil), s.Shape...)
		if !values || removed[i] {
			s.Values = nil
		} else if s.Values != nil {
			s.Values = append([]float32(nil), s.Values...)
		}
		t.Tensors[i] = s
	}
	t.Commands = append([]Command(nil), t.Commands...)
	for i, c := range t.Commands {
		c.Inputs = append([]int(nil), c.Inputs...)
		t.Commands[i] = c
	}
	t.Outputs = append([]int(nil), t.Outputs...)
	return t
}

// RunDevice executes the pair once. Success includes both tapes' device
// completion; no host download occurs at the read-only handoff.
func (p *ResidentTapeHandoff) RunDevice(ctx context.Context) error {
	if p == nil || p.closed || p.ran || p.terminal != nil || ctx == nil {
		return fmt.Errorf("nil/closed/executed/terminal handoff")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	p.ran = true
	if err := p.producer.RunDevice(ctx); err != nil {
		p.terminal = err
		return err
	}
	for _, e := range p.edges {
		b, err := p.producer.OutputBuffer(e.Output)
		if err != nil {
			p.terminal = err
			return err
		}
		p.targetBindings[e.Input] = b
	}
	if err := p.driver.ValidateHandoffTape(ctx, p.target, p.targetBindings, p.maxBytes); err != nil {
		p.terminal = err
		return err
	}
	var err error
	p.consumer, err = p.driver.NewBoundTape(ctx, p.target, p.targetBindings, p.maxBytes)
	if err != nil {
		p.terminal = err
		return err
	}
	if p.consumer == nil {
		p.terminal = fmt.Errorf("handoff nil consumer")
		return p.terminal
	}
	if err = p.consumer.RunDevice(ctx); err != nil {
		p.terminal = err
	}
	return err
}
func (p *ResidentTapeHandoff) OutputBuffer(id int) (ResidentBuffer, error) {
	if p == nil || p.closed || !p.ran || p.consumer == nil || p.terminal != nil {
		return nil, fmt.Errorf("unavailable handoff output")
	}
	declared := false
	for _, output := range p.target.Outputs {
		if output == id {
			declared = true
		}
	}
	if !declared {
		return nil, fmt.Errorf("handoff output not declared")
	}
	b, err := p.consumer.OutputBuffer(id)
	if err != nil {
		p.terminal = err
	}
	return b, err
}
func (p *ResidentTapeHandoff) DownloadOutputs(ctx context.Context, ids []int) ([][]float32, error) {
	if p == nil || p.closed || !p.ran || p.consumer == nil || p.terminal != nil {
		return nil, fmt.Errorf("unavailable handoff download")
	}
	if ctx == nil {
		return nil, fmt.Errorf("nil handoff download context")
	}
	for _, id := range ids {
		declared := false
		for _, output := range p.target.Outputs {
			if output == id {
				declared = true
			}
		}
		if !declared {
			return nil, fmt.Errorf("handoff download not declared")
		}
	}
	values, err := p.consumer.DownloadOutputs(ctx, ids)
	if err != nil {
		p.terminal = err
	}
	return values, err
}

// RetainFailure records uncertain external work using a borrowed consumer
// output (for example accumulation). Call before cleanup on any such error;
// it prevents freeing either tape while that work may still reference them.
func (p *ResidentTapeHandoff) RetainFailure(err error) error {
	if err != nil && p != nil && p.terminal == nil {
		p.terminal = err
	}
	return err
}
func (p *ResidentTapeHandoff) Close() error {
	if p == nil || p.closed {
		return nil
	}
	if p.terminal != nil {
		return fmt.Errorf("handoff resources retained: %w", p.terminal)
	}
	// Never free a producer while its consumer might still reference it.
	if p.consumer != nil {
		if err := p.consumer.Close(); err != nil {
			p.terminal = err
			return err
		}
	}
	if p.producer != nil {
		if err := p.producer.Close(); err != nil {
			p.terminal = err
			return err
		}
	}
	p.closed = true
	return nil
}
