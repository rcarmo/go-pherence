//go:build linux

package pockettts

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

type sharedAttentionError struct {
	Values, Changed, SignChanges int
	MaxAbs, Square               float64
}

func (e *sharedAttentionError) add(cpu, observed float32) {
	d := math.Abs(float64(cpu) - float64(observed))
	e.Values++
	if d != 0 {
		e.Changed++
	}
	if (cpu < 0 && observed > 0) || (cpu > 0 && observed < 0) {
		e.SignChanges++
	}
	e.MaxAbs = max(e.MaxAbs, d)
	e.Square += d * d
}

type sharedAttentionLayerObservation struct {
	Layer                                                      int
	Score, Probability, Output, DQ, DK, DV                     sharedAttentionError
	ReferenceDQ, ReferenceDK, ReferenceDV                      sharedAttentionError
	QKVProjection, OutProjection, FC1Projection, FC2Projection sharedAttentionError
}

type sharedAttentionDiagnosticBackend struct {
	observations []sharedAttentionLayerObservation
}

// Independent output-owned serial F32 reductions match shader indexing and
// evaluation order. Host exp cannot establish device transcendental behaviour.
func sharedAttentionForwardDiagnostic(q, k, v []float32, rows, heads, dim, contextRows int) ([]float32, []float32, []float32) {
	width := heads * dim
	scale := float32(1 / math.Sqrt(float64(dim)))
	scores, p, out := make([]float32, heads*rows*rows), make([]float32, heads*rows*rows), make([]float32, rows*width)
	for head := 0; head < heads; head++ {
		for query := 0; query < rows; query++ {
			start := 0
			if contextRows > 0 {
				start = max(0, query-contextRows+1)
			}
			base := (head*rows + query) * rows
			maximum := float32(-math.MaxFloat32)
			for key := start; key <= query; key++ {
				var sum float32
				for d := 0; d < dim; d++ {
					sum += float32(q[query*width+head*dim+d] * k[key*width+head*dim+d])
				}
				scores[base+key] = sum * scale
				maximum = max(maximum, scores[base+key])
			}
			var sum float32
			for key := start; key <= query; key++ {
				p[base+key] = float32(math.Exp(float64(scores[base+key] - maximum)))
				sum += p[base+key]
			}
			for key := start; key <= query; key++ {
				p[base+key] /= sum
			}
			for d := 0; d < dim; d++ {
				var sum float32
				for key := start; key <= query; key++ {
					sum += float32(p[base+key] * v[key*width+head*dim+d])
				}
				out[query*width+head*dim+d] = sum
			}
		}
	}
	return scores, p, out
}

func sharedAttentionBackwardDiagnostic(q, k, v, p, dy []float32, rows, heads, dim, contextRows int) ([]float32, []float32, []float32, []float32, []float32, []float32) {
	width := heads * dim
	scale := float32(1 / math.Sqrt(float64(dim)))
	dp, ds := make([]float32, heads*rows*rows), make([]float32, heads*rows*rows)
	allowed := func(query, key int) bool { return key <= query && (contextRows == 0 || query-key < contextRows) }
	for head := 0; head < heads; head++ {
		for query := 0; query < rows; query++ {
			base := (head*rows + query) * rows
			for key := 0; key <= query; key++ {
				if !allowed(query, key) {
					continue
				}
				var sum float32
				for d := 0; d < dim; d++ {
					sum += float32(dy[query*width+head*dim+d] * v[key*width+head*dim+d])
				}
				dp[base+key] = sum
			}
			var sum float32
			for key := 0; key <= query; key++ {
				if allowed(query, key) {
					sum += float32(p[base+key] * dp[base+key])
				}
			}
			for key := 0; key <= query; key++ {
				if allowed(query, key) {
					ds[base+key] = (p[base+key] * (dp[base+key] - sum)) * scale
				}
			}
		}
	}
	dq, dk, dv := make([]float32, rows*width), make([]float32, rows*width), make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		for head := 0; head < heads; head++ {
			for d := 0; d < dim; d++ {
				i := row*width + head*dim + d
				for key := 0; key <= row; key++ {
					if allowed(row, key) {
						dq[i] += float32(ds[(head*rows+row)*rows+key] * k[key*width+head*dim+d])
					}
				}
				for query := row; query < rows; query++ {
					if allowed(query, row) {
						index := (head*rows+query)*rows + row
						dk[i] += float32(ds[index] * q[query*width+head*dim+d])
						dv[i] += float32(p[index] * dy[query*width+head*dim+d])
					}
				}
			}
		}
	}
	// Reconstruct unchanged query/scatter CPU ordering separately. These values
	// alone return into AD; diagnostic observations never replace them.
	cq, ck, cv := make([]float32, rows*width), make([]float32, rows*width), make([]float32, rows*width)
	for query := 0; query < rows; query++ {
		for head := 0; head < heads; head++ {
			base := (head*rows + query) * rows
			var weighted float32
			for key := 0; key <= query; key++ {
				if allowed(query, key) {
					weighted += dp[base+key] * p[base+key]
					for d := 0; d < dim; d++ {
						cv[key*width+head*dim+d] += p[base+key] * dy[query*width+head*dim+d]
					}
				}
			}
			for key := 0; key <= query; key++ {
				if allowed(query, key) {
					score := p[base+key] * (dp[base+key] - weighted) * scale
					for d := 0; d < dim; d++ {
						cq[query*width+head*dim+d] += score * k[key*width+head*dim+d]
						ck[key*width+head*dim+d] += score * q[query*width+head*dim+d]
					}
				}
			}
		}
	}
	return dq, dk, dv, cq, ck, cv
}

// Limit expensive independent dense observations to first/final rows. This
// cannot localise whole-graph propagation; it measures input-fixed reductions.
func sharedProjectionObservation(linear LinearF32, input, cpu []float32, rows int) sharedAttentionError {
	var e sharedAttentionError
	for _, row := range []int{0, rows - 1} {
		for out := 0; out < linear.Out; out++ {
			var sum float32
			for in := 0; in < linear.In; in++ {
				sum += float32(input[row*linear.In+in] * linear.Weight[out*linear.In+in])
			}
			e.add(cpu[row*linear.Out+out], sum)
		}
	}
	return e
}

func (b *sharedAttentionDiagnosticBackend) EvaluateTransformer(m *TransformerCPU, sequence, seed []float32, rows int) (TransformerTrainingResult, error) {
	if b == nil {
		return TransformerTrainingResult{}, fmt.Errorf("nil attention diagnostic")
	}
	if err := validateTrainableTransformer(m, sequence, seed, rows); err != nil {
		return TransformerTrainingResult{}, err
	}
	tape, output := m.forwardTraining(rows, sequence)
	observations := make([]sharedAttentionLayerObservation, len(m.Layers))
	for i, l := range m.Layers {
		lt := tape.layers[i]
		o := &observations[i]
		o.Layer = i
		score, prob, att := sharedAttentionForwardDiagnostic(lt.q, lt.k, lt.v, rows, m.Heads, m.HeadDim, m.Context)
		for j := range prob {
			o.Probability.add(lt.probabilities[j], prob[j])
		}
		for j := range att {
			o.Output.add(lt.attention[j], att[j])
		}
		// CPU score computation at fixed recorded Q/K, before softmax.
		for head := 0; head < m.Heads; head++ {
			for query := 0; query < rows; query++ {
				start := 0
				if m.Context > 0 {
					start = max(0, query-m.Context+1)
				}
				for key := start; key <= query; key++ {
					s := dotScalar(lt.q[query*m.Width+head*m.HeadDim:query*m.Width+(head+1)*m.HeadDim], lt.k[key*m.Width+head*m.HeadDim:key*m.Width+(head+1)*m.HeadDim]) * float32(1/math.Sqrt(float64(m.HeadDim)))
					o.Score.add(s, score[(head*rows+query)*rows+key])
				}
			}
		}
		qkv := linearForwardRowsTraining(l.InProjection, lt.norm1.normalized, rows)
		o.QKVProjection = sharedProjectionObservation(l.InProjection, lt.norm1.normalized, qkv, rows)
		o.OutProjection = sharedProjectionObservation(l.OutProjection, lt.attention, lt.projected, rows)
		o.FC1Projection = sharedProjectionObservation(l.FC1, lt.norm2.normalized, lt.fc1Pre, rows)
		o.FC2Projection = sharedProjectionObservation(l.FC2, lt.fc1, lt.down, rows)
	}
	layer := len(m.Layers) - 1
	backend := attentionBackendFunc(func(q, k, v, p, dy []float32, s, h, d, c int) ([]float32, []float32, []float32, error) {
		sq, sk, sv, cq, ck, cv := sharedAttentionBackwardDiagnostic(q, k, v, p, dy, s, h, d, c)
		rq, rk, rv, err := referenceAttentionBackend(q, k, v, p, dy, s, h, d, c)
		if err != nil {
			return nil, nil, nil, err
		}
		o := &observations[layer]
		for i := range cq {
			o.DQ.add(cq[i], sq[i])
			o.DK.add(ck[i], sk[i])
			o.DV.add(cv[i], sv[i])
			o.ReferenceDQ.add(cq[i], rq[i])
			o.ReferenceDK.add(ck[i], rk[i])
			o.ReferenceDV.add(cv[i], rv[i])
		}
		layer--
		return cq, ck, cv, nil
	})
	g := newTransformerGradients(m)
	dx, err := m.backwardTrainingWithAttention(rows, tape, seed, g, backend)
	if err != nil {
		return TransformerTrainingResult{}, err
	}
	if !zeroTrainingSeed(seed) {
		b.observations = observations
	}
	all := map[string][]float32{}
	addFlowLMGradients(all, &FlowLMTrainingGradients{Transformer: g})
	for name := range all {
		if !strings.HasPrefix(name, "flow_lm.transformer.layers.") && !strings.HasPrefix(name, "flow_lm.out_norm.") {
			delete(all, name)
		}
	}
	return TransformerTrainingResult{Output: output, DSequence: dx, Parameters: all}, nil
}

func TestPocketSharedAttentionDiagnosticBoundary(t *testing.T) {
	lm := tinyFlowLMTraining()
	rows := 4
	sequence, seed := make([]float32, rows*lm.Hidden), make([]float32, rows*lm.Hidden)
	for i := range sequence {
		sequence[i] = float32(i%7-3) * .1
		seed[i] = float32(i%3-1) * .03
	}
	for _, contextRows := range []int{0, 1, 2, 8} {
		lm.Transformer.Context = contextRows
		want, g, dx, err := lm.Transformer.ForwardBackward(sequence, seed, rows)
		if err != nil {
			t.Fatal(err)
		}
		b := &sharedAttentionDiagnosticBackend{}
		r, err := b.EvaluateTransformer(lm.Transformer, sequence, seed, rows)
		if err != nil {
			t.Fatal(err)
		}
		if err := validateTransformerBackendResult(lm.Transformer, r, sequence, seed, rows); err != nil {
			t.Fatal(err)
		}
		assertSliceClose(t, "diagnostic forward", r.Output, want, 3e-5)
		assertSliceClose(t, "diagnostic dx", r.DSequence, dx, 3e-5)
		grads := map[string][]float32{}
		addFlowLMGradients(grads, &FlowLMTrainingGradients{Transformer: g})
		for name, got := range r.Parameters {
			assertSliceClose(t, name, got, grads[name], 3e-5)
		}
		if len(b.observations) != len(lm.Transformer.Layers) {
			t.Fatal("observation layers")
		}
		for _, o := range b.observations {
			if o.Probability.MaxAbs > 3e-5 || o.Output.MaxAbs > 3e-5 || o.DQ.MaxAbs > 3e-5 || o.DK.MaxAbs > 3e-5 || o.DV.MaxAbs > 3e-5 {
				t.Fatal("tiny serial attention gate")
			}
			if o.ReferenceDQ.MaxAbs > 3e-5 || o.ReferenceDK.MaxAbs > 3e-5 || o.ReferenceDV.MaxAbs > 3e-5 {
				t.Fatal("independent F64 attention adjoint gate")
			}
		}
	}
	if _, err := (*sharedAttentionDiagnosticBackend)(nil).EvaluateTransformer(nil, nil, nil, 0); err == nil {
		t.Fatal("nil observer")
	}
	if _, err := (&sharedAttentionDiagnosticBackend{}).EvaluateTransformer(nil, nil, nil, 0); err == nil {
		t.Fatal("invalid geometry")
	}
}
