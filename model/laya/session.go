package laya

import (
	"fmt"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/model/modernbert"
	"math"
)

type Session struct {
	m                                                                       *Model
	encoder                                                                 *modernbert.Session
	maxSeq, maxOptions                                                      int
	x, norm, qkv, att, proj, ff, scores, scoreNorm, scoreMid, features, act []float32
}

func (m *Model) NewSession(maxSeq, maxOptions int) (*Session, error) {
	if m == nil || maxSeq < 1 || maxOptions < 2 || maxOptions > maxSeq {
		return nil, fmt.Errorf("laya: invalid session capacity")
	}
	enc, e := m.Encoder.NewSession(maxSeq)
	if e != nil {
		return nil, e
	}
	h := m.hidden
	return &Session{m: m, encoder: enc, maxSeq: maxSeq, maxOptions: maxOptions, x: make([]float32, maxSeq*h), norm: make([]float32, maxSeq*h), qkv: make([]float32, maxSeq*3*h), att: make([]float32, maxSeq*h), proj: make([]float32, maxSeq*h), ff: make([]float32, maxSeq*4*h), scores: make([]float32, maxSeq*maxSeq), scoreNorm: make([]float32, h), scoreMid: make([]float32, h), features: make([]float32, h+4), act: make([]float32, 256)}, nil
}

// ForwardInto writes logits, action logits and probabilities without warm allocations.
func (s *Session) ForwardInto(logits, actions, probs []float32, ids []int, mask []bool, markers []int, qtype QuestionType) (float32, error) {
	if s == nil || s.m == nil || qtype < Choice || qtype > Noul || len(ids) < 1 || len(ids) > s.maxSeq || len(mask) != len(ids) || len(markers) < 2 || len(markers) > s.maxOptions || len(logits) != len(markers) || len(probs) != len(markers) || len(actions) != s.m.actions {
		return 0, fmt.Errorf("laya: invalid input/output")
	}
	m := s.m
	h, n := m.hidden, len(ids)
	x := s.x[:n*h]
	if e := s.encoder.ForwardInto(x, ids, mask); e != nil {
		return 0, e
	}
	for r := 0; r < n; r++ {
		for j := 0; j < h; j++ {
			x[r*h+j] += m.typeEmb[int(qtype)*h+j]
		}
	}
	for _, w := range m.layers {
		norm := s.norm[:n*h]
		ln(norm, x, w.n1w, w.n1b, n, h)
		qkv := s.qkv[:n*3*h]
		dense(qkv, norm, w.qkv, w.qkvb, n, 3*h, h)
		att := s.att[:n*h]
		clear(att)
		heads := max(1, h/64)
		hd := h / heads
		scores := s.scores[:n*n]
		for head := 0; head < heads; head++ {
			for i := 0; i < n; i++ {
				row := scores[i*n : (i+1)*n]
				for j := 0; j < n; j++ {
					if !mask[j] {
						row[j] = float32(math.Inf(-1))
						continue
					}
					var z float32
					for d := 0; d < hd; d++ {
						z += qkv[i*3*h+head*hd+d] * qkv[j*3*h+h+head*hd+d]
					}
					row[j] = z / float32(math.Sqrt(float64(hd)))
				}
				simd.SoftmaxRowsInPlace(row, 1, n)
				for d := 0; d < hd; d++ {
					var z float32
					for j, p := range row {
						z += p * qkv[j*3*h+2*h+head*hd+d]
					}
					att[i*h+head*hd+d] = z
				}
			}
		}
		proj := s.proj[:n*h]
		dense(proj, att, w.outw, w.outb, n, h, h)
		for i := range x {
			x[i] += proj[i]
		}
		ln(norm, x, w.n2w, w.n2b, n, h)
		ff := s.ff[:n*4*h]
		dense(ff, norm, w.ff1, w.ff1b, n, 4*h, h)
		for i, v := range ff {
			if v < 0 {
				ff[i] = 0
			}
		}
		dense(proj, ff, w.ff2, w.ff2b, n, h, 4*h)
		for i := range x {
			x[i] += proj[i]
		}
	}
	for i, pos := range markers {
		if pos < 0 || pos >= n || !mask[pos] {
			return 0, fmt.Errorf("laya: invalid marker")
		}
		row := x[pos*h : (pos+1)*h]
		ln(s.scoreNorm, row, m.scoreNormW, m.scoreNormB, 1, h)
		dense(s.scoreMid, s.scoreNorm, m.scoreW, m.scoreB, 1, h, h)
		simd.GELUErfF32To(s.scoreMid, s.scoreMid)
		dense(logits[i:i+1], s.scoreMid, m.scoreOut, m.scoreOutB, 1, 1, h)
	}
	copy(probs, logits)
	simd.SoftmaxRowsInPlace(probs, 1, len(probs))
	top1, top2 := float32(0), float32(0)
	for _, p := range probs {
		if p > top1 {
			top2, top1 = top1, p
		} else if p > top2 {
			top2 = p
		}
	}
	var ent float64
	for _, p := range probs {
		if p > 0 {
			ent -= float64(p) * math.Log(float64(p))
		}
	}
	entropy := float32(ent / math.Log(float64(len(probs))))
	features := s.features
	copy(features, x[:h])
	features[h], features[h+1], features[h+2], features[h+3] = top1, top1-top2, entropy, float32(len(probs))/255
	dense(s.act, features, m.actW, m.actB, 1, 256, h+4)
	simd.GELUErfF32To(s.act, s.act)
	dense(actions, s.act, m.actOut, m.actOutB, 1, m.actions, 256)
	temp := m.temp[qtype]
	if temp < 1e-3 {
		temp = 1e-3
	}
	copy(probs, logits)
	for i := range probs {
		probs[i] /= temp
	}
	simd.SoftmaxRowsInPlace(probs, 1, len(probs))
	var e2 float64
	for _, p := range probs {
		if p > 0 {
			e2 -= float64(p) * math.Log(float64(p))
		}
	}
	return float32(1 - e2/math.Log(float64(len(probs)))), nil
}
