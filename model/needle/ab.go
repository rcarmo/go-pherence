package needle

import (
	"fmt"
	"strings"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

// AB scales broadcast against the reduction-axis-last view, not the on-disk
// matrix orientation. Preserve that mapping for both values and STE gradients.
func abShape(name string, shape []int) []int {
	s := append([]int(nil), shape...)
	if cqSecondLast(name) {
		s[len(s)-1], s[len(s)-2] = s[len(s)-2], s[len(s)-1]
	}
	return s
}
func validateAB(tensors map[string]checkpoint.Tensor) error {
	for key, scale := range tensors {
		if !strings.HasPrefix(key, "ab_scales/") {
			continue
		}
		rest := strings.TrimPrefix(key, "ab_scales/")
		if !strings.HasSuffix(rest, "/a") && !strings.HasSuffix(rest, "/b") {
			return fmt.Errorf("needle: invalid AB scale %s", key)
		}
		name := rest[:len(rest)-2]
		w, ok := tensors[name]
		if !ok || !isCQ(name) || len(w.Shape) < 2 {
			return fmt.Errorf("needle: unknown AB target %s", name)
		}
		shape := abShape(name, w.Shape)
		if len(scale.Shape) > len(shape) {
			return fmt.Errorf("needle: AB broadcast rank for %s", key)
		}
		offset := len(shape) - len(scale.Shape)
		for i, d := range scale.Shape {
			if d != 1 && d != shape[offset+i] {
				return fmt.Errorf("needle: AB broadcast shape for %s", key)
			}
		}
		other := "a"
		if strings.HasSuffix(rest, "/a") {
			other = "b"
		}
		if _, ok := tensors["ab_scales/"+name+"/"+other]; !ok {
			return fmt.Errorf("needle: AB scales require both a and b for %s", name)
		}
	}
	return nil
}
func (e *execution) abScale(name, which string) *value {
	key := "ab_scales/" + name + "/" + which
	s := e.param(key, -1)
	w := e.m.tensors[name]
	shape := abShape(name, w.Shape)
	scaleShape := e.m.tensors[key].Shape
	idx := e.t.ints(len(w.Data))
	rank := len(shape)
	coords := make([]int, rank)
	original := w.Shape
	for pos := range idx {
		v := pos
		for d := rank - 1; d >= 0; d-- {
			coords[d] = v % original[d]
			v /= original[d]
		}
		if cqSecondLast(name) {
			coords[rank-1], coords[rank-2] = coords[rank-2], coords[rank-1]
		}
		j := 0
		for d, size := range scaleShape {
			coord := coords[rank-len(scaleShape)+d]
			if size == 1 {
				coord = 0
			}
			j = j*size + coord
		}
		idx[pos] = j
	}
	return e.t.gather(s, 1, len(idx), idx)
}
