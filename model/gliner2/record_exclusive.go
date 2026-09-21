package gliner2

import "math"

// Selections keyed by [instance,field], then candidate index. Natural anchor
// fields are forced separately by the decoder. Exact assignment ties use the
// Go solver's deterministic lexicographic policy, not scipy's unspecified tie.
func exclusiveRecordSelections(g DenseRecordGroupOutput, order []int, temperature, threshold float64) (map[[2]int]map[int]float64, error) {
	out := map[[2]int]map[int]float64{}
	if len(order) == 0 {
		return out, nil
	}
	for f, field := range g.Fields {
		if !field.Exclusive {
			continue
		}
		candidates := []int{}
		for c, valid := range g.FieldMembership[f] {
			if valid {
				candidates = append(candidates, c)
			}
		}
		if !field.Scalar {
			for _, c := range candidates {
				best, prob := -1, float64(-1)
				for _, inst := range order {
					v := sigmoid(float64(g.AssignLogits[inst][f][c+1]) / temperature)
					if v > prob {
						best, prob = inst, v
					}
				}
				if prob >= threshold {
					key := [2]int{best, f}
					if out[key] == nil {
						out[key] = map[int]float64{}
					}
					out[key][c] = prob
				}
			}
			continue
		}
		probs := make([][]float64, len(order))
		cost := make([][]float64, len(order))
		maxCost := float64(0)
		for i, inst := range order {
			row := g.AssignLogits[inst][f]
			mx := float64(row[0]) / temperature
			for _, c := range candidates {
				mx = math.Max(mx, float64(row[c+1])/temperature)
			}
			p := make([]float64, 1+len(candidates))
			p[0] = math.Exp(float64(row[0])/temperature - mx)
			sum := p[0]
			for j, c := range candidates {
				p[j+1] = math.Exp(float64(row[c+1])/temperature - mx)
				sum += p[j+1]
			}
			for j := range p {
				p[j] /= sum
			}
			probs[i] = p
			cost[i] = make([]float64, len(candidates)+len(order))
			for j := range candidates {
				v := -math.Log(math.Max(p[j+1], 1e-8))
				cost[i][j] = v
				maxCost = math.Max(maxCost, v)
			}
		}
		for i := range order {
			absent := -math.Log(math.Max(probs[i][0], 1e-8))
			if field.Required {
				absent = maxCost + 50
			}
			for j := range order {
				cost[i][len(candidates)+j] = absent + 1000
			}
			cost[i][len(candidates)+i] = absent
		}
		assignment, err := minimumCostAssignment(cost)
		if err != nil {
			return nil, err
		}
		for i, col := range assignment {
			if col >= len(candidates) {
				continue
			}
			prob := probs[i][col+1]
			if !field.Required && prob < threshold {
				continue
			}
			out[[2]int{order[i], f}] = map[int]float64{candidates[col]: prob}
		}
	}
	return out, nil
}
