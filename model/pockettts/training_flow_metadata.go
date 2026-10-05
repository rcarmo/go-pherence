package pockettts

import train "github.com/rcarmo/go-pherence/backends/training"

// Count reverse commands without constructing another tape. Only adjoint
// presence is simulated; numerical execution and IDs remain in reverseIDs.
func (g *flowTapeGraph) reverseCommandCount() int {
	present := make([]bool, len(g.tape.Tensors))
	for id := range g.adj {
		present[id] = true
	}
	commands := 0
	accumulate := func(id int) {
		if present[id] {
			commands++
		} else {
			present[id] = true
		}
	}
	for i := len(g.nodes) - 1; i >= 0; i-- {
		node := g.nodes[i]
		present[node.output.v], present[node.output.t] = true, true
		switch node.op {
		case "linear":
			commands += 11
			x := node.input[0]
			for _, id := range []int{x.v, x.t, node.w, node.b} {
				accumulate(id)
			}
		case "norm":
			commands += 9
			if node.kind == 0 {
				commands++
			}
			x := node.input[0]
			for _, id := range []int{x.v, x.t, node.w} {
				accumulate(id)
			}
			if node.kind == 0 {
				accumulate(node.b)
			}
		case "silu":
			commands += 2
			x := node.input[0]
			accumulate(x.v)
			accumulate(x.t)
		case "mod":
			commands += 4
			a, b := node.input[0], node.input[1]
			for _, id := range []int{a.v, a.t, b.v, b.t} {
				accumulate(id)
			}
		case "add":
			commands += 2
			a, b := node.input[0], node.input[1]
			for _, id := range []int{a.v, a.t, b.v, b.t} {
				accumulate(id)
			}
		case "slice":
			commands += 2
			x := node.input[0]
			accumulate(x.v)
			accumulate(x.t)
		}
	}
	return commands
}
func (g *flowTapeGraph) reserveReverseMetadata() {
	// Bounded room for time roots and caller finish commands avoids immediately
	// regrowing this slice. Capacity does not alter the full-source byte ceiling.
	count := min(4096, len(g.tape.Commands)+g.reverseCommandCount()+64)
	if count > cap(g.tape.Commands) {
		next := make([]train.Command, len(g.tape.Commands), count)
		copy(next, g.tape.Commands)
		g.tape.Commands = next
	}
}
func flowForwardCommandCapacity(times, blocks int, deviceTime bool) int {
	// Each fixed forward topology has 35 base commands, 23 per time module,
	// 39 per residual block, and two device time-embedding commands per module.
	// Saturate incrementally so arbitrarily large admitted topology cannot cause
	// capacity arithmetic overflow. Validation still rejects oversize graphs.
	n := 35
	perTime := 23
	if deviceTime {
		perTime += 2
	}
	for i := 0; i < times && n < 4096; i++ {
		n = min(4096, n+perTime)
	}
	for i := 0; i < blocks && n < 4096; i++ {
		n = min(4096, n+39)
	}
	return min(4096, n+16) // bounded caller loss/seed composition headroom
}
