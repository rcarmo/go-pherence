package needle

// NumPy RandomState uses MT19937 plus masked rejection for shuffle intervals.
// These fixed permutations are architecture constants, not training randomness.
type mt19937 struct {
	state [624]uint32
	pos   int
}

func newMT(seed uint32) *mt19937 {
	m := &mt19937{pos: 624}
	m.state[0] = seed
	for i := 1; i < 624; i++ {
		m.state[i] = 1812433253*(m.state[i-1]^(m.state[i-1]>>30)) + uint32(i)
	}
	return m
}
func (m *mt19937) next() uint32 {
	if m.pos == 624 {
		for i := 0; i < 624; i++ {
			y := (m.state[i] & 0x80000000) | (m.state[(i+1)%624] & 0x7fffffff)
			m.state[i] = m.state[(i+397)%624] ^ (y >> 1)
			if y&1 != 0 {
				m.state[i] ^= 0x9908b0df
			}
		}
		m.pos = 0
	}
	y := m.state[m.pos]
	m.pos++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}
func (m *mt19937) interval(max uint32) uint32 {
	mask := max
	mask |= mask >> 1
	mask |= mask >> 2
	mask |= mask >> 4
	mask |= mask >> 8
	mask |= mask >> 16
	for {
		v := m.next() & mask
		if v <= max {
			return v
		}
	}
}
func numpyPermutation(n int, seed uint32, split bool) []int {
	if split {
		a := numpyPermutation(n/2, seed, false)
		b := numpyPermutation(n/2, seed+977, false)
		for _, i := range b {
			a = append(a, n/2+i)
		}
		return a
	}
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	m := newMT(seed)
	for i := n - 1; i > 0; i-- {
		j := int(m.interval(uint32(i)))
		p[i], p[j] = p[j], p[i]
	}
	return p
}
