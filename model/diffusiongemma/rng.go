package diffusiongemma

// canvasRNG is the minimal random interface used by the DiffusionGemma
// entropy-bound loop. *rand.Rand satisfies it, and mt19937RNG below matches the
// reference engine family used by llama.cpp (std::mt19937) for engine-created
// runs.
type canvasRNG interface {
	Intn(n int) int
	Float64() float64
}

// mt19937RNG implements the 32-bit MT19937 engine used by std::mt19937. The
// exact C++ uniform distribution implementation is library-defined, but using
// the same engine avoids the much larger divergence of Go's math/rand source.
type mt19937RNG struct {
	mt    [624]uint32
	index int
}

func NewMT19937RNG(seed int64) *mt19937RNG {
	r := &mt19937RNG{}
	r.seed(uint32(seed))
	return r
}

func (r *mt19937RNG) seed(seed uint32) {
	r.mt[0] = seed
	for i := 1; i < len(r.mt); i++ {
		r.mt[i] = 1812433253*(r.mt[i-1]^(r.mt[i-1]>>30)) + uint32(i)
	}
	r.index = len(r.mt)
}

func (r *mt19937RNG) Uint32() uint32 {
	if r.index >= len(r.mt) {
		r.twist()
	}
	y := r.mt[r.index]
	r.index++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

func (r *mt19937RNG) twist() {
	const n = 624
	const m = 397
	const matrixA uint32 = 0x9908b0df
	const upperMask uint32 = 0x80000000
	const lowerMask uint32 = 0x7fffffff
	for i := 0; i < n; i++ {
		y := (r.mt[i] & upperMask) | (r.mt[(i+1)%n] & lowerMask)
		r.mt[i] = r.mt[(i+m)%n] ^ (y >> 1)
		if y&1 != 0 {
			r.mt[i] ^= matrixA
		}
	}
	r.index = 0
}

func (r *mt19937RNG) Float64() float64 {
	// std::uniform_real_distribution<float>(0,1) is implementation-defined;
	// use one 32-bit MT draw scaled into [0,1), matching the reference engine's
	// effective single-draw float distribution closely enough for parity work.
	return float64(r.Uint32()) / 4294967296.0
}

func (r *mt19937RNG) Intn(n int) int {
	if n <= 0 || uint64(n) > (uint64(1)<<32) {
		return 0
	}
	// Unbiased rejection sampling over the full 2^32 MT output domain.
	bound := uint64(n)
	domain := uint64(1) << 32
	limit := domain - domain%bound
	for {
		x := uint64(r.Uint32())
		if x < limit {
			return int(x % bound)
		}
	}
}
