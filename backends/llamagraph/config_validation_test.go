package llamagraph

import (
	"math"
	"testing"
)

func testConfig() Config {
	return Config{NVocab: 16, NEmbd: 8, NHeads: 2, NHeadsKV: 1, NLayers: 2, NFF: 16, NCtx: 32, RopeBase: 10000, RmsEps: 1e-5, RopeDims: 4, NThreads: 1, WQType: []int{0, 0}, WKType: []int{0, 0}, WVType: []int{0, 0}, WOType: []int{0, 0}, FFNGateType: []int{0, 0}, FFNUpType: []int{0, 0}, FFNDownType: []int{0, 0}}
}
func TestConfigValidDefaultAndGQAOverride(t *testing.T) {
	c := testConfig()
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	c.WQOut = []int{16, 16}
	c.WOIn = []int{16, 16}
	c.WKOut = []int{8, 8}
	c.WVOut = []int{8, 8}
	c.RopeDims = 8
	c.HasQKNorm = true
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
}
func TestConfigRejectsBeforeNativeArrayAccess(t *testing.T) {
	cases := map[string]func(*Config){
		"zero heads": func(c *Config) { c.NHeads = 0 }, "negative layers": func(c *Config) { c.NLayers = -1 }, "array limit": func(c *Config) { c.NLayers = 129 },
		"short dtype": func(c *Config) { c.WQType = nil }, "short override": func(c *Config) { c.WKOut = []int{4} }, "negative override": func(c *Config) { c.WQOut = []int{-1, 0} },
		"narrowing": func(c *Config) { c.NVocab = int(^uint(0) >> 1) }, "bad grouping": func(c *Config) { c.NHeadsKV = 3 }, "odd rope": func(c *Config) { c.RopeDims = 3 },
		"NaN": func(c *Config) { c.RmsEps = float32(math.NaN()) }, "Inf": func(c *Config) { c.RopeBase = float32(math.Inf(1)) },
		"bad dtype": func(c *Config) { c.OutputType = -1 }, "inconsistent Q": func(c *Config) { c.WQOut = []int{8, 16} }, "inconsistent K": func(c *Config) { c.WKOut = []int{4, 8} },
		"extent overflow": func(c *Config) {
			c.NCtx = math.MaxInt32
			c.NEmbd = math.MaxInt32 - 1
			c.NFF = math.MaxInt32
			c.RopeDims = 2
		},
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			c := testConfig()
			change(&c)
			if c.Validate() == nil {
				t.Fatal("accepted invalid config")
			}
		})
	}
}

// All build variants must expose the same optional methods.
var _ interface {
	TieOutputEmbeddings()
	SetLayerQNorm(int, []byte)
	SetLayerKNorm(int, []byte)
	SetMTPENorm([]byte)
	SetMTPHNorm([]byte)
	SetMTPEHProj([]byte)
	SetMTPSharedHeadNorm([]byte)
} = (*Model)(nil)
