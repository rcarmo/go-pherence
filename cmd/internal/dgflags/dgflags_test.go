package dgflags

import (
	"math"
	"strconv"
	"testing"
)

func TestExpertCacheBudgetBytesNoWrap(t *testing.T) {
	for _, tc := range []struct {
		env            string
		fallback, want int64
	}{
		{"", 7, 7 << 20}, {"12", 7, 12 << 20}, {"0", 7, 0}, {"bad", 7, 7 << 20}, {"-1", 7, 7 << 20}, {"", -1, 0},
		{strconv.FormatInt(math.MaxInt64, 10), 7, math.MaxInt64}, {"", math.MaxInt64, math.MaxInt64},
	} {
		t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_EXPERT_CACHE_MB", tc.env)
		if got := ExpertCacheBudgetBytes(tc.fallback); got != tc.want {
			t.Fatal(tc, got)
		}
	}
}
