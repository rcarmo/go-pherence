package gguf

import (
	"regexp"
	"strconv"
)

func ggufMetaFloat64(v any) (float64, bool) {
	switch x := v.(type) {
	case float32:
		return float64(x), true
	case float64:
		return x, true
	case uint32:
		return float64(x), true
	case uint64:
		return float64(x), true
	case int32:
		return float64(x), true
	case int64:
		return float64(x), true
	case int:
		return float64(x), true
	}
	return 0, false
}

func inferREAPRatioFromName(name string) (float64, bool) {
	re := regexp.MustCompile(`(?i)reap[-_ ]?(\d{1,2})(?:\D|$)`)
	m := re.FindStringSubmatch(name)
	if len(m) < 2 {
		return 0, false
	}
	n, err := strconv.Atoi(m[1])
	if err != nil || n <= 0 || n >= 100 {
		return 0, false
	}
	return float64(n) / 100.0, true
}
