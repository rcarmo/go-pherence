package main

import (
	"bytes"
	"testing"
)

func TestUsageAndValidation(t *testing.T) {
	for _, tc := range []struct {
		args []string
		ok   bool
	}{{[]string{"-help"}, true}, {nil, false}, {[]string{"-model", "none", "-text", "x", "-label", "person", "-threshold", "2"}, false}} {
		var out, errout bytes.Buffer
		err := run(tc.args, &out, &errout)
		if (err == nil) != tc.ok {
			t.Fatalf("%v: %v", tc.args, err)
		}
	}
}
