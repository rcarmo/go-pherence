package main

import "testing"

func TestProbePathIsAnOpaqueArgument(t *testing.T) {
	dir := "a\"; raise Exception('not code') #\\tail"
	c := probeCommand("gen.py", dir)
	if len(c.Args) != 3 || c.Args[1] != "gen.py" || c.Args[2] != dir {
		t.Fatal(c.Args)
	}
}
