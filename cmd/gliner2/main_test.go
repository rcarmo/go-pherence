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

func TestRecordSpec(t *testing.T) {
	s, err := recordSpec([]string{"name:str", "city:list"}, "natural", "")
	if err != nil || s.AnchorQueryID != 0 || !s.Fields[0].Scalar || s.Fields[1].Scalar {
		t.Fatal(s, err)
	}
	for _, fields := range [][]string{nil, {"name:required"}, {"name:str", "name:list"}, {"name"}} {
		if _, err := recordSpec(fields, "natural", ""); err == nil {
			t.Fatal("accepted", fields)
		}
	}
	if _, err := recordSpec([]string{"name:str"}, "latent", "name"); err == nil {
		t.Fatal("latent anchor accepted")
	}
	if _, err := recordSpec([]string{"name:str"}, "natural", "missing"); err == nil {
		t.Fatal("missing anchor accepted")
	}
}

func TestRecordFieldModifiers(t *testing.T) {
	s, err := recordSpec([]string{"name:str,required,exclusive", "city:list,exclusive"}, "natural", "")
	if err != nil || !s.Fields[0].Required || !s.Fields[0].Exclusive || !s.Fields[1].Exclusive {
		t.Fatal(s, err)
	}
	for _, v := range []string{"name:list,required", "name:str,unknown", "name:str,exclusive,exclusive", "name:str,"} {
		if _, err := recordSpec([]string{v}, "natural", ""); err == nil {
			t.Fatal("accepted", v)
		}
	}
}
