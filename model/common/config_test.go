package common

import "testing"

func TestModelMetadataPredicates(t *testing.T) {
	if (Config{}).HasNativeMTP() || (Config{}).IsOrthrus() {
		t.Fatal("empty config advertised support")
	}
	c := Config{MTPNumHiddenLayers: 1, OrthrusBlockSize: 8, OrthrusMaskTokenID: 0, Architectures: []string{"OrthrusLM"}}
	if !c.HasNativeMTP() || !c.IsOrthrus() {
		t.Fatal(c)
	}
	c.OrthrusMaskTokenID = -1
	if c.IsOrthrus() {
		t.Fatal("negative mask")
	}
	c.OrthrusMaskTokenID = 0
	c.Architectures = []string{"Other"}
	if c.IsOrthrus() {
		t.Fatal("wrong architecture")
	}
}
