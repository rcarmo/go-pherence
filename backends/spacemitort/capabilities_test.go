package spacemitort

import "testing"

func TestRuntimeCapabilities(t *testing.T) {
	c := RuntimeCapabilities()
	if c.Arch == "" {
		t.Fatal("empty arch")
	}
	if c.LikelyUsable && (!c.HasDevTCM || !c.HasSpineTCMLibrary || !c.HasSpacemitEPLibrary || !c.HasONNXRuntimeLibrary) {
		t.Fatalf("LikelyUsable set with incomplete deps: %+v", c)
	}
	t.Logf("spacemitort capabilities: %+v", c)
}

func TestProviderOptions(t *testing.T) {
	opts := Options{IntraThreadNum: 4, DumpSubgraphs: true, DebugProfilePrefix: "profile_", DenseAccuracyLevel: 1, EnableDMA: true}
	m := opts.ProviderOptions()
	if m["SPACEMIT_EP_INTRA_THREAD_NUM"] != "4" || m["SPACEMIT_EP_DUMP_SUBGRAPHS"] != "1" || m["SPACEMIT_EP_DEBUG_PROFILE"] != "profile_" || m["SPACEMIT_EP_DENSE_ACCURACY_LEVEL"] != "1" || m["SPACEMIT_EP_ENABLE_DMA"] != "1" {
		t.Fatalf("unexpected provider options: %#v", m)
	}
}
