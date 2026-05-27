package speaker

import (
	"strings"
	"testing"
)

func TestLoadSpeechBrainECAPASafetensorsRejectsIncompleteCheckpoint(t *testing.T) {
	shapes := map[string][]int{}
	addTDNNShapes(shapes, "blocks.0", []int{1024, 80, 5}, 1024)
	path := writeECAPATestSafetensors(t, shapes)
	_, err := LoadSpeechBrainECAPASafetensors(path)
	if err == nil || !strings.Contains(err.Error(), "blocks.1.tdnn1.conv.conv.weight") {
		t.Fatalf("err=%v, want missing first ECAPA block tensor", err)
	}
}

func TestSpeechBrainECAPAContractShapeTable(t *testing.T) {
	shapes := speechBrainECAPAShapes()
	checks := map[string][]int{
		"blocks.0.conv.conv.weight":                        {1024, 80, 5},
		"blocks.1.tdnn1.conv.conv.weight":                  {1024, 1024, 1},
		"blocks.1.res2net_block.blocks.0.conv.conv.weight": {128, 128, 3},
		"mfa.conv.conv.weight":                             {3072, 3072, 1},
		"asp.tdnn.conv.conv.weight":                        {128, 9216, 1},
		"asp_bn.norm.weight":                               {6144},
		"fc.conv.weight":                                   {192, 6144, 1},
	}
	for name, want := range checks {
		if got := shapes[name]; !sameShape(got, want) {
			t.Fatalf("%s shape=%v want %v", name, got, want)
		}
	}
}

func speechBrainECAPAShapes() map[string][]int {
	shapes := map[string][]int{}
	addTDNNShapes(shapes, "blocks.0", []int{1024, 80, 5}, 1024)
	for i := 1; i <= 3; i++ {
		prefix := "blocks." + string(rune('0'+i))
		addTDNNShapes(shapes, prefix+".tdnn1", []int{1024, 1024, 1}, 1024)
		for j := 0; j < 7; j++ {
			addTDNNShapes(shapes, prefix+".res2net_block.blocks."+string(rune('0'+j)), []int{128, 128, 3}, 128)
		}
		addTDNNShapes(shapes, prefix+".tdnn2", []int{1024, 1024, 1}, 1024)
		addConvShapes(shapes, prefix+".se_block.conv1", []int{128, 1024, 1})
		addConvShapes(shapes, prefix+".se_block.conv2", []int{1024, 128, 1})
	}
	addTDNNShapes(shapes, "mfa", []int{3072, 3072, 1}, 3072)
	addTDNNShapes(shapes, "asp.tdnn", []int{128, 9216, 1}, 128)
	addConvShapes(shapes, "asp.conv", []int{3072, 128, 1})
	addBNShapes(shapes, "asp_bn", 6144)
	addConvShapes(shapes, "fc", []int{192, 6144, 1})
	return shapes
}

func addTDNNShapes(shapes map[string][]int, prefix string, conv []int, norm int) {
	addConvShapes(shapes, prefix+".conv", conv)
	addBNShapes(shapes, prefix+".norm", norm)
}

func addConvShapes(shapes map[string][]int, prefix string, conv []int) {
	shapes[prefix+".conv.weight"] = conv
	shapes[prefix+".conv.bias"] = []int{conv[0]}
}

func addBNShapes(shapes map[string][]int, prefix string, dim int) {
	shapes[prefix+".norm.weight"] = []int{dim}
	shapes[prefix+".norm.bias"] = []int{dim}
	shapes[prefix+".norm.running_mean"] = []int{dim}
	shapes[prefix+".norm.running_var"] = []int{dim}
}
