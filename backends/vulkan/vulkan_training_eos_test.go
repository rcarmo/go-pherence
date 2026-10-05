package vulkan

import (
	"context"
	"encoding/binary"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
	"unsafe"
)

func TestVulkanOfflineTrainingEOS(t *testing.T) {
	c, err := InspectVulkanShader(spirvTrainingEOS)
	if err != nil || c.StorageBindings != 7 || c.PushBytes != 16 || c.LocalSize != [3]uint32{256, 1, 1} {
		t.Fatal(c, err)
	}
	_, k, _ := newLifetimeMock(t)
	k.numBuffers = 3
	k.pushSize = 16
	mockVK(t, &vkCmdPushConstants, func(VkCommandBuffer, VkPipelineLayout, uint32, uint32, uint32, unsafe.Pointer) {})
	op := &VkTrainingF32{eos: k}
	x, labels := linearMetadataTensor(1, 257), linearMetadataTensor(2, 257)
	for mode := uint32(0); mode < 3; mode++ {
		n, _ := train.EOSShape(257, 129, mode, .1)
		dst := linearMetadataTensor(3, n)
		s, err := op.EOSStage(context.Background(), dst, x, labels, 129, mode, .1)
		groups := uint32(2)
		if mode == 2 {
			groups = 1
		}
		if err != nil || s.PushWords[1] != 129 || s.PushWords[2] != mode || s.PushWords[3] != math.Float32bits(.1) || s.Groups != [3]uint32{groups, 1, 1} {
			t.Fatal(s, err)
		}
		if _, err := op.tapeStage(context.Background(), []*VkTensorF32{x, labels, dst}, train.Command{Op: "eos", Inputs: []int{0, 1}, Output: 2, Rows: 129, Mode: mode, Scale: .1}); err != nil {
			t.Fatal(err)
		}
	}
	dst := linearMetadataTensor(3, 257)
	for _, bad := range []*VkTrainingF32{nil, {}} {
		if _, err := bad.EOSStage(context.Background(), dst, x, labels, 129, 0, .1); err == nil {
			t.Fatal("owner")
		}
	}
	for _, b := range [][3]*VkTensorF32{{x, x, labels}, {dst, nil, labels}, {nil, x, labels}, {dst, x, nil}, {dst, x, linearMetadataTensor(4, 256)}, {linearMetadataTensor(4, 1, 257), x, labels}} {
		if _, err := op.EOSStage(context.Background(), b[0], b[1], b[2], 129, 0, .1); err == nil {
			t.Fatal("extent/alias")
		}
	}
	if _, err := op.EOSStage(context.Background(), dst, x, labels, 0, 0, .1); err == nil {
		t.Fatal("active count")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := op.EOSStage(ctx, dst, x, labels, 129, 0, .1); err == nil {
		t.Fatal("cancel")
	}
	if op.admitTapeKernels(train.Tape{Commands: []train.Command{{Op: "eos"}}}) != nil {
		t.Fatal("preflight")
	}
	// New unary Log is accepted; nearby Log2 is still rejected.
	code := append([]byte(nil), spirvTrainingEOS...)
	changed := false
	for pos := 20; pos < len(code); {
		word := binary.LittleEndian.Uint32(code[pos:])
		n := int(word >> 16)
		if uint16(word) == 12 && binary.LittleEndian.Uint32(code[pos+16:]) == 28 {
			binary.LittleEndian.PutUint32(code[pos+16:], 30)
			changed = true
			break
		}
		pos += 4 * n
	}
	if !changed {
		t.Fatal("Log missing")
	}
	if _, err := InspectVulkanShader(code); err == nil {
		t.Fatal("Log2 admitted")
	}
}
