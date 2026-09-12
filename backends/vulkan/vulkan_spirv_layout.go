package vulkan

import "fmt"

type vkSPIRVLayoutVariable struct {
	pointer, storage   uint32
	local, initializer bool
}
type vkSPIRVLayout struct {
	structs       map[uint32][]uint32
	variables     map[uint32]vkSPIRVLayoutVariable
	decorations   map[[2]uint32]uint32
	offsets       map[[2]uint32]uint32
	memberTargets [][2]uint32
}

// Inspect only layouts we can bind without guessing. No recursion, descriptor
// arrays, multiple sets, UBOs, nested blocks, matrices/vectors or decorated shared
// layouts. Declarations are counted even if unused by main. General descriptor
// liveness and instruction/variable-use validation still require spirv-val.
func (l vkSPIRVLayout) inspect(types map[uint32]vkSPIRVType, version uint32) (uint32, uint32, error) {
	fail := func(s string) (uint32, uint32, error) { return 0, 0, fmt.Errorf("shader interface: %s", s) }
	dec := func(id, kind uint32) (uint32, bool) { v, ok := l.decorations[[2]uint32{id, kind}]; return v, ok }
	scalar := func(id uint32) bool { t := types[id]; return t.op == 21 || t.op == 22 }
	// Reject misplaced layout metadata instead of silently ignoring it.
	for key := range l.decorations {
		id, kind := key[0], key[1]
		switch kind {
		case 2, 3:
			if types[id].op != 30 {
				return fail("block decoration on non-struct")
			}
			if _, both := dec(id, 5-kind); both {
				return fail("Block and BufferBlock conflict")
			}
		case 6:
			if t := types[id]; t.op != 28 && t.op != 29 {
				return fail("ArrayStride on non-array")
			}
		case 33, 34:
			v, ok := l.variables[id]
			if !ok || (v.storage != 2 && v.storage != 12) {
				return fail("descriptor decoration on non-buffer variable")
			}
		}
	}
	for _, target := range l.memberTargets {
		members, ok := l.structs[target[0]]
		if !ok || uint64(target[1]) >= uint64(len(members)) {
			return fail("member decoration index/struct")
		}
	}
	var bindings, push uint32
	pushSeen := false
	for id, v := range l.variables {
		p := types[v.pointer]
		if p.op != 32 || p.a != v.storage {
			return fail("pointer storage mismatch")
		}
		switch v.storage {
		case 2, 12:
			if v.local || v.initializer {
				return fail("descriptor scope/initializer")
			}
			if v.storage == 12 && version < 0x10300 {
				return fail("StorageBuffer needs SPIR-V1.3 in extension-free envelope")
			}
			set, hasSet := dec(id, 34)
			binding, hasBinding := dec(id, 33)
			if !hasSet || !hasBinding || set != 0 || binding >= 16 {
				return fail("requires set0 binding0..15")
			}
			bit := uint32(1) << binding
			if bindings&bit != 0 {
				return fail("duplicate binding")
			}
			bindings |= bit
			members, ok := l.structs[p.b]
			if !ok || len(members) != 1 {
				return fail("storage block must have one runtime-array member")
			}
			marker := uint32(3)
			if v.storage == 12 {
				marker = 2
			}
			if _, ok := dec(p.b, marker); !ok {
				return fail("storage class/block marker mismatch (UBO unsupported)")
			}
			offset, ok := l.offsets[[2]uint32{p.b, 0}]
			if !ok || offset != 0 {
				return fail("storage member must start at0")
			}
			array := types[members[0]]
			if array.op != 29 || !scalar(array.a) {
				return fail("storage requires runtime array of32-bit scalars")
			}
			stride, ok := dec(members[0], 6)
			if !ok || stride != 4 {
				return fail("storage scalar stride must be4")
			}
		case 9:
			if v.local || v.initializer || pushSeen {
				return fail("one module-scope push block required")
			}
			pushSeen = true
			members, ok := l.structs[p.b]
			if !ok || len(members) == 0 || len(members) > 32 {
				return fail("push block member count")
			}
			if _, ok := dec(p.b, 2); !ok {
				return fail("push struct requires Block")
			}
			// Four-byte scalars with explicit aligned non-overlapping offsets. Gaps
			// allowed, but whole declared end must fit our fixed128-byte host cap.
			var occupied uint32
			for i, member := range members {
				if !scalar(member) {
					return fail("push members must be32-bit scalars")
				}
				offset, ok := l.offsets[[2]uint32{p.b, uint32(i)}]
				if !ok || offset%4 != 0 || offset > 124 {
					return fail("push offset missing/unaligned/out of range")
				}
				bit := uint32(1) << (offset / 4)
				if occupied&bit != 0 {
					return fail("overlapping push members")
				}
				occupied |= bit
				push = max(push, offset+4)
			}
		}
	}
	return bindings, push, nil
}

// Extra unused descriptor slots and a larger (already device-checked) push
// range are legal. Require every declared resource, not exact counts: static
// use analysis is intentionally absent. Legacy LoadSPIRV has no push range.
func vkCheckShaderInterface(c VulkanShaderContract, buffers, pushBytes int) error {
	if buffers < 1 || buffers > 16 || pushBytes < 0 || pushBytes > 128 || pushBytes%4 != 0 {
		return fmt.Errorf("%w: invalid caller layout", ErrVulkanShaderContract)
	}
	mask := (uint32(1) << uint(buffers)) - 1
	if c.StorageBindings&^mask != 0 {
		return fmt.Errorf("%w: shader binding mask=%#x exceeds%d slots", ErrVulkanShaderContract, c.StorageBindings, buffers)
	}
	if c.PushBytes > uint32(pushBytes) {
		return fmt.Errorf("%w: shader push bytes=%d exceed caller%d", ErrVulkanShaderContract, c.PushBytes, pushBytes)
	}
	return nil
}
