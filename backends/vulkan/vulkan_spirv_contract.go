package vulkan

import (
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
)

var ErrVulkanShaderContract = errors.New("unsupported or malformed Vulkan shader contract")

// VulkanShaderContract is bounded, conservative module metadata. Successful
// inspection is NOT full SPIR-V validation or proof of numerical correctness,
// memory safety or device compatibility. Descriptor/push reflection covers only
// the narrow scalar-buffer/flat-push envelope documented below. The supported
// envelope is intentionally narrower than Vulkan: SPIR-V1.0..1.3, Shader only,
// Logical/GLSL450, one GLCompute main, fixed LocalSize and core32bit types.
// Extended instructions are limited to FAbs, Exp, InverseSqrt and Fma.
// Workgroup storage may be 32-bit scalars or fixed one-dimensional scalar arrays.
// No specialization, extensions, optional features or decorated shared layouts.
type VulkanShaderContract struct {
	LocalSize   [3]uint32
	SharedBytes uint32
	// Set0 storage-buffer bindings, one descriptor each (bits0..15). Callers
	// may supply unused extra slots. PushBytes is the declared range from zero;
	// padded/superset caller ranges are allowed. Both are conservative over all
	// module declarations, not entrypoint liveness analysis.
	StorageBindings uint32
	PushBytes       uint32
}

type vkSPIRVType struct {
	op   uint16
	a, b uint32
}
type vkSPIRVConstant struct{ typ, value uint32 }

// InspectVulkanShader does no driver calls. It copies the input before parsing;
// callers must not mutate input concurrently. Admission is capped at1MiB/65536
// IDs, with a closed opcode envelope and unique result IDs. Known instructions
// still need normal offline spirv-val validation: function/control-flow/operand
// semantics remain outside this check. Interfaces support set0 storage blocks
// of one runtime array of32-bit scalars and one flat32-bit-scalar push block.
func InspectVulkanShader(code []byte) (VulkanShaderContract, error) {
	if len(code) < 20 || len(code) > 1<<20 || len(code)%4 != 0 {
		return VulkanShaderContract{}, fmt.Errorf("%w: byte length", ErrVulkanShaderContract)
	}
	words := make([]uint32, len(code)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(code[4*i:])
	}
	return vkInspectSPIRV(words)
}
func vkInspectSPIRV(w []uint32) (VulkanShaderContract, error) {
	fail := func(reason string) (VulkanShaderContract, error) {
		return VulkanShaderContract{}, fmt.Errorf("%w: %s", ErrVulkanShaderContract, reason)
	}
	if len(w) < 5 || len(w) > 1<<18 || w[0] != 0x07230203 || w[1] < 0x10000 || w[1] > 0x10300 || w[1]&0xff != 0 || w[3] == 0 || w[3] > 65536 || w[4] != 0 {
		return fail("header/version/ID bound")
	}
	bound := w[3]
	ids := make([]bool, bound)
	types := map[uint32]vkSPIRVType{}
	constants := map[uint32]vkSPIRVConstant{}
	composites := map[uint32][]uint32{}
	variables := map[uint32]uint32{}
	allVariables := [][2]uint32{} // pointer type and declared storage must agree
	layout := vkSPIRVLayout{structs: map[uint32][]uint32{}, variables: map[uint32]vkSPIRVLayoutVariable{}, decorations: map[[2]uint32]uint32{}, offsets: map[[2]uint32]uint32{}}
	imports := map[uint32]bool{}
	extSets := []uint32{}
	functions := map[uint32]bool{}
	decorated := map[uint32]bool{}
	builtinObjects := []uint32{}
	var entry, modeEntry uint32
	var contract VulkanShaderContract
	capability, memory, entrySeen, modeSeen := false, false, false, false
	insideFunction := false
	// Decoration targets/pairs must be unambiguous (in particular BuiltIn).
	seenDecorations := map[[3]uint32]bool{}
	validID := func(id uint32) bool { return id > 0 && id < bound }
	for pos := 5; pos < len(w); {
		count, op := int(w[pos]>>16), uint16(w[pos])
		info, ok := vkSPIRVOps[op]
		if !ok || count < info.minWords || count > len(w)-pos || (info.maxWords != 0 && count != info.maxWords) {
			return fail("opcode/word framing")
		}
		a := w[pos : pos+count]
		pos += count
		if info.result > 0 {
			if info.result >= len(a) || !validID(a[info.result]) || ids[a[info.result]] {
				return fail("duplicate/out-of-bound result ID")
			}
			ids[a[info.result]] = true
		}
		switch op {
		case 17: // Capability
			if count != 2 || a[1] != 1 || capability {
				return fail("only Shader capability supported")
			}
			capability = true
		case 11: // ExtInstImport (only baseline GLSL.std.450)
			name, n, ok := vkSPIRVString(a[2:])
			if !ok || n != len(a)-2 || name != "GLSL.std.450" {
				return fail("extended instruction import")
			}
			imports[a[1]] = true
		case 12: // GLSL.std.450 FAbs/Exp/InverseSqrt or ternary Fma.
			unary := count == 6 && (a[4] == 4 || a[4] == 27 || a[4] == 32)
			ternary := count == 8 && a[4] == 50
			if !unary && !ternary {
				return fail("extended instruction not admitted")
			}
			extSets = append(extSets, a[3])
		case 14: // MemoryModel
			if count != 3 || memory || a[1] != 0 || a[2] != 1 {
				return fail("Logical GLSL450 memory model required")
			}
			memory = true
		case 15: // EntryPoint
			if count < 4 || entrySeen || a[1] != 5 || !validID(a[2]) {
				return fail("one compute entrypoint required")
			}
			name, n, ok := vkSPIRVString(a[3:])
			if !ok || name != "main" {
				return fail("entrypoint must be main")
			}
			for _, id := range a[3+n:] {
				if !validID(id) {
					return fail("interface ID")
				}
			}
			entry = a[2]
			entrySeen = true
		case 16: // ExecutionMode (LocalSize only; no IDs/spec constants)
			if count != 6 || modeSeen || a[2] != 17 || !validID(a[1]) {
				return fail("one fixed LocalSize required")
			}
			modeEntry = a[1]
			modeSeen = true
			copy(contract.LocalSize[:], a[3:6])
			for _, n := range contract.LocalSize {
				if n == 0 {
					return fail("zero local size")
				}
			}
		case 21: // TypeInt
			if count != 4 || a[2] != 32 || a[3] > 1 {
				return fail("only32bit integers supported")
			}
			types[a[1]] = vkSPIRVType{op, a[2], a[3]}
		case 22: // TypeFloat
			if count != 3 || a[2] != 32 {
				return fail("only32bit floats supported")
			}
			types[a[1]] = vkSPIRVType{op, a[2], 0}
		case 23: // TypeVector
			if count != 4 || a[3] < 2 || a[3] > 4 {
				return fail("vector type")
			}
			types[a[1]] = vkSPIRVType{op, a[2], a[3]}
		case 28, 32: // TypeArray / TypePointer
			if count != 4 {
				return fail("array/pointer arity")
			}
			types[a[1]] = vkSPIRVType{op, a[2], a[3]}
			if op == 32 && !vkSPIRVStorage(a[2]) {
				return fail("storage class")
			}
		case 29: // RuntimeArray
			types[a[1]] = vkSPIRVType{op, a[2], 0}
		case 30: // Struct (flat layouts inspected after all declarations)
			types[a[1]] = vkSPIRVType{op, 0, 0}
			layout.structs[a[1]] = append([]uint32(nil), a[2:]...)
		case 19, 20, 33: // other core type declarations
			types[a[1]] = vkSPIRVType{op, 0, 0}
		case 43: // Constant (32-bit only)
			if count != 4 {
				return fail("constant width")
			}
			constants[a[2]] = vkSPIRVConstant{a[1], a[3]}
		case 44: // ConstantComposite; retain type and constituent IDs
			composites[a[2]] = append([]uint32(nil), a[1:]...)
		case 59: // Variable
			if count < 4 || count > 5 || !vkSPIRVStorage(a[3]) {
				return fail("variable storage")
			}
			allVariables = append(allVariables, [2]uint32{a[1], a[3]})
			layout.variables[a[2]] = vkSPIRVLayoutVariable{pointer: a[1], storage: a[3], local: insideFunction, initializer: count == 5}
			if a[3] == 4 {
				if insideFunction || count != 4 {
					return fail("shared initializer/scope")
				}
				variables[a[2]] = a[1]
			}
		case 54:
			if insideFunction {
				return fail("nested function")
			}
			insideFunction = true
			functions[a[2]] = true
		case 56:
			if !insideFunction {
				return fail("function end")
			}
			insideFunction = false
		case 71: // Decorate: block/layout/interface decorations used by embedded shaders
			if count < 3 || !validID(a[1]) {
				return fail("decoration target")
			}
			key := [3]uint32{a[1], ^uint32(0), a[2]}
			if seenDecorations[key] {
				return fail("duplicate decoration")
			}
			seenDecorations[key] = true
			decorated[a[1]] = true
			switch a[2] {
			case 2, 3, 24, 25, 42: //42=NoContraction (precise arithmetic)
				if count != 3 {
					return fail("decoration arity")
				}
			case 6, 33, 34:
				if count != 4 {
					return fail("layout decoration arity")
				}
			case 11:
				if count != 4 {
					return fail("builtin arity")
				}
				switch a[3] {
				case 24, 26, 27, 28, 29:
				case 25:
					builtinObjects = append(builtinObjects, a[1])
				default:
					return fail("unsupported builtin")
				}
			default:
				return fail("unsupported decoration")
			}
			value := uint32(0)
			if count == 4 {
				value = a[3]
			}
			layout.decorations[[2]uint32{a[1], a[2]}] = value
		case 72: // MemberDecorate: only offset/read/write annotation
			if count < 4 || !validID(a[1]) {
				return fail("member target")
			}
			key := [3]uint32{a[1], a[2], a[3]}
			if seenDecorations[key] {
				return fail("duplicate member decoration")
			}
			seenDecorations[key] = true
			decorated[a[1]] = true
			if (a[3] == 35 && count == 5) || ((a[3] == 24 || a[3] == 25) && count == 4) {
			} else {
				return fail("unsupported member decoration")
			}
			layout.memberTargets = append(layout.memberTargets, [2]uint32{a[1], a[2]})
			if a[3] == 35 {
				layout.offsets[[2]uint32{a[1], a[2]}] = a[4]
			}
		}
	}
	if !capability || !memory || !entrySeen || !modeSeen || entry != modeEntry || !functions[entry] || insideFunction {
		return fail("missing/inconsistent compute declarations")
	}
	for _, set := range extSets {
		if !imports[set] {
			return fail("undefined extended instruction set")
		}
	}
	for _, variable := range allVariables {
		p := types[variable[0]]
		if p.op != 32 || p.a != variable[1] {
			return fail("variable/pointer storage mismatch")
		}
	}
	scalar32 := func(id uint32) bool { t := types[id]; return t.op == 21 || t.op == 22 }
	integer := func(id uint32) (uint32, bool) {
		c, ok := constants[id]
		t := types[c.typ]
		return c.value, ok && t.op == 21 && (t.b == 0 || c.value <= 0x7fffffff)
	}
	// WorkgroupSize builtin overrides execution mode in SPIR-V. Accept only a
	// literal uint/int32 vec3 composite agreeing exactly with LocalSize.
	if len(builtinObjects) > 1 {
		return fail("multiple WorkgroupSize builtins")
	}
	for _, id := range builtinObjects {
		c := composites[id]
		if len(c) != 5 {
			return fail("WorkgroupSize must be literal vector3")
		}
		v := types[c[0]]
		if v.op != 23 || v.b != 3 || types[v.a].op != 21 {
			return fail("WorkgroupSize vector type")
		}
		for i, cid := range c[2:] {
			n, ok := integer(cid)
			if !ok || constants[cid].typ != v.a || n != contract.LocalSize[i] {
				return fail("WorkgroupSize disagrees with LocalSize")
			}
		}
	}
	var shared uint64
	for variable, pointer := range variables {
		ptr := types[pointer]
		if ptr.op != 32 || ptr.a != 4 || decorated[variable] || decorated[pointer] {
			return fail("shared pointer/layout")
		}
		target := ptr.b
		if decorated[target] {
			return fail("decorated shared type")
		}
		size := uint64(4)
		if !scalar32(target) {
			arr := types[target]
			if arr.op != 28 || !scalar32(arr.a) || decorated[arr.a] {
				return fail("shared layout not a flat32bit scalar array")
			}
			n, ok := integer(arr.b)
			if !ok || n == 0 {
				return fail("shared array length")
			}
			size = uint64(n) * 4
		}
		shared += size
		if shared > uint64(^uint32(0)) {
			return fail("shared size overflow")
		}
	}
	contract.SharedBytes = uint32(shared)
	bindings, pushBytes, err := layout.inspect(types, w[1])
	if err != nil {
		return fail(err.Error())
	}
	contract.StorageBindings, contract.PushBytes = bindings, pushBytes
	return contract, nil
}

func vkSPIRVStorage(storage uint32) bool {
	switch storage {
	case 1, 2, 4, 6, 7, 9, 12:
		return true
	}
	return false
}

// Return a NUL-terminated literal string and consumed words; reject nonzero
// padding so string termination cannot hide other operands in the same word.
func vkSPIRVString(words []uint32) (string, int, bool) {
	var b strings.Builder
	for i, w := range words {
		for j := uint(0); j < 4; j++ {
			v := byte(w >> (8 * j))
			if v == 0 {
				if w>>(8*j) != 0 {
					return "", 0, false
				}
				return b.String(), i + 1, true
			}
			b.WriteByte(v)
		}
	}
	return "", 0, false
}
func vkCheckShaderLimits(contract VulkanShaderContract, limits VulkanDeviceLimits) error {
	if err := limits.validate(); err != nil {
		return err
	}
	product := uint64(1)
	for i, n := range contract.LocalSize {
		if n == 0 || n > limits.WorkgroupSize[i] {
			return fmt.Errorf("%w: shader local size axis%d=%d max=%d", ErrVulkanLimit, i, n, limits.WorkgroupSize[i])
		}
		if uint64(n) > uint64(limits.WorkgroupInvocations)/product {
			return fmt.Errorf("%w: shader local invocations exceed%d", ErrVulkanLimit, limits.WorkgroupInvocations)
		}
		product *= uint64(n)
	}
	if contract.SharedBytes > limits.SharedMemoryBytes {
		return fmt.Errorf("%w: shader shared bytes=%d max=%d", ErrVulkanLimit, contract.SharedBytes, limits.SharedMemoryBytes)
	}
	return nil
}
