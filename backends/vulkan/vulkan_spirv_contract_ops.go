package vulkan

// Closed core-opcode envelope for the existing embedded compute shaders.
// Unknown instructions fail admission. This is framing/result-ID metadata,
// NOT semantic SPIR-V validation. Operand positions and fixed arities from
// SPIRV-Headers vulkan-sdk-1.3.296.0 spirv.core.grammar.json SHA256
// 83ee1c44ae5d87d54bb118be8ff041a9349661dd6c5ce0e02e1e4b539a6996bc.
// result is a word index including the instruction header; zero means absent.
type vkSPIRVOp struct{ minWords, maxWords, result int }

var vkSPIRVOps = map[uint16]vkSPIRVOp{
	0:   {1, 1, 0}, // OpNop
	3:   {3, 0, 0}, // OpSource
	5:   {3, 0, 0}, // OpName
	6:   {4, 0, 0}, // OpMemberName
	11:  {3, 0, 1}, // OpExtInstImport
	12:  {5, 0, 2}, // OpExtInst
	14:  {3, 3, 0}, // OpMemoryModel
	15:  {4, 0, 0}, // OpEntryPoint
	16:  {3, 0, 0}, // OpExecutionMode
	17:  {2, 2, 0}, // OpCapability
	19:  {2, 2, 1}, // OpTypeVoid
	20:  {2, 2, 1}, // OpTypeBool
	21:  {4, 4, 1}, // OpTypeInt
	22:  {3, 0, 1}, // OpTypeFloat
	23:  {4, 4, 1}, // OpTypeVector
	28:  {4, 4, 1}, // OpTypeArray
	29:  {3, 3, 1}, // OpTypeRuntimeArray
	30:  {2, 0, 1}, // OpTypeStruct
	32:  {4, 4, 1}, // OpTypePointer
	33:  {3, 0, 1}, // OpTypeFunction
	43:  {4, 0, 2}, // OpConstant
	44:  {3, 0, 2}, // OpConstantComposite
	54:  {5, 5, 2}, // OpFunction
	55:  {3, 3, 2}, // OpFunctionParameter
	56:  {1, 1, 0}, // OpFunctionEnd
	57:  {4, 0, 2}, // OpFunctionCall
	59:  {4, 0, 2}, // OpVariable
	61:  {4, 0, 2}, // OpLoad
	62:  {3, 0, 0}, // OpStore
	65:  {4, 0, 2}, // OpAccessChain
	71:  {3, 0, 0}, // OpDecorate
	72:  {4, 0, 0}, // OpMemberDecorate
	112: {4, 4, 2}, // OpConvertUToF
	124: {4, 4, 2}, // OpBitcast
	127: {4, 4, 2}, // OpFNegate
	128: {5, 5, 2}, // OpIAdd
	129: {5, 5, 2}, // OpFAdd
	130: {5, 5, 2}, // OpISub
	131: {5, 5, 2}, // OpFSub
	132: {5, 5, 2}, // OpIMul
	133: {5, 5, 2}, // OpFMul
	134: {5, 5, 2}, // OpUDiv
	136: {5, 5, 2}, // OpFDiv
	137: {5, 5, 2}, // OpUMod
	168: {4, 4, 2}, // OpLogicalNot
	170: {5, 5, 2}, // OpIEqual
	171: {5, 5, 2}, // OpINotEqual
	172: {5, 5, 2}, // OpUGreaterThan
	174: {5, 5, 2}, // OpUGreaterThanEqual
	176: {5, 5, 2}, // OpULessThan
	184: {5, 5, 2}, // OpFOrdLessThan
	186: {5, 5, 2}, // OpFOrdGreaterThan
	188: {5, 5, 2}, // OpFOrdLessThanEqual
	190: {5, 5, 2}, // OpFOrdGreaterThanEqual
	194: {5, 5, 2}, // OpShiftRightLogical
	196: {5, 5, 2}, // OpShiftLeftLogical
	197: {5, 5, 2}, // OpBitwiseOr
	199: {5, 5, 2}, // OpBitwiseAnd
	224: {4, 4, 0}, // OpControlBarrier
	245: {3, 0, 2}, // OpPhi
	246: {4, 0, 0}, // OpLoopMerge
	247: {3, 3, 0}, // OpSelectionMerge
	248: {2, 2, 1}, // OpLabel
	249: {2, 2, 0}, // OpBranch
	250: {4, 0, 0}, // OpBranchConditional
	253: {1, 1, 0}, // OpReturn
	254: {2, 2, 0}, // OpReturnValue
}
