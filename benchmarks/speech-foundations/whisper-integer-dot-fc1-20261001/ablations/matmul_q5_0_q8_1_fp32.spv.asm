; SPIR-V
; Version: 1.5
; Generator: Google Shaderc over Glslang; 11
; Bound: 1762
; Schema: 0
               OpCapability Shader
               OpCapability Int16
               OpCapability StorageBuffer16BitAccess
               OpCapability StorageBuffer8BitAccess
               OpCapability DotProductInput4x8BitPacked
               OpCapability DotProduct
               OpExtension "SPV_KHR_integer_dot_product"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %4 "main" %42 %55 %221 %230 %gl_WorkGroupID %318 %gl_LocalInvocationID %859
               OpExecutionMode %4 LocalSize 1 1 1
               OpDecorate %38 SpecId 1
               OpDecorate %_arr_ushort_uint_2 ArrayStride 2
               OpDecorate %_arr_ushort_uint_8 ArrayStride 2
               OpMemberDecorate %_struct_51 0 Offset 0
               OpMemberDecorate %_struct_51 1 Offset 2
               OpMemberDecorate %_struct_51 2 Offset 6
               OpDecorate %_runtimearr__struct_51 ArrayStride 22
               OpDecorate %_struct_53 Block
               OpMemberDecorate %_struct_53 0 NonWritable
               OpMemberDecorate %_struct_53 0 Offset 0
               OpDecorate %55 NonWritable
               OpDecorate %55 Binding 0
               OpDecorate %55 DescriptorSet 0
               OpDecorate %89 SpecId 6
               OpDecorate %90 SpecId 7
               OpDecorate %217 SpecId 2
               OpDecorate %_arr_v2half_uint_4 ArrayStride 4
               OpDecorate %_arr_v4int_uint_8 ArrayStride 16
               OpMemberDecorate %_struct_226 0 Offset 0
               OpMemberDecorate %_struct_226 1 Offset 16
               OpDecorate %_runtimearr__struct_226 ArrayStride 144
               OpDecorate %_struct_228 Block
               OpMemberDecorate %_struct_228 0 NonWritable
               OpMemberDecorate %_struct_228 0 Offset 0
               OpDecorate %230 NonWritable
               OpDecorate %230 Binding 1
               OpDecorate %230 DescriptorSet 0
               OpDecorate %gl_WorkGroupID BuiltIn WorkgroupId
               OpDecorate %_struct_316 Block
               OpMemberDecorate %_struct_316 0 Offset 0
               OpMemberDecorate %_struct_316 1 Offset 4
               OpMemberDecorate %_struct_316 2 Offset 8
               OpMemberDecorate %_struct_316 3 Offset 12
               OpMemberDecorate %_struct_316 4 Offset 16
               OpMemberDecorate %_struct_316 5 Offset 20
               OpMemberDecorate %_struct_316 6 Offset 24
               OpMemberDecorate %_struct_316 7 Offset 28
               OpMemberDecorate %_struct_316 8 Offset 32
               OpMemberDecorate %_struct_316 9 Offset 36
               OpMemberDecorate %_struct_316 10 Offset 40
               OpMemberDecorate %_struct_316 11 Offset 44
               OpMemberDecorate %_struct_316 12 Offset 48
               OpMemberDecorate %_struct_316 13 Offset 52
               OpMemberDecorate %_struct_316 14 Offset 56
               OpMemberDecorate %_struct_316 15 Offset 60
               OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId
               OpDecorate %374 SpecId 10
               OpDecorate %382 SpecId 4
               OpDecorate %470 SpecId 5
               OpDecorate %473 SpecId 8
               OpDecorate %552 SpecId 0
               OpDecorate %_runtimearr_float ArrayStride 4
               OpDecorate %_struct_857 Block
               OpMemberDecorate %_struct_857 0 NonReadable
               OpMemberDecorate %_struct_857 0 Offset 0
               OpDecorate %859 NonReadable
               OpDecorate %859 Binding 2
               OpDecorate %859 DescriptorSet 0
               OpDecorate %886 SpecId 0
               OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
       %bool = OpTypeBool
     %uint_4 = OpConstant %uint 4
%_arr_uint_uint_4 = OpTypeArray %uint %uint_4
 %_struct_37 = OpTypeStruct %_arr_uint_uint_4 %uint %float
         %38 = OpSpecConstant %uint 64
         %39 = OpSpecConstantOp %uint IMul %38 %uint_4
%_arr__struct_37_39 = OpTypeArray %_struct_37 %39
%_ptr_Workgroup__arr__struct_37_39 = OpTypePointer Workgroup %_arr__struct_37_39
         %42 = OpVariable %_ptr_Workgroup__arr__struct_37_39 Workgroup
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
       %half = OpTypeFloat 16
     %ushort = OpTypeInt 16 0
     %uint_2 = OpConstant %uint 2
%_arr_ushort_uint_2 = OpTypeArray %ushort %uint_2
     %uint_8 = OpConstant %uint 8
%_arr_ushort_uint_8 = OpTypeArray %ushort %uint_8
 %_struct_51 = OpTypeStruct %half %_arr_ushort_uint_2 %_arr_ushort_uint_8
%_runtimearr__struct_51 = OpTypeRuntimeArray %_struct_51
 %_struct_53 = OpTypeStruct %_runtimearr__struct_51
%_ptr_StorageBuffer__struct_53 = OpTypePointer StorageBuffer %_struct_53
         %55 = OpVariable %_ptr_StorageBuffer__struct_53 StorageBuffer
      %int_2 = OpConstant %int 2
%_ptr_StorageBuffer_ushort = OpTypePointer StorageBuffer %ushort
     %uint_1 = OpConstant %uint 1
   %v2ushort = OpTypeVector %ushort 2
%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint
     %uint_0 = OpConstant %uint 0
%_ptr_StorageBuffer_half = OpTypePointer StorageBuffer %half
%_ptr_Workgroup_float = OpTypePointer Workgroup %float
      %int_1 = OpConstant %int 1
         %89 = OpSpecConstant %uint 2
         %90 = OpSpecConstant %uint 4
         %91 = OpSpecConstantOp %uint IMul %89 %90
%_arr__struct_37_91 = OpTypeArray %_struct_37 %91
%_ptr_Function_uint = OpTypePointer Function %uint
%uint_252645135 = OpConstant %uint 252645135
     %int_15 = OpConstant %int 15
%int_33818640 = OpConstant %int 33818640
%int_269488144 = OpConstant %int 269488144
      %int_4 = OpConstant %int 4
     %int_16 = OpConstant %int 16
%_arr_int_uint_8 = OpTypeArray %int %uint_8
    %v2float = OpTypeVector %float 2
%_struct_169 = OpTypeStruct %_arr_int_uint_8 %v2float
   %float_16 = OpConstant %float 16
        %217 = OpSpecConstant %uint 64
        %218 = OpSpecConstantOp %uint IMul %217 %uint_4
%_arr__struct_169_218 = OpTypeArray %_struct_169 %218
%_ptr_Workgroup__arr__struct_169_218 = OpTypePointer Workgroup %_arr__struct_169_218
        %221 = OpVariable %_ptr_Workgroup__arr__struct_169_218 Workgroup
     %v2half = OpTypeVector %half 2
%_arr_v2half_uint_4 = OpTypeArray %v2half %uint_4
      %v4int = OpTypeVector %int 4
%_arr_v4int_uint_8 = OpTypeArray %v4int %uint_8
%_struct_226 = OpTypeStruct %_arr_v2half_uint_4 %_arr_v4int_uint_8
%_runtimearr__struct_226 = OpTypeRuntimeArray %_struct_226
%_struct_228 = OpTypeStruct %_runtimearr__struct_226
%_ptr_StorageBuffer__struct_228 = OpTypePointer StorageBuffer %_struct_228
        %230 = OpVariable %_ptr_StorageBuffer__struct_228 StorageBuffer
%_ptr_StorageBuffer_v2half = OpTypePointer StorageBuffer %v2half
%_ptr_Workgroup_v2float = OpTypePointer Workgroup %v2float
%_ptr_StorageBuffer_v4int = OpTypePointer StorageBuffer %v4int
%_ptr_Workgroup_int = OpTypePointer Workgroup %int
     %uint_3 = OpConstant %uint 3
    %float_0 = OpConstant %float 0
        %274 = OpConstantComposite %v2float %float_0 %float_0
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_WorkGroupID = OpVariable %_ptr_Input_v3uint Input
%_ptr_Input_uint = OpTypePointer Input %uint
%_struct_316 = OpTypeStruct %uint %uint %uint %uint %uint %uint %uint %uint %uint %uint %uint %uint %uint %uint %uint %uint
%_ptr_PushConstant__struct_316 = OpTypePointer PushConstant %_struct_316
        %318 = OpVariable %_ptr_PushConstant__struct_316 PushConstant
      %int_9 = OpConstant %int 9
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
     %int_13 = OpConstant %int 13
     %int_14 = OpConstant %int 14
     %int_12 = OpConstant %int 12
%gl_LocalInvocationID = OpVariable %_ptr_Input_v3uint Input
        %374 = OpSpecConstant %uint 32
        %382 = OpSpecConstant %uint 32
        %383 = OpSpecConstantOp %uint UDiv %382 %89
        %384 = OpSpecConstantOp %uint UDiv %383 %90
        %388 = OpSpecConstantOp %uint UDiv %383 %90
        %392 = OpSpecConstantOp %uint UDiv %38 %382
        %396 = OpSpecConstantOp %uint UDiv %38 %382
     %int_11 = OpConstant %int 11
      %int_6 = OpConstant %int 6
    %uint_32 = OpConstant %uint 32
      %int_3 = OpConstant %int 3
      %int_7 = OpConstant %int 7
        %469 = OpSpecConstantOp %uint IMul %89 %90
        %470 = OpSpecConstant %uint 32
        %471 = OpSpecConstantOp %uint IMul %382 %470
        %472 = OpSpecConstantOp %uint IMul %374 %90
        %473 = OpSpecConstant %uint 2
        %474 = OpSpecConstantOp %uint IMul %472 %473
        %475 = OpSpecConstantOp %uint IMul %474 %89
        %476 = OpSpecConstantOp %uint UDiv %471 %475
        %477 = OpSpecConstantOp %uint IMul %469 %476
        %478 = OpSpecConstantOp %uint IMul %477 %473
        %480 = OpSpecConstantOp %uint IMul %89 %90
        %481 = OpSpecConstantOp %uint IMul %480 %476
        %482 = OpSpecConstantOp %uint IMul %481 %473
%_arr_float_482 = OpTypeArray %float %482
%_ptr_Function__arr_float_482 = OpTypePointer Function %_arr_float_482
%_ptr_Function_float = OpTypePointer Function %float
        %552 = OpSpecConstant %uint 64
        %553 = OpSpecConstantOp %uint IMul %552 %uint_8
        %554 = OpSpecConstantOp %uint UDiv %553 %uint_32
    %uint_16 = OpConstant %uint 16
        %607 = OpSpecConstantOp %uint IMul %552 %uint_16
        %608 = OpSpecConstantOp %uint UDiv %607 %uint_32
   %uint_264 = OpConstant %uint 264
        %689 = OpSpecConstantOp %uint UDiv %470 %476
        %725 = OpSpecConstantOp %uint IMul %89 %90
   %uint_128 = OpConstant %uint 128
      %int_8 = OpConstant %int 8
     %int_10 = OpConstant %int 10
%_runtimearr_float = OpTypeRuntimeArray %float
%_struct_857 = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer__struct_857 = OpTypePointer StorageBuffer %_struct_857
        %859 = OpVariable %_ptr_StorageBuffer__struct_857 StorageBuffer
      %int_5 = OpConstant %int 5
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
        %886 = OpSpecConstant %uint 1
%gl_WorkGroupSize = OpSpecConstantComposite %v3uint %886 %uint_1 %uint_1
%_ptr_Function__arr__struct_37_91 = OpTypePointer Function %_arr__struct_37_91
     %uint_5 = OpConstant %uint 5
     %uint_6 = OpConstant %uint 6
     %uint_7 = OpConstant %uint 7
    %uint_12 = OpConstant %uint 12
    %uint_64 = OpConstant %uint 64
    %uint_96 = OpConstant %uint 96
          %4 = OpFunction %void None %3
          %5 = OpLabel
         %94 = OpVariable %_ptr_Function__arr__struct_37_91 Function
        %485 = OpVariable %_ptr_Function__arr_float_482 Function
        %311 = OpAccessChain %_ptr_Input_uint %gl_WorkGroupID %uint_1
        %312 = OpLoad %uint %311
        %314 = OpAccessChain %_ptr_Input_uint %gl_WorkGroupID %uint_2
        %315 = OpLoad %uint %314
        %321 = OpAccessChain %_ptr_PushConstant_uint %318 %int_9
        %322 = OpLoad %uint %321
        %323 = OpIAdd %uint %315 %322
        %327 = OpAccessChain %_ptr_PushConstant_uint %318 %int_13
        %328 = OpLoad %uint %327
        %329 = OpUDiv %uint %323 %328
        %334 = OpUMod %uint %323 %328
        %337 = OpAccessChain %_ptr_PushConstant_uint %318 %int_15
        %338 = OpLoad %uint %337
        %339 = OpUDiv %uint %329 %338
        %343 = OpAccessChain %_ptr_PushConstant_uint %318 %int_14
        %344 = OpLoad %uint %343
        %345 = OpUDiv %uint %334 %344
        %349 = OpAccessChain %_ptr_PushConstant_uint %318 %int_12
        %350 = OpLoad %uint %349
        %351 = OpIMul %uint %339 %350
        %353 = OpIAdd %uint %351 %345
        %355 = OpAccessChain %_ptr_PushConstant_uint %318 %int_0
        %356 = OpLoad %uint %355
        %357 = OpIAdd %uint %356 %38
        %358 = OpISub %uint %357 %uint_1
        %359 = OpUDiv %uint %358 %38
        %361 = OpAccessChain %_ptr_Input_uint %gl_WorkGroupID %uint_0
        %362 = OpLoad %uint %361
        %364 = OpUMod %uint %362 %359
        %369 = OpUDiv %uint %362 %359
        %372 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
        %373 = OpLoad %uint %372
        %375 = OpUDiv %uint %373 %374
        %379 = OpUMod %uint %373 %374
        %385 = OpUMod %uint %379 %384
        %389 = OpUDiv %uint %379 %388
        %393 = OpUMod %uint %375 %392
        %397 = OpUDiv %uint %375 %396
        %401 = OpUMod %uint %373 %uint_4
        %405 = OpUDiv %uint %373 %uint_4
        %409 = OpUMod %uint %373 %uint_2
        %413 = OpUDiv %uint %373 %uint_2
        %417 = OpAccessChain %_ptr_PushConstant_uint %318 %int_11
        %418 = OpLoad %uint %417
        %419 = OpIMul %uint %369 %418
        %421 = OpAccessChain %_ptr_PushConstant_uint %318 %int_2
        %422 = OpLoad %uint %421
        %424 = OpIAdd %uint %369 %uint_1
        %427 = OpIMul %uint %424 %418
        %428 = OpExtInst %uint %1 UMin %422 %427
        %432 = OpAccessChain %_ptr_PushConstant_uint %318 %int_6
        %433 = OpLoad %uint %432
        %435 = OpUDiv %uint %433 %uint_32
        %436 = OpIMul %uint %353 %435
        %438 = OpIMul %uint %364 %38
        %440 = OpAccessChain %_ptr_PushConstant_uint %318 %int_3
        %441 = OpLoad %uint %440
        %442 = OpIMul %uint %438 %441
        %444 = OpIAdd %uint %442 %419
        %445 = OpUDiv %uint %444 %uint_32
        %446 = OpIAdd %uint %436 %445
        %450 = OpAccessChain %_ptr_PushConstant_uint %318 %int_7
        %451 = OpLoad %uint %450
        %452 = OpIMul %uint %323 %451
        %454 = OpIMul %uint %312 %217
        %455 = OpAccessChain %_ptr_PushConstant_uint %318 %int_4
        %456 = OpLoad %uint %455
        %457 = OpIMul %uint %454 %456
        %458 = OpIAdd %uint %452 %457
        %460 = OpIAdd %uint %458 %419
        %461 = OpUDiv %uint %460 %uint_32
               OpBranch %463
        %463 = OpLabel
       %1113 = OpPhi %uint %uint_0 %5 %490 %464
        %479 = OpULessThan %bool %1113 %478
               OpLoopMerge %465 %464 Unroll
               OpBranchConditional %479 %464 %465
        %464 = OpLabel
        %488 = OpAccessChain %_ptr_Function_float %485 %1113
               OpStore %488 %float_0
        %490 = OpIAdd %uint %1113 %int_1
               OpBranch %463
        %465 = OpLabel
               OpBranch %493
        %493 = OpLabel
       %1136 = OpPhi %uint %461 %465 %615 %496
       %1132 = OpPhi %uint %446 %465 %613 %496
       %1114 = OpPhi %uint %419 %465 %751 %496
        %500 = OpULessThan %bool %1114 %428
               OpLoopMerge %495 %496 None
               OpBranchConditional %500 %494 %495
        %494 = OpLabel
               OpBranch %502
        %502 = OpLabel
       %1128 = OpPhi %uint %uint_0 %494 %556 %505
        %509 = OpIAdd %uint %405 %1128
        %510 = OpULessThan %bool %509 %38
               OpLoopMerge %504 %505 Unroll
               OpBranchConditional %510 %503 %504
        %503 = OpLabel
        %520 = OpIMul %uint %509 %441
        %521 = OpUDiv %uint %520 %uint_32
        %522 = OpIAdd %uint %1132 %521
               OpSelectionMerge %540 None
               OpBranchConditional %500 %539 %540
        %539 = OpLabel
        %898 = OpIMul %uint %401 %uint_2
        %899 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %522 %int_2 %898
        %900 = OpLoad %ushort %899
        %902 = OpIAdd %uint %898 %uint_1
        %903 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %522 %int_2 %902
        %904 = OpLoad %ushort %903
        %905 = OpCompositeConstruct %v2ushort %900 %904
        %906 = OpBitcast %uint %905
        %907 = OpAccessChain %_ptr_Workgroup_uint %42 %509 %int_0 %401
               OpStore %907 %906
        %908 = OpIEqual %bool %401 %uint_0
               OpSelectionMerge %921 None
               OpBranchConditional %908 %909 %921
        %909 = OpLabel
        %910 = OpAccessChain %_ptr_StorageBuffer_half %55 %int_0 %522 %int_0
        %911 = OpLoad %half %910
        %912 = OpFConvert %float %911
        %913 = OpAccessChain %_ptr_Workgroup_float %42 %509 %int_2
               OpStore %913 %912
        %914 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %522 %int_1 %int_0
        %915 = OpLoad %ushort %914
        %916 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %522 %int_1 %int_1
        %917 = OpLoad %ushort %916
        %918 = OpCompositeConstruct %v2ushort %915 %917
        %919 = OpBitcast %uint %918
        %920 = OpAccessChain %_ptr_Workgroup_uint %42 %509 %int_1
               OpStore %920 %919
               OpBranch %921
        %921 = OpLabel
               OpBranch %540
        %540 = OpLabel
       %1252 = OpIAdd %uint %1114 %uint_32
       %1253 = OpULessThan %bool %1252 %428
               OpSelectionMerge %1282 None
               OpBranchConditional %1253 %1254 %1282
       %1254 = OpLabel
       %1256 = OpIAdd %uint %38 %509
       %1257 = OpIAdd %uint %522 %uint_1
       %1258 = OpIMul %uint %401 %uint_2
       %1259 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %1257 %int_2 %1258
       %1260 = OpLoad %ushort %1259
       %1262 = OpIAdd %uint %1258 %uint_1
       %1263 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %1257 %int_2 %1262
       %1264 = OpLoad %ushort %1263
       %1265 = OpCompositeConstruct %v2ushort %1260 %1264
       %1266 = OpBitcast %uint %1265
       %1267 = OpAccessChain %_ptr_Workgroup_uint %42 %1256 %int_0 %401
               OpStore %1267 %1266
       %1268 = OpIEqual %bool %401 %uint_0
               OpSelectionMerge %1281 None
               OpBranchConditional %1268 %1269 %1281
       %1269 = OpLabel
       %1270 = OpAccessChain %_ptr_StorageBuffer_half %55 %int_0 %1257 %int_0
       %1271 = OpLoad %half %1270
       %1272 = OpFConvert %float %1271
       %1273 = OpAccessChain %_ptr_Workgroup_float %42 %1256 %int_2
               OpStore %1273 %1272
       %1274 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %1257 %int_1 %int_0
       %1275 = OpLoad %ushort %1274
       %1276 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %1257 %int_1 %int_1
       %1277 = OpLoad %ushort %1276
       %1278 = OpCompositeConstruct %v2ushort %1275 %1277
       %1279 = OpBitcast %uint %1278
       %1280 = OpAccessChain %_ptr_Workgroup_uint %42 %1256 %int_1
               OpStore %1280 %1279
               OpBranch %1281
       %1281 = OpLabel
               OpBranch %1282
       %1282 = OpLabel
       %1291 = OpIAdd %uint %1114 %uint_64
       %1292 = OpULessThan %bool %1291 %428
               OpSelectionMerge %1321 None
               OpBranchConditional %1292 %1293 %1321
       %1293 = OpLabel
       %1294 = OpIMul %uint %uint_2 %38
       %1295 = OpIAdd %uint %1294 %509
       %1296 = OpIAdd %uint %522 %uint_2
       %1297 = OpIMul %uint %401 %uint_2
       %1298 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %1296 %int_2 %1297
       %1299 = OpLoad %ushort %1298
       %1301 = OpIAdd %uint %1297 %uint_1
       %1302 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %1296 %int_2 %1301
       %1303 = OpLoad %ushort %1302
       %1304 = OpCompositeConstruct %v2ushort %1299 %1303
       %1305 = OpBitcast %uint %1304
       %1306 = OpAccessChain %_ptr_Workgroup_uint %42 %1295 %int_0 %401
               OpStore %1306 %1305
       %1307 = OpIEqual %bool %401 %uint_0
               OpSelectionMerge %1320 None
               OpBranchConditional %1307 %1308 %1320
       %1308 = OpLabel
       %1309 = OpAccessChain %_ptr_StorageBuffer_half %55 %int_0 %1296 %int_0
       %1310 = OpLoad %half %1309
       %1311 = OpFConvert %float %1310
       %1312 = OpAccessChain %_ptr_Workgroup_float %42 %1295 %int_2
               OpStore %1312 %1311
       %1313 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %1296 %int_1 %int_0
       %1314 = OpLoad %ushort %1313
       %1315 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %1296 %int_1 %int_1
       %1316 = OpLoad %ushort %1315
       %1317 = OpCompositeConstruct %v2ushort %1314 %1316
       %1318 = OpBitcast %uint %1317
       %1319 = OpAccessChain %_ptr_Workgroup_uint %42 %1295 %int_1
               OpStore %1319 %1318
               OpBranch %1320
       %1320 = OpLabel
               OpBranch %1321
       %1321 = OpLabel
       %1330 = OpIAdd %uint %1114 %uint_96
       %1331 = OpULessThan %bool %1330 %428
               OpSelectionMerge %1360 None
               OpBranchConditional %1331 %1332 %1360
       %1332 = OpLabel
       %1333 = OpIMul %uint %uint_3 %38
       %1334 = OpIAdd %uint %1333 %509
       %1335 = OpIAdd %uint %522 %uint_3
       %1336 = OpIMul %uint %401 %uint_2
       %1337 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %1335 %int_2 %1336
       %1338 = OpLoad %ushort %1337
       %1340 = OpIAdd %uint %1336 %uint_1
       %1341 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %1335 %int_2 %1340
       %1342 = OpLoad %ushort %1341
       %1343 = OpCompositeConstruct %v2ushort %1338 %1342
       %1344 = OpBitcast %uint %1343
       %1345 = OpAccessChain %_ptr_Workgroup_uint %42 %1334 %int_0 %401
               OpStore %1345 %1344
       %1346 = OpIEqual %bool %401 %uint_0
               OpSelectionMerge %1359 None
               OpBranchConditional %1346 %1347 %1359
       %1347 = OpLabel
       %1348 = OpAccessChain %_ptr_StorageBuffer_half %55 %int_0 %1335 %int_0
       %1349 = OpLoad %half %1348
       %1350 = OpFConvert %float %1349
       %1351 = OpAccessChain %_ptr_Workgroup_float %42 %1334 %int_2
               OpStore %1351 %1350
       %1352 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %1335 %int_1 %int_0
       %1353 = OpLoad %ushort %1352
       %1354 = OpAccessChain %_ptr_StorageBuffer_ushort %55 %int_0 %1335 %int_1 %int_1
       %1355 = OpLoad %ushort %1354
       %1356 = OpCompositeConstruct %v2ushort %1353 %1355
       %1357 = OpBitcast %uint %1356
       %1358 = OpAccessChain %_ptr_Workgroup_uint %42 %1334 %int_1
               OpStore %1358 %1357
               OpBranch %1359
       %1359 = OpLabel
               OpBranch %1360
       %1360 = OpLabel
               OpBranch %505
        %505 = OpLabel
        %556 = OpIAdd %uint %1128 %554
               OpBranch %502
        %504 = OpLabel
               OpBranch %558
        %558 = OpLabel
       %1129 = OpPhi %uint %uint_0 %504 %610 %561
        %565 = OpIAdd %uint %413 %1129
        %566 = OpULessThan %bool %565 %217
               OpLoopMerge %560 %561 Unroll
               OpBranchConditional %566 %559 %560
        %559 = OpLabel
        %576 = OpIMul %uint %565 %456
        %577 = OpUDiv %uint %576 %uint_32
        %578 = OpIAdd %uint %1136 %577
               OpSelectionMerge %979 None
               OpBranchConditional %500 %926 %963
        %926 = OpLabel
        %927 = OpUDiv %uint %578 %uint_4
        %928 = OpUMod %uint %578 %uint_4
        %929 = OpIEqual %bool %409 %uint_0
               OpSelectionMerge %937 None
               OpBranchConditional %929 %930 %937
        %930 = OpLabel
        %933 = OpAccessChain %_ptr_StorageBuffer_v2half %230 %int_0 %927 %int_0 %928
        %934 = OpLoad %v2half %933
        %935 = OpFConvert %v2float %934
        %936 = OpAccessChain %_ptr_Workgroup_v2float %221 %565 %int_1
               OpStore %936 %935
               OpBranch %937
        %937 = OpLabel
        %940 = OpIMul %uint %928 %uint_2
        %941 = OpIAdd %uint %940 %409
        %942 = OpAccessChain %_ptr_StorageBuffer_v4int %230 %int_0 %927 %int_1 %941
        %943 = OpLoad %v4int %942
        %944 = OpIMul %uint %409 %uint_4
        %946 = OpCompositeExtract %int %943 0
        %947 = OpAccessChain %_ptr_Workgroup_int %221 %565 %int_0 %944
               OpStore %947 %946
        %949 = OpIAdd %uint %944 %uint_1
        %951 = OpCompositeExtract %int %943 1
        %952 = OpAccessChain %_ptr_Workgroup_int %221 %565 %int_0 %949
               OpStore %952 %951
        %954 = OpIAdd %uint %944 %uint_2
        %956 = OpCompositeExtract %int %943 2
        %957 = OpAccessChain %_ptr_Workgroup_int %221 %565 %int_0 %954
               OpStore %957 %956
        %959 = OpIAdd %uint %944 %uint_3
        %961 = OpCompositeExtract %int %943 3
        %962 = OpAccessChain %_ptr_Workgroup_int %221 %565 %int_0 %959
               OpStore %962 %961
               OpBranch %979
        %963 = OpLabel
        %964 = OpIEqual %bool %409 %uint_0
               OpSelectionMerge %967 None
               OpBranchConditional %964 %965 %967
        %965 = OpLabel
        %966 = OpAccessChain %_ptr_Workgroup_v2float %221 %565 %int_1
               OpStore %966 %274
               OpBranch %967
        %967 = OpLabel
        %968 = OpIMul %uint %409 %uint_4
        %969 = OpAccessChain %_ptr_Workgroup_int %221 %565 %int_0 %968
               OpStore %969 %int_0
        %971 = OpIAdd %uint %968 %uint_1
        %972 = OpAccessChain %_ptr_Workgroup_int %221 %565 %int_0 %971
               OpStore %972 %int_0
        %974 = OpIAdd %uint %968 %uint_2
        %975 = OpAccessChain %_ptr_Workgroup_int %221 %565 %int_0 %974
               OpStore %975 %int_0
        %977 = OpIAdd %uint %968 %uint_3
        %978 = OpAccessChain %_ptr_Workgroup_int %221 %565 %int_0 %977
               OpStore %978 %int_0
               OpBranch %979
        %979 = OpLabel
       %1369 = OpIAdd %uint %217 %565
       %1370 = OpIAdd %uint %578 %uint_1
       %1372 = OpIAdd %uint %1114 %uint_32
       %1373 = OpULessThan %bool %1372 %428
               OpSelectionMerge %1419 None
               OpBranchConditional %1373 %1390 %1374
       %1374 = OpLabel
       %1375 = OpIEqual %bool %409 %uint_0
               OpSelectionMerge %1378 None
               OpBranchConditional %1375 %1376 %1378
       %1376 = OpLabel
       %1377 = OpAccessChain %_ptr_Workgroup_v2float %221 %1369 %int_1
               OpStore %1377 %274
               OpBranch %1378
       %1378 = OpLabel
       %1379 = OpIMul %uint %409 %uint_4
       %1380 = OpAccessChain %_ptr_Workgroup_int %221 %1369 %int_0 %1379
               OpStore %1380 %int_0
       %1382 = OpIAdd %uint %1379 %uint_1
       %1383 = OpAccessChain %_ptr_Workgroup_int %221 %1369 %int_0 %1382
               OpStore %1383 %int_0
       %1385 = OpIAdd %uint %1379 %uint_2
       %1386 = OpAccessChain %_ptr_Workgroup_int %221 %1369 %int_0 %1385
               OpStore %1386 %int_0
       %1388 = OpIAdd %uint %1379 %uint_3
       %1389 = OpAccessChain %_ptr_Workgroup_int %221 %1369 %int_0 %1388
               OpStore %1389 %int_0
               OpBranch %1419
       %1390 = OpLabel
       %1391 = OpUDiv %uint %1370 %uint_4
       %1392 = OpUMod %uint %1370 %uint_4
       %1393 = OpIEqual %bool %409 %uint_0
               OpSelectionMerge %1399 None
               OpBranchConditional %1393 %1394 %1399
       %1394 = OpLabel
       %1395 = OpAccessChain %_ptr_StorageBuffer_v2half %230 %int_0 %1391 %int_0 %1392
       %1396 = OpLoad %v2half %1395
       %1397 = OpFConvert %v2float %1396
       %1398 = OpAccessChain %_ptr_Workgroup_v2float %221 %1369 %int_1
               OpStore %1398 %1397
               OpBranch %1399
       %1399 = OpLabel
       %1400 = OpIMul %uint %1392 %uint_2
       %1401 = OpIAdd %uint %1400 %409
       %1402 = OpAccessChain %_ptr_StorageBuffer_v4int %230 %int_0 %1391 %int_1 %1401
       %1403 = OpLoad %v4int %1402
       %1404 = OpIMul %uint %409 %uint_4
       %1405 = OpCompositeExtract %int %1403 0
       %1406 = OpAccessChain %_ptr_Workgroup_int %221 %1369 %int_0 %1404
               OpStore %1406 %1405
       %1408 = OpIAdd %uint %1404 %uint_1
       %1409 = OpCompositeExtract %int %1403 1
       %1410 = OpAccessChain %_ptr_Workgroup_int %221 %1369 %int_0 %1408
               OpStore %1410 %1409
       %1412 = OpIAdd %uint %1404 %uint_2
       %1413 = OpCompositeExtract %int %1403 2
       %1414 = OpAccessChain %_ptr_Workgroup_int %221 %1369 %int_0 %1412
               OpStore %1414 %1413
       %1416 = OpIAdd %uint %1404 %uint_3
       %1417 = OpCompositeExtract %int %1403 3
       %1418 = OpAccessChain %_ptr_Workgroup_int %221 %1369 %int_0 %1416
               OpStore %1418 %1417
               OpBranch %1419
       %1419 = OpLabel
       %1427 = OpIMul %uint %uint_2 %217
       %1428 = OpIAdd %uint %1427 %565
       %1429 = OpIAdd %uint %578 %uint_2
       %1431 = OpIAdd %uint %1114 %uint_64
       %1432 = OpULessThan %bool %1431 %428
               OpSelectionMerge %1478 None
               OpBranchConditional %1432 %1449 %1433
       %1433 = OpLabel
       %1434 = OpIEqual %bool %409 %uint_0
               OpSelectionMerge %1437 None
               OpBranchConditional %1434 %1435 %1437
       %1435 = OpLabel
       %1436 = OpAccessChain %_ptr_Workgroup_v2float %221 %1428 %int_1
               OpStore %1436 %274
               OpBranch %1437
       %1437 = OpLabel
       %1438 = OpIMul %uint %409 %uint_4
       %1439 = OpAccessChain %_ptr_Workgroup_int %221 %1428 %int_0 %1438
               OpStore %1439 %int_0
       %1441 = OpIAdd %uint %1438 %uint_1
       %1442 = OpAccessChain %_ptr_Workgroup_int %221 %1428 %int_0 %1441
               OpStore %1442 %int_0
       %1444 = OpIAdd %uint %1438 %uint_2
       %1445 = OpAccessChain %_ptr_Workgroup_int %221 %1428 %int_0 %1444
               OpStore %1445 %int_0
       %1447 = OpIAdd %uint %1438 %uint_3
       %1448 = OpAccessChain %_ptr_Workgroup_int %221 %1428 %int_0 %1447
               OpStore %1448 %int_0
               OpBranch %1478
       %1449 = OpLabel
       %1450 = OpUDiv %uint %1429 %uint_4
       %1451 = OpUMod %uint %1429 %uint_4
       %1452 = OpIEqual %bool %409 %uint_0
               OpSelectionMerge %1458 None
               OpBranchConditional %1452 %1453 %1458
       %1453 = OpLabel
       %1454 = OpAccessChain %_ptr_StorageBuffer_v2half %230 %int_0 %1450 %int_0 %1451
       %1455 = OpLoad %v2half %1454
       %1456 = OpFConvert %v2float %1455
       %1457 = OpAccessChain %_ptr_Workgroup_v2float %221 %1428 %int_1
               OpStore %1457 %1456
               OpBranch %1458
       %1458 = OpLabel
       %1459 = OpIMul %uint %1451 %uint_2
       %1460 = OpIAdd %uint %1459 %409
       %1461 = OpAccessChain %_ptr_StorageBuffer_v4int %230 %int_0 %1450 %int_1 %1460
       %1462 = OpLoad %v4int %1461
       %1463 = OpIMul %uint %409 %uint_4
       %1464 = OpCompositeExtract %int %1462 0
       %1465 = OpAccessChain %_ptr_Workgroup_int %221 %1428 %int_0 %1463
               OpStore %1465 %1464
       %1467 = OpIAdd %uint %1463 %uint_1
       %1468 = OpCompositeExtract %int %1462 1
       %1469 = OpAccessChain %_ptr_Workgroup_int %221 %1428 %int_0 %1467
               OpStore %1469 %1468
       %1471 = OpIAdd %uint %1463 %uint_2
       %1472 = OpCompositeExtract %int %1462 2
       %1473 = OpAccessChain %_ptr_Workgroup_int %221 %1428 %int_0 %1471
               OpStore %1473 %1472
       %1475 = OpIAdd %uint %1463 %uint_3
       %1476 = OpCompositeExtract %int %1462 3
       %1477 = OpAccessChain %_ptr_Workgroup_int %221 %1428 %int_0 %1475
               OpStore %1477 %1476
               OpBranch %1478
       %1478 = OpLabel
       %1486 = OpIMul %uint %uint_3 %217
       %1487 = OpIAdd %uint %1486 %565
       %1488 = OpIAdd %uint %578 %uint_3
       %1490 = OpIAdd %uint %1114 %uint_96
       %1491 = OpULessThan %bool %1490 %428
               OpSelectionMerge %1537 None
               OpBranchConditional %1491 %1508 %1492
       %1492 = OpLabel
       %1493 = OpIEqual %bool %409 %uint_0
               OpSelectionMerge %1496 None
               OpBranchConditional %1493 %1494 %1496
       %1494 = OpLabel
       %1495 = OpAccessChain %_ptr_Workgroup_v2float %221 %1487 %int_1
               OpStore %1495 %274
               OpBranch %1496
       %1496 = OpLabel
       %1497 = OpIMul %uint %409 %uint_4
       %1498 = OpAccessChain %_ptr_Workgroup_int %221 %1487 %int_0 %1497
               OpStore %1498 %int_0
       %1500 = OpIAdd %uint %1497 %uint_1
       %1501 = OpAccessChain %_ptr_Workgroup_int %221 %1487 %int_0 %1500
               OpStore %1501 %int_0
       %1503 = OpIAdd %uint %1497 %uint_2
       %1504 = OpAccessChain %_ptr_Workgroup_int %221 %1487 %int_0 %1503
               OpStore %1504 %int_0
       %1506 = OpIAdd %uint %1497 %uint_3
       %1507 = OpAccessChain %_ptr_Workgroup_int %221 %1487 %int_0 %1506
               OpStore %1507 %int_0
               OpBranch %1537
       %1508 = OpLabel
       %1509 = OpUDiv %uint %1488 %uint_4
       %1510 = OpUMod %uint %1488 %uint_4
       %1511 = OpIEqual %bool %409 %uint_0
               OpSelectionMerge %1517 None
               OpBranchConditional %1511 %1512 %1517
       %1512 = OpLabel
       %1513 = OpAccessChain %_ptr_StorageBuffer_v2half %230 %int_0 %1509 %int_0 %1510
       %1514 = OpLoad %v2half %1513
       %1515 = OpFConvert %v2float %1514
       %1516 = OpAccessChain %_ptr_Workgroup_v2float %221 %1487 %int_1
               OpStore %1516 %1515
               OpBranch %1517
       %1517 = OpLabel
       %1518 = OpIMul %uint %1510 %uint_2
       %1519 = OpIAdd %uint %1518 %409
       %1520 = OpAccessChain %_ptr_StorageBuffer_v4int %230 %int_0 %1509 %int_1 %1519
       %1521 = OpLoad %v4int %1520
       %1522 = OpIMul %uint %409 %uint_4
       %1523 = OpCompositeExtract %int %1521 0
       %1524 = OpAccessChain %_ptr_Workgroup_int %221 %1487 %int_0 %1522
               OpStore %1524 %1523
       %1526 = OpIAdd %uint %1522 %uint_1
       %1527 = OpCompositeExtract %int %1521 1
       %1528 = OpAccessChain %_ptr_Workgroup_int %221 %1487 %int_0 %1526
               OpStore %1528 %1527
       %1530 = OpIAdd %uint %1522 %uint_2
       %1531 = OpCompositeExtract %int %1521 2
       %1532 = OpAccessChain %_ptr_Workgroup_int %221 %1487 %int_0 %1530
               OpStore %1532 %1531
       %1534 = OpIAdd %uint %1522 %uint_3
       %1535 = OpCompositeExtract %int %1521 3
       %1536 = OpAccessChain %_ptr_Workgroup_int %221 %1487 %int_0 %1534
               OpStore %1536 %1535
               OpBranch %1537
       %1537 = OpLabel
               OpBranch %561
        %561 = OpLabel
        %610 = OpIAdd %uint %1129 %608
               OpBranch %558
        %560 = OpLabel
               OpControlBarrier %uint_2 %uint_2 %uint_264
        %613 = OpIAdd %uint %1132 %uint_4
        %615 = OpIAdd %uint %1136 %uint_4
               OpBranch %617
        %617 = OpLabel
       %1138 = OpPhi %uint %uint_0 %560 %748 %620
        %623 = OpULessThan %bool %1138 %uint_4
               OpLoopMerge %619 %620 None
               OpBranchConditional %623 %618 %619
        %618 = OpLabel
               OpBranch %625
        %625 = OpLabel
       %1142 = OpPhi %uint %uint_0 %618 %665 %628
        %631 = OpULessThan %bool %1142 %89
               OpLoopMerge %627 %628 Unroll
               OpBranchConditional %631 %626 %627
        %626 = OpLabel
               OpBranch %633
        %633 = OpLabel
       %1162 = OpPhi %uint %uint_0 %626 %663 %634
        %639 = OpULessThan %bool %1162 %90
               OpLoopMerge %635 %634 Unroll
               OpBranchConditional %639 %634 %635
        %634 = OpLabel
        %642 = OpIMul %uint %1142 %90
        %644 = OpIAdd %uint %642 %1162
        %647 = OpIMul %uint %393 %382
        %649 = OpIMul %uint %1142 %383
        %650 = OpIAdd %uint %647 %649
        %652 = OpIMul %uint %385 %90
        %653 = OpIAdd %uint %650 %652
        %655 = OpIAdd %uint %653 %1162
        %658 = OpIMul %uint %1138 %38
        %660 = OpIAdd %uint %658 %655
        %982 = OpAccessChain %_ptr_Workgroup_float %42 %660 %int_2
        %983 = OpLoad %float %982
        %984 = OpAccessChain %_ptr_Function_float %94 %644 %int_2
               OpStore %984 %983
        %985 = OpAccessChain %_ptr_Workgroup_uint %42 %660 %int_1
        %986 = OpLoad %uint %985
        %987 = OpAccessChain %_ptr_Function_uint %94 %644 %int_1
               OpStore %987 %986
        %995 = OpAccessChain %_ptr_Workgroup_uint %42 %660 %int_0 %uint_0
        %996 = OpLoad %uint %995
        %997 = OpAccessChain %_ptr_Function_uint %94 %644 %int_0 %uint_0
               OpStore %997 %996
       %1545 = OpAccessChain %_ptr_Workgroup_uint %42 %660 %int_0 %uint_1
       %1546 = OpLoad %uint %1545
       %1547 = OpAccessChain %_ptr_Function_uint %94 %644 %int_0 %uint_1
               OpStore %1547 %1546
       %1555 = OpAccessChain %_ptr_Workgroup_uint %42 %660 %int_0 %uint_2
       %1556 = OpLoad %uint %1555
       %1557 = OpAccessChain %_ptr_Function_uint %94 %644 %int_0 %uint_2
               OpStore %1557 %1556
       %1565 = OpAccessChain %_ptr_Workgroup_uint %42 %660 %int_0 %uint_3
       %1566 = OpLoad %uint %1565
       %1567 = OpAccessChain %_ptr_Function_uint %94 %644 %int_0 %uint_3
               OpStore %1567 %1566
        %663 = OpIAdd %uint %1162 %int_1
               OpBranch %633
        %635 = OpLabel
               OpBranch %628
        %628 = OpLabel
        %665 = OpIAdd %uint %1142 %int_1
               OpBranch %625
        %627 = OpLabel
               OpBranch %667
        %667 = OpLabel
       %1143 = OpPhi %uint %uint_0 %627 %746 %670
        %673 = OpULessThan %bool %1143 %476
               OpLoopMerge %669 %670 Unroll
               OpBranchConditional %673 %668 %669
        %668 = OpLabel
               OpBranch %675
        %675 = OpLabel
       %1146 = OpPhi %uint %uint_0 %668 %744 %678
        %681 = OpULessThan %bool %1146 %473
               OpLoopMerge %677 %678 Unroll
               OpBranchConditional %681 %676 %677
        %676 = OpLabel
        %684 = OpIMul %uint %1138 %217
        %686 = OpIMul %uint %397 %470
        %687 = OpIAdd %uint %684 %686
        %690 = OpIMul %uint %1143 %689
        %691 = OpIAdd %uint %687 %690
        %693 = OpIMul %uint %389 %473
        %694 = OpIAdd %uint %691 %693
        %696 = OpIAdd %uint %694 %1146
       %1004 = OpAccessChain %_ptr_Workgroup_v2float %221 %696 %int_1
       %1005 = OpLoad %v2float %1004
       %1014 = OpAccessChain %_ptr_Workgroup_int %221 %696 %int_0 %uint_0
       %1015 = OpLoad %int %1014
       %1575 = OpAccessChain %_ptr_Workgroup_int %221 %696 %int_0 %uint_1
       %1576 = OpLoad %int %1575
       %1585 = OpAccessChain %_ptr_Workgroup_int %221 %696 %int_0 %uint_2
       %1586 = OpLoad %int %1585
       %1595 = OpAccessChain %_ptr_Workgroup_int %221 %696 %int_0 %uint_3
       %1596 = OpLoad %int %1595
       %1605 = OpAccessChain %_ptr_Workgroup_int %221 %696 %int_0 %uint_4
       %1606 = OpLoad %int %1605
       %1615 = OpAccessChain %_ptr_Workgroup_int %221 %696 %int_0 %uint_5
       %1616 = OpLoad %int %1615
       %1625 = OpAccessChain %_ptr_Workgroup_int %221 %696 %int_0 %uint_6
       %1626 = OpLoad %int %1625
       %1635 = OpAccessChain %_ptr_Workgroup_int %221 %696 %int_0 %uint_7
       %1636 = OpLoad %int %1635
               OpBranch %700
        %700 = OpLabel
       %1150 = OpPhi %uint %uint_0 %676 %742 %703
        %706 = OpULessThan %bool %1150 %89
               OpLoopMerge %702 %703 Unroll
               OpBranchConditional %706 %701 %702
        %701 = OpLabel
               OpBranch %708
        %708 = OpLabel
       %1153 = OpPhi %uint %uint_0 %701 %740 %709
        %714 = OpULessThan %bool %1153 %90
               OpLoopMerge %710 %709 Unroll
               OpBranchConditional %714 %709 %710
        %709 = OpLabel
        %717 = OpIMul %uint %1150 %90
        %719 = OpIAdd %uint %717 %1153
        %722 = OpIMul %uint %1143 %473
        %724 = OpIAdd %uint %722 %1146
        %726 = OpIMul %uint %724 %725
        %729 = OpIAdd %uint %726 %717
        %731 = OpIAdd %uint %729 %1153
       %1037 = OpAccessChain %_ptr_Function_uint %94 %719 %int_0 %uint_0
       %1038 = OpLoad %uint %1037
       %1039 = OpAccessChain %_ptr_Function_uint %94 %719 %int_1
       %1040 = OpLoad %uint %1039
       %1044 = OpBitcast %int %1040
       %1046 = OpBitwiseAnd %uint %1038 %uint_252645135
       %1047 = OpBitcast %int %1046
       %1049 = OpBitwiseAnd %int %1044 %int_15
       %1050 = OpIMul %int %1049 %int_33818640
       %1051 = OpBitwiseAnd %int %1050 %int_269488144
       %1052 = OpBitwiseOr %int %1047 %1051
       %1054 = OpShiftRightLogical %uint %1038 %int_4
       %1055 = OpBitwiseAnd %uint %1054 %uint_252645135
       %1056 = OpBitcast %int %1055
       %1058 = OpShiftRightArithmetic %int %1044 %int_16
       %1059 = OpBitwiseAnd %int %1058 %int_15
       %1060 = OpIMul %int %1059 %int_33818640
       %1061 = OpBitwiseAnd %int %1060 %int_269488144
       %1062 = OpBitwiseOr %int %1056 %1061
       %1072 = OpSDot %int %1052 %1015 PackedVectorFormat4x8Bit
       %1077 = OpSDot %int %1062 %1606 PackedVectorFormat4x8Bit
       %1079 = OpIAdd %int %1072 %1077
       %1646 = OpAccessChain %_ptr_Function_uint %94 %719 %int_0 %uint_1
       %1647 = OpLoad %uint %1646
       %1649 = OpLoad %uint %1039
       %1651 = OpShiftRightLogical %uint %1649 %uint_4
       %1652 = OpBitcast %int %1651
       %1653 = OpBitwiseAnd %uint %1647 %uint_252645135
       %1654 = OpBitcast %int %1653
       %1655 = OpBitwiseAnd %int %1652 %int_15
       %1656 = OpIMul %int %1655 %int_33818640
       %1657 = OpBitwiseAnd %int %1656 %int_269488144
       %1658 = OpBitwiseOr %int %1654 %1657
       %1659 = OpShiftRightLogical %uint %1647 %int_4
       %1660 = OpBitwiseAnd %uint %1659 %uint_252645135
       %1661 = OpBitcast %int %1660
       %1662 = OpShiftRightArithmetic %int %1652 %int_16
       %1663 = OpBitwiseAnd %int %1662 %int_15
       %1664 = OpIMul %int %1663 %int_33818640
       %1665 = OpBitwiseAnd %int %1664 %int_269488144
       %1666 = OpBitwiseOr %int %1661 %1665
       %1672 = OpSDot %int %1658 %1576 PackedVectorFormat4x8Bit
       %1673 = OpIAdd %int %1079 %1672
       %1674 = OpSDot %int %1666 %1616 PackedVectorFormat4x8Bit
       %1675 = OpIAdd %int %1673 %1674
       %1684 = OpAccessChain %_ptr_Function_uint %94 %719 %int_0 %uint_2
       %1685 = OpLoad %uint %1684
       %1687 = OpLoad %uint %1039
       %1689 = OpShiftRightLogical %uint %1687 %uint_8
       %1690 = OpBitcast %int %1689
       %1691 = OpBitwiseAnd %uint %1685 %uint_252645135
       %1692 = OpBitcast %int %1691
       %1693 = OpBitwiseAnd %int %1690 %int_15
       %1694 = OpIMul %int %1693 %int_33818640
       %1695 = OpBitwiseAnd %int %1694 %int_269488144
       %1696 = OpBitwiseOr %int %1692 %1695
       %1697 = OpShiftRightLogical %uint %1685 %int_4
       %1698 = OpBitwiseAnd %uint %1697 %uint_252645135
       %1699 = OpBitcast %int %1698
       %1700 = OpShiftRightArithmetic %int %1690 %int_16
       %1701 = OpBitwiseAnd %int %1700 %int_15
       %1702 = OpIMul %int %1701 %int_33818640
       %1703 = OpBitwiseAnd %int %1702 %int_269488144
       %1704 = OpBitwiseOr %int %1699 %1703
       %1710 = OpSDot %int %1696 %1586 PackedVectorFormat4x8Bit
       %1711 = OpIAdd %int %1675 %1710
       %1712 = OpSDot %int %1704 %1626 PackedVectorFormat4x8Bit
       %1713 = OpIAdd %int %1711 %1712
       %1722 = OpAccessChain %_ptr_Function_uint %94 %719 %int_0 %uint_3
       %1723 = OpLoad %uint %1722
       %1725 = OpLoad %uint %1039
       %1727 = OpShiftRightLogical %uint %1725 %uint_12
       %1728 = OpBitcast %int %1727
       %1729 = OpBitwiseAnd %uint %1723 %uint_252645135
       %1730 = OpBitcast %int %1729
       %1731 = OpBitwiseAnd %int %1728 %int_15
       %1732 = OpIMul %int %1731 %int_33818640
       %1733 = OpBitwiseAnd %int %1732 %int_269488144
       %1734 = OpBitwiseOr %int %1730 %1733
       %1735 = OpShiftRightLogical %uint %1723 %int_4
       %1736 = OpBitwiseAnd %uint %1735 %uint_252645135
       %1737 = OpBitcast %int %1736
       %1738 = OpShiftRightArithmetic %int %1728 %int_16
       %1739 = OpBitwiseAnd %int %1738 %int_15
       %1740 = OpIMul %int %1739 %int_33818640
       %1741 = OpBitwiseAnd %int %1740 %int_269488144
       %1742 = OpBitwiseOr %int %1737 %1741
       %1748 = OpSDot %int %1734 %1596 PackedVectorFormat4x8Bit
       %1749 = OpIAdd %int %1713 %1748
       %1750 = OpSDot %int %1742 %1636 PackedVectorFormat4x8Bit
       %1751 = OpIAdd %int %1749 %1750
       %1084 = OpAccessChain %_ptr_Function_float %94 %719 %int_2
       %1085 = OpLoad %float %1084
       %1087 = OpConvertSToF %float %1751
       %1089 = OpCompositeExtract %float %1005 0
       %1090 = OpFMul %float %1087 %1089
       %1092 = OpCompositeExtract %float %1005 1
       %1093 = OpFMul %float %float_16 %1092
       %1094 = OpFSub %float %1090 %1093
       %1095 = OpFMul %float %1085 %1094
        %735 = OpAccessChain %_ptr_Function_float %485 %731
        %736 = OpLoad %float %735
        %737 = OpFAdd %float %736 %1095
               OpStore %735 %737
        %740 = OpIAdd %uint %1153 %int_1
               OpBranch %708
        %710 = OpLabel
               OpBranch %703
        %703 = OpLabel
        %742 = OpIAdd %uint %1150 %int_1
               OpBranch %700
        %702 = OpLabel
               OpBranch %678
        %678 = OpLabel
        %744 = OpIAdd %uint %1146 %int_1
               OpBranch %675
        %677 = OpLabel
               OpBranch %670
        %670 = OpLabel
        %746 = OpIAdd %uint %1143 %int_1
               OpBranch %667
        %669 = OpLabel
               OpBranch %620
        %620 = OpLabel
        %748 = OpIAdd %uint %1138 %int_1
               OpBranch %617
        %619 = OpLabel
               OpControlBarrier %uint_2 %uint_2 %uint_264
               OpBranch %496
        %496 = OpLabel
        %751 = OpIAdd %uint %1114 %uint_128
               OpBranch %493
        %495 = OpLabel
        %756 = OpIMul %uint %393 %382
        %757 = OpIAdd %uint %438 %756
        %762 = OpIMul %uint %397 %470
        %763 = OpIAdd %uint %454 %762
        %767 = OpAccessChain %_ptr_PushConstant_uint %318 %int_8
        %768 = OpLoad %uint %767
        %769 = OpIMul %uint %323 %768
        %773 = OpIMul %uint %369 %768
        %775 = OpAccessChain %_ptr_PushConstant_uint %318 %int_10
        %776 = OpLoad %uint %775
        %777 = OpIMul %uint %773 %776
        %778 = OpIAdd %uint %769 %777
               OpBranch %780
        %780 = OpLabel
       %1115 = OpPhi %uint %uint_0 %495 %885 %783
        %786 = OpULessThan %bool %1115 %476
               OpLoopMerge %782 %783 Unroll
               OpBranchConditional %786 %781 %782
        %781 = OpLabel
               OpBranch %788
        %788 = OpLabel
       %1116 = OpPhi %uint %uint_0 %781 %883 %791
        %794 = OpULessThan %bool %1116 %89
               OpLoopMerge %790 %791 Unroll
               OpBranchConditional %794 %789 %790
        %789 = OpLabel
        %798 = OpIMul %uint %1116 %383
        %799 = OpIAdd %uint %757 %798
        %801 = OpIMul %uint %385 %90
        %802 = OpIAdd %uint %799 %801
        %806 = OpIMul %uint %1115 %689
        %807 = OpIAdd %uint %763 %806
        %809 = OpIMul %uint %389 %473
        %810 = OpIAdd %uint %807 %809
               OpBranch %812
        %812 = OpLabel
       %1118 = OpPhi %uint %uint_0 %789 %881 %815
        %818 = OpULessThan %bool %1118 %473
               OpLoopMerge %814 %815 Unroll
               OpBranchConditional %818 %813 %814
        %813 = OpLabel
               OpBranch %820
        %820 = OpLabel
       %1120 = OpPhi %uint %uint_0 %813 %879 %823
        %826 = OpULessThan %bool %1120 %90
               OpLoopMerge %822 %823 Unroll
               OpBranchConditional %826 %821 %822
        %821 = OpLabel
        %829 = OpIMul %uint %1115 %473
        %831 = OpIAdd %uint %829 %1118
        %832 = OpIMul %uint %831 %89
       %1754 = OpIAdd %uint %832 %1116
        %836 = OpIMul %uint %90 %1754
        %838 = OpIAdd %uint %836 %1120
        %841 = OpIAdd %uint %802 %1120
        %844 = OpULessThan %bool %841 %356
               OpSelectionMerge %846 None
               OpBranchConditional %844 %845 %846
        %845 = OpLabel
        %849 = OpIAdd %uint %810 %1118
        %850 = OpAccessChain %_ptr_PushConstant_uint %318 %int_1
        %851 = OpLoad %uint %850
        %852 = OpULessThan %bool %849 %851
               OpBranch %846
        %846 = OpLabel
        %853 = OpPhi %bool %844 %821 %852 %845
               OpSelectionMerge %855 None
               OpBranchConditional %853 %854 %855
        %854 = OpLabel
        %863 = OpIAdd %uint %810 %1118
        %865 = OpAccessChain %_ptr_PushConstant_uint %318 %int_5
        %866 = OpLoad %uint %865
        %867 = OpIMul %uint %863 %866
        %868 = OpIAdd %uint %778 %867
        %870 = OpIAdd %uint %868 %802
        %872 = OpIAdd %uint %870 %1120
        %874 = OpAccessChain %_ptr_Function_float %485 %838
        %875 = OpLoad %float %874
        %877 = OpAccessChain %_ptr_StorageBuffer_float %859 %int_0 %872
               OpStore %877 %875
               OpBranch %855
        %855 = OpLabel
               OpBranch %823
        %823 = OpLabel
        %879 = OpIAdd %uint %1120 %int_1
               OpBranch %820
        %822 = OpLabel
               OpBranch %815
        %815 = OpLabel
        %881 = OpIAdd %uint %1118 %int_1
               OpBranch %812
        %814 = OpLabel
               OpBranch %791
        %791 = OpLabel
        %883 = OpIAdd %uint %1116 %int_1
               OpBranch %788
        %790 = OpLabel
               OpBranch %783
        %783 = OpLabel
        %885 = OpIAdd %uint %1115 %int_1
               OpBranch %780
        %782 = OpLabel
               OpReturn
               OpFunctionEnd
