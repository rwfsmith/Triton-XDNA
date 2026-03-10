// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for SwiGLU (AIE2P): out = SiLU(gate) * up
//
// SwiGLU(gate, up) = gate * sigmoid(gate) * up
//
// The Linalg IR has the silu chain (extf, negf/subf, exp, addf, divf, mulf)
// plus an additional mulf for the final * up. After fuse_elementwise_linalg,
// this becomes a single generic with 2 bf16 inputs (gate, up) and 1 bf16
// output (out).
//
// AIE2P type mapping:
//   - math.exp:   bf16 ONLY -> needs vector_type_cast
//   - arith.divf: f32 ONLY  -> keep as f32
//   - arith.subf/addf/mulf: bf16 ONLY -> needs vector_type_cast
//
// Strategy: fuse_elementwise_linalg -> 3-operand tiling (like axpy) ->
// vectorize at 16 -> cast exp, subf, addf, mulf to bf16; divf stays f32.
//
// No extern_func.o needed on AIE2P (native bf16 exp intrinsic).
////////////////////////////////////////////////////////////////////////////////

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

    //===================================================================
    // PHASE 1: Initial Canonicalization
    //===================================================================
    %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func0 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op
    transform.apply_cse to %func0 : !transform.any_op

    //===================================================================
    // PHASE 2: Fuse Elementwise Chain
    //===================================================================
    // Fuse the entire chain: extf + sigmoid(negf/exp/addf/divf) + mulf(gate,sig)
    // + truncf + mulf(silu,up) into a single bf16->bf16 generic.
    // Result: 2 bf16 tensor inputs (gate, up) + 1 bf16 tensor output (out).
    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func1_fused = transform.air.fuse_elementwise_linalg %func1 : (!transform.any_op) -> !transform.any_op

    %func1a = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1a {
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func1a : !transform.any_op

    //===================================================================
    // PHASE 3: Vec-Add-Style Tiling Pattern
    //===================================================================
    // After fusion: single linalg.generic (2 bf16 inputs, 1 bf16 output).
    // Flatten -> bufferize result to L2 -> tile forall [256]

    %op = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    %op_flattened = transform.structured.flatten_elementwise %op
    : (!transform.any_op) -> !transform.any_op

    %op_res_shared, %new_op = transform.structured.bufferize_to_allocation %op_flattened
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %op_1 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled_op_1, %forall_op_1 =
      transform.structured.tile_using_forall %op_1 tile_sizes [256] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    //===================================================================
    // PHASE 4: Canonicalization
    //===================================================================
    %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_2 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_2 : !transform.any_op

    //===================================================================
    // PHASE 5: Pad and Promote to L1 (3 operands: gate, up, out)
    //===================================================================
    %op_2 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // 3 operands (gate, up, out) -- all bf16 after fusion
    %padded_op, %pad_op, %__ = transform.structured.pad %op_2 {
        padding_values=[0.0 : bf16, 0.0 : bf16, 0.0 : bf16],
        padding_dimensions=[0, 1, 2],
        nofold_flags=[1, 1, 1],
        copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad_op : (!transform.any_op) -> !transform.any_op

    // Promote input GATE to L1 (memory_space=2)
    %padded_gate = transform.get_producer_of_operand %padded_op[0] : (!transform.any_op) -> (!transform.any_op)
    %padded_gate_buffer, %padded_gate_new = transform.structured.bufferize_to_allocation %padded_gate
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote input UP to L1 (memory_space=2)
    %padded_up = transform.get_producer_of_operand %padded_op[1] : (!transform.any_op) -> (!transform.any_op)
    %padded_up_buffer, %padded_up_new = transform.structured.bufferize_to_allocation %padded_up
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote output OUT to L1 (memory_space=2)
    %padded_out = transform.get_producer_of_operand %padded_op[2] : (!transform.any_op) -> (!transform.any_op)
    %padded_out_buffer, %padded_out_new = transform.structured.bufferize_to_allocation %padded_out
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    //===================================================================
    // PHASE 6: Canonicalization
    //===================================================================
    %func_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_3 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op

    //===================================================================
    // PHASE 7: Bufferization
    //===================================================================
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

    //===================================================================
    // PHASE 8: Post-Bufferization Cleanup
    //===================================================================
    %func6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func6 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func6 : !transform.any_op
    transform.apply_patterns to %func6 {
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    %linalg_copies = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %memref_copies = transform.structured.linalg_copy_to_memref %linalg_copies : (!transform.any_op) -> !transform.any_op
    %func_op_updated = transform.air.remove_uninitialized_copy %func6 : (!transform.any_op) -> !transform.any_op
    %func_op_updated_1 = transform.air.eliminate_cascade_memcpy %func_op_updated : (!transform.any_op) -> !transform.any_op

    //===================================================================
    // PHASE 9: Vectorization Tiling (16-lane for bf16)
    //===================================================================
    %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %inner_most_generics, %vec_loops:1 =
      transform.structured.tile_using_for %linalg_generics tile_sizes [16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    //===================================================================
    // PHASE 10: AIR Constructs Mapping + Type Casts
    //===================================================================
    %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %parallel = transform.loop.forall_to_parallel %forall_as_herd : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %parallel : (!transform.any_op) -> !transform.any_op

    %copies_in_herd = transform.structured.match ops{["memref.copy", "linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %dmas_from_copies = transform.air.copy_to_dma %copies_in_herd : (!transform.any_op) -> !transform.any_op

    %vectorized_herd = transform.air.herd_vectorize %herd : (!transform.any_op) -> !transform.any_op

    // AIE2P vector type constraints (same as silu):
    //   bf16 ONLY: addf, subf, mulf, exp
    //   f32 ONLY:  divf
    // Cast all bf16-only ops from f32 to bf16 after vectorization.

    // math.exp -> bf16
    %vector_exps = transform.structured.match ops{["math.exp"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
    %exp_cast = transform.air.vector_type_cast %vector_exps {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    // arith.subf -> bf16
    %vector_subs = transform.structured.match ops{["arith.subf"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
    %sub_cast = transform.air.vector_type_cast %vector_subs {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    // arith.addf -> bf16
    %vector_adds = transform.structured.match ops{["arith.addf"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
    %add_cast = transform.air.vector_type_cast %vector_adds {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    // arith.mulf -> bf16 (covers both gate*sigmoid and silu*up multiplies)
    %vector_muls = transform.structured.match ops{["arith.mulf"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
    %mul_cast = transform.air.vector_type_cast %vector_muls {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    // arith.divf stays f32 (AIE2P supports f32 vector div)

    transform.yield
  }
}
