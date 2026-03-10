// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for AXPY (AIE2P): out = alpha * x + y
//
// The Linalg IR from triton-shared-opt for a BF16 axpy with f32 alpha has:
//   1. linalg.fill (alpha constant in f32)
//   2. linalg.generic: extf (x: bf16 -> f32)
//   3. linalg.generic: mulf (alpha, x_f32 -> f32)
//   4. linalg.generic: extf (y: bf16 -> f32)
//   5. linalg.generic: addf (alpha_x, y_f32 -> f32)
//   6. linalg.generic: truncf (f32 -> bf16)
//
// After fuse_elementwise_linalg: single generic with 2 bf16 inputs (x, y),
// 1 bf16 output (out), body contains extf+mulf+extf+addf+truncf with alpha
// folded as a scalar constant.
//
// Strategy: fuse_elementwise_linalg -> vec-add-style 3-operand tiling ->
// vectorize at 16 -> cast mulf and addf to bf16.
//
// No extern_func.o needed -- mulf and addf are native AIE instructions.
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
    // Fuse extf + mulf + extf + addf + truncf + fill into a single
    // bf16->bf16 generic. Alpha constant gets folded into the body.
    // Result: 2 bf16 tensor inputs (x, y) + 1 bf16 tensor output (out).
    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func1_fused = transform.air.fuse_elementwise_linalg %func1 : (!transform.any_op) -> !transform.any_op

    %func1a = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1a {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func1a : !transform.any_op

    //===================================================================
    // PHASE 3: Vec-Add-Style Tiling Pattern
    //===================================================================
    // After fusion: single linalg.generic (2 bf16 inputs, 1 bf16 output).
    // Flatten -> bufferize result to L2 -> tile forall [256]

    // Match the fused generic
    %op = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Flatten to 1D
    %op_flattened = transform.structured.flatten_elementwise %op
    : (!transform.any_op) -> !transform.any_op

    // Bufferize result to L2 (memory_space=1)
    %op_res_shared, %new_op = transform.structured.bufferize_to_allocation %op_flattened
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Tile with forall [256] for multi-core
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
    // PHASE 5: Pad and Promote to L1 (3 operands: x, y, out)
    //===================================================================
    %op_2 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // 3 operands (x, y, out) -- all bf16 after fusion
    %padded_op, %pad_op, %__ = transform.structured.pad %op_2 {
        padding_values=[0.0 : bf16, 0.0 : bf16, 0.0 : bf16],
        padding_dimensions=[0, 1, 2],
        nofold_flags=[1, 1, 1],
        copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad_op : (!transform.any_op) -> !transform.any_op

    // Promote input X to L1 (memory_space=2)
    %padded_x = transform.get_producer_of_operand %padded_op[0] : (!transform.any_op) -> (!transform.any_op)
    %padded_x_buffer, %padded_x_new = transform.structured.bufferize_to_allocation %padded_x
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote input Y to L1 (memory_space=2)
    %padded_y = transform.get_producer_of_operand %padded_op[1] : (!transform.any_op) -> (!transform.any_op)
    %padded_y_buffer, %padded_y_new = transform.structured.bufferize_to_allocation %padded_y
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

    // AIE2P vector type constraints:
    //   arith.mulf: bf16 ONLY (vector 16, 64)
    //   arith.addf: bf16 ONLY (vector 16, 32)
    // The fused generic body has f32 mulf and addf (from extf promotion).
    // Cast both to bf16 for AIE2P hardware support.

    // arith.mulf -> bf16
    %vector_muls = transform.structured.match ops{["arith.mulf"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
    %mul_cast = transform.air.vector_type_cast %vector_muls {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    // arith.addf -> bf16
    %vector_adds = transform.structured.match ops{["arith.addf"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
    %add_cast = transform.air.vector_type_cast %vector_adds {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
