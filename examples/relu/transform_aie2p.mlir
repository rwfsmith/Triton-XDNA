// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for ReLU (AIE2P): Elementwise Unary Operation
//
// The Linalg IR from triton-shared-opt for a BF16 relu has this structure:
//   1. linalg.fill (zero constant in f32)
//   2. linalg.generic: extf (bf16 -> f32)
//   3. linalg.generic: maxnumf (f32, f32 -> f32)  -- the actual relu
//   4. linalg.generic: truncf (f32 -> bf16)        -- output
//
// TARGET AIR structure (from mlir-air/programming_examples/relu/relu.py):
//   - 1x2 herd, direct L3-L1 DMA
//   - 2 L1 buffers (input bf16, output bf16)
//   - Single vectorized loop: read bf16 -> maximumf bf16 -> write bf16
//
// Strategy: Use fuse_elementwise_linalg to fuse extf+maxnumf+truncf into a
// single bf16->bf16 generic, then apply the vec-add-style pad+promote pattern.
// Also fuse the fill constant into the fused generic to eliminate the
// separate f32 zero buffer.
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
    // Fuse extf + maxnumf + truncf + fill into a single bf16->bf16 generic.
    // This eliminates intermediate f32 buffers and matches the mlir-air
    // reference structure.
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
    // After fusion, there should be a single linalg.generic (bf16 in, bf16 out).
    // Follow vec-add pattern: flatten -> bufferize L2 -> tile forall ->
    // pad -> promote L1 -> bufferize -> vectorize -> herd.

    // Match the fused generic
    %op = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Flatten to 1D
    %op_flattened = transform.structured.flatten_elementwise %op
    : (!transform.any_op) -> !transform.any_op

    // Bufferize result to L2
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
    // PHASE 5: Pad and Promote to L1
    //===================================================================
    %op_2 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // After fusion, the generic should be bf16 -> bf16
    %padded_op, %pad_op, %__ = transform.structured.pad %op_2 {
        padding_values=[0.0 : bf16, 0.0 : bf16],
        padding_dimensions=[0, 1],
        nofold_flags=[1, 1],
        copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad_op : (!transform.any_op) -> !transform.any_op

    // Promote input to L1 (memory_space=2)
    %padded_input = transform.get_producer_of_operand %padded_op[0] : (!transform.any_op) -> (!transform.any_op)
    %padded_input_buffer, %padded_input_new = transform.structured.bufferize_to_allocation %padded_input
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote output to L1 (memory_space=2)
    %padded_result = transform.get_producer_of_operand %padded_op[1] : (!transform.any_op) -> (!transform.any_op)
    %padded_result_buffer, %padded_result_new = transform.structured.bufferize_to_allocation %padded_result
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
    // PHASE 10: AIR Constructs Mapping
    //===================================================================
    %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %parallel = transform.loop.forall_to_parallel %forall_as_herd : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %parallel : (!transform.any_op) -> !transform.any_op

    %copies_in_herd = transform.structured.match ops{["memref.copy", "linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %dmas_from_copies = transform.air.copy_to_dma %copies_in_herd : (!transform.any_op) -> !transform.any_op

    %vectorized_herd = transform.air.herd_vectorize %herd : (!transform.any_op) -> !transform.any_op

    // Cast maxnumf from f32 to bf16 -- AIE only supports bf16 vector max.
    // The fused generic still computes max in f32 (via extf/truncf wrappers);
    // this cast makes the vectorized max use bf16 intrinsics.
    %vector_maxs = transform.structured.match ops{["arith.maxnumf"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
    %result_cast = transform.air.vector_type_cast %vector_maxs {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
