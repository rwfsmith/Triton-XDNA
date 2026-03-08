// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// RMS Norm transform for AIE2P. Requires mlir-air >= 1b0ae6e.
// Uses the EXACT relu pattern (flatten → L2 → forall → pad → L1 → bufferize → herd)
// applied to the fused output generic only. Reduce+scalar chain stays outside.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

    // Phase 1: Canonicalize
    %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func0 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op
    transform.apply_cse to %func0 : !transform.any_op

    // Phase 2: Fuse post-reduce chain FIRST, then pre-allocate remaining tensors.
    // Fusion must happen before bufferize_to_allocation to preserve the
    // elementwise chain that fuse_elementwise_linalg needs.
    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func1_fused = transform.air.fuse_elementwise_linalg %func1 : (!transform.any_op) -> !transform.any_op
    %func1a = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1a { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %func1a : !transform.any_op

    // Phase 3: EXACT relu pattern on the fused output generic
    %all_gens = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %other_gens, %op = transform.split_handle %all_gens {overflow_result = 0}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Flatten to 1D (no-op for already-1D generic)
    %op_flattened = transform.structured.flatten_elementwise %op
    : (!transform.any_op) -> !transform.any_op

    // Bufferize result to L2
    %op_res_shared, %new_op = transform.structured.bufferize_to_allocation %op_flattened
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Tile the flattened output generic
    %tiled_op_1, %forall_op_1 =
      transform.structured.tile_using_forall %op_flattened tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pre-allocate tensor ops OUTSIDE the forall to L2 (memory_space 1).
    // These are NOT inside the forall so they don't affect the pad step.
    // Match alloc_tensor and tensor.empty OUTSIDE the forall scope.
    %at = transform.structured.match ops{["bufferization.alloc_tensor"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %at_buf, %at_new = transform.structured.bufferize_to_allocation %at
        {memory_space = 1, emit_dealloc} : !transform.any_op

    // Phase 4: Canonicalize
    %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_2 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_2 : !transform.any_op

    // Phase 5: Pad and promote to L1 (EXACT relu pattern)
    %op_2 = transform.structured.match ops{["linalg.generic"]} in %forall_op_1 : (!transform.any_op) -> !transform.any_op
    %padded_op, %pad_op, %__ = transform.structured.pad %op_2 {
        padding_values=[0.0 : bf16, 0.0 : bf16],
        padding_dimensions=[0, 1],
        nofold_flags=[1, 1],
        copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad_op : (!transform.any_op) -> !transform.any_op

    %padded_input = transform.get_producer_of_operand %padded_op[0] : (!transform.any_op) -> (!transform.any_op)
    %pi_buf, %pi_new = transform.structured.bufferize_to_allocation %padded_input
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %padded_result = transform.get_producer_of_operand %padded_op[1] : (!transform.any_op) -> (!transform.any_op)
    %pr_buf, %pr_new = transform.structured.bufferize_to_allocation %padded_result
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Phase 6: Canonicalize
    %func_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_3 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op

    // Phase 7: Bufferize
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

    // Phase 8: Cleanup
    %func6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func6 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func6 : !transform.any_op
    transform.apply_patterns to %func6 { transform.apply_patterns.canonicalization } : !transform.any_op
    %lc = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mc = transform.structured.linalg_copy_to_memref %lc : (!transform.any_op) -> !transform.any_op
    %func_upd = transform.air.remove_uninitialized_copy %func6 : (!transform.any_op) -> !transform.any_op
    %func_upd2 = transform.air.eliminate_cascade_memcpy %func_upd : (!transform.any_op) -> !transform.any_op

    // Phase 9: Vectorize generics at 16-lane
    %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %inner_gen, %gen_lps:1 =
      transform.structured.tile_using_for %linalg_generics tile_sizes [16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Scalarize reduces and fills
    %all_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %red_loops = transform.structured.convert_to_loops %all_reduces : (!transform.any_op) -> !transform.any_op
    %all_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill_loops = transform.structured.convert_to_loops %all_fills : (!transform.any_op) -> !transform.any_op

    // Phase 10: Forall → Herd
    %forall_h = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %par = transform.loop.forall_to_parallel %forall_h : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %par : (!transform.any_op) -> !transform.any_op

    // DMA inside herd
    %lc2 = transform.structured.match ops{["linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %mc2 = transform.structured.match ops{["memref.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %mc3 = transform.structured.linalg_copy_to_memref %lc2 : (!transform.any_op) -> !transform.any_op
    %all_c = transform.merge_handles %mc2, %mc3 { deduplicate } : !transform.any_op
    %dmas = transform.air.copy_to_dma %all_c : (!transform.any_op) -> !transform.any_op

    // Vectorize herd
    %vh = transform.air.herd_vectorize %herd : (!transform.any_op) -> !transform.any_op

    // AIE2P bf16 casts for mulf (x * rsqrt)
    %vm = transform.structured.match ops{["arith.mulf"]} in %vh : (!transform.any_op) -> !transform.any_op
    %mc4 = transform.air.vector_type_cast %vm {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
