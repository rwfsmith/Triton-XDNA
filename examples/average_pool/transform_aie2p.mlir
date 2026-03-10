// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for Average Pooling (AIE2P)
//
// avg_pool(x) = mean(x, dim=-1) per row
//
// 2D kernel [BLOCK_M, BLOCK_N] with reduction over columns.
// Uses the rms_norm reduction pattern with linalg_promote for L1 staging.
// Requires mlir-air >= 4bc5734 (fix for linalg_promote memref.cast #1399).
////////////////////////////////////////////////////////////////////////////////

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

    // Phase 1: Canonicalization
    %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func0 {
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op
    transform.apply_cse to %func0 : !transform.any_op

    // Phase 2: Transpose reduce + fuse elementwise
    %reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tr = transform.air.transpose_reduce %reduces : (!transform.any_op) -> !transform.any_op
    %func1a = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1a { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %func1a : !transform.any_op

    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %f = transform.air.fuse_elementwise_linalg %func1 : (!transform.any_op) -> !transform.any_op
    %fa = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fa { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %fa : !transform.any_op

    // Phase 3: Match, tile, fuse
    // After fusion: 1 generic (fused extf+divf+truncf), 1 reduce, 1 fill
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %reduce = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // L2 output alloc
    %ob, %nb = transform.structured.bufferize_to_allocation %generic
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op
    // Tile at [2] not [1]: single bf16 = 2 bytes, below 4-byte DMA alignment
    %t, %fl = transform.structured.tile_using_forall %generic tile_sizes [2]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Fuse into forall
    %f1, %fl1 = transform.structured.fuse_into_containing_op %reduce into %fl
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f2, %fl2 = transform.structured.fuse_into_containing_op %fill into %fl1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Phase 4: Fill dest to L1
    %fills3 = transform.structured.match ops{["linalg.fill"]} in %fl2 : (!transform.any_op) -> !transform.any_op
    %fill_buf, %fill_new = transform.structured.bufferize_to_allocation %fills3
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Phase 5: Canonicalize + bufferize
    %f2c = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f2c { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %f2c : !transform.any_op
    %fop = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fb = transform.bufferization.one_shot_bufferize %fop : (!transform.any_op) -> !transform.any_op
    %f6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f6 { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %f6 : !transform.any_op
    %lc = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mc = transform.structured.linalg_copy_to_memref %lc : (!transform.any_op) -> !transform.any_op
    %fu = transform.air.remove_uninitialized_copy %f6 : (!transform.any_op) -> (!transform.any_op)
    %fu2 = transform.air.eliminate_cascade_memcpy %fu : (!transform.any_op) -> (!transform.any_op)

    // Phase 6: L1 promote (linalg_promote with fix from mlir-air #1399)
    %forall_op = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %gens_f = transform.structured.match ops{["linalg.generic"]} in %forall_op : (!transform.any_op) -> !transform.any_op
    %reds_f = transform.structured.match ops{["linalg.reduce"]} in %forall_op : (!transform.any_op) -> !transform.any_op
    %all_linalg_f = transform.merge_handles %reds_f, %gens_f { deduplicate } : !transform.any_op
    %promoted = transform.air.linalg_promote %all_linalg_f {memory_space = "L1"} : (!transform.any_op) -> !transform.any_op

    // Phase 7: Herd + DMA
    %fh = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %pa = transform.loop.forall_to_parallel %fh : (!transform.any_op) -> !transform.any_op
    %h = transform.air.par_to_herd %pa : (!transform.any_op) -> !transform.any_op
    %lc2 = transform.structured.match ops{["linalg.copy"]} in %h : (!transform.any_op) -> !transform.any_op
    %mc2 = transform.structured.match ops{["memref.copy"]} in %h : (!transform.any_op) -> !transform.any_op
    %mc3 = transform.structured.linalg_copy_to_memref %lc2 : (!transform.any_op) -> !transform.any_op
    %ac = transform.merge_handles %mc2, %mc3 { deduplicate } : !transform.any_op
    %dm = transform.air.copy_to_dma %ac : (!transform.any_op) -> !transform.any_op

    // Phase 8: Vectorization
    %h2 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Tile reduce at [0, 16] for vectorization
    %reds_h = transform.structured.match ops{["linalg.reduce"]} in %h2 : (!transform.any_op) -> !transform.any_op
    %inner_r, %inner_rl:1 = transform.structured.tile_using_for %reds_h tile_sizes [0, 16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Generic is scalar (divf per row) -- convert to loops
    %gens_h = transform.structured.match ops{["linalg.generic"]} in %h2 : (!transform.any_op) -> !transform.any_op
    %gen_scl = transform.structured.convert_to_loops %gens_h : (!transform.any_op) -> !transform.any_op

    // Fill is scalar -- convert to loops
    %fills_h = transform.structured.match ops{["linalg.fill"]} in %h2 : (!transform.any_op) -> !transform.any_op
    %fill_scl = transform.structured.convert_to_loops %fills_h : (!transform.any_op) -> !transform.any_op

    %vh = transform.air.herd_vectorize %h2 : (!transform.any_op) -> !transform.any_op

    // Phase 9: Lower reductions + type casts
    %func_final = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_final {
        transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
        transform.apply_patterns.vector.lower_contraction
        transform.apply_patterns.vector.lower_transfer
    } : !transform.any_op
    transform.apply_cse to %func_final : !transform.any_op

    // addf -> bf16 (from reduction lowering)
    %vh2 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %vector_adds = transform.structured.match ops{["arith.addf"]} in %vh2 : (!transform.any_op) -> !transform.any_op
    %add_cast = transform.air.vector_type_cast %vector_adds {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    %func_s1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_s1_done = transform.air.convert_size1_vector_to_scalar %func_s1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_s1_done {
        transform.apply_patterns.vector.cast_away_vector_leading_one_dim
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_s1_done : !transform.any_op

    transform.yield
  }
}
