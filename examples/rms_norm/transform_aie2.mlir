// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Triton RMS Norm Transform Script (AIE2P)
//===----------------------------------------------------------------------===//
// y = x * rsqrt(mean(x^2) + eps)
//
// Follows the softmax pattern: tile last generic, fuse predecessors,
// allocate to L1, bufferize, vectorize, map to herd.
//===----------------------------------------------------------------------===//

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

    //===================================================================
    // PHASE 1: Canonicalization
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
    // PHASE 2: Navigate and tile output
    //===================================================================
    // IR: generic(x*x) -> reduce(sum) -> scalar(divf,rsqrt) ->
    //     generic(extf) -> fill(rsqrt) -> generic(mulf) -> generic(truncf)
    // The truncf generic is the output -- tile it and fuse predecessors.

    %generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sq_op, %extf_op, %mulf_op, %truncf_op = transform.split_handle %generics
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %reduce = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Bufferize output to L2
    %truncf_buf, %new_truncf = transform.structured.bufferize_to_allocation %truncf_op
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Tile output with forall [64] -- single tile for 1x1 herd
    %tiled_truncf, %forall =
      transform.structured.tile_using_forall %truncf_op tile_sizes [64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse predecessors into forall.
    // Order: mulf depends on fill+extf; fill depends on reduce; reduce depends on sq.
    // extf depends on x (shared with sq). Fuse direct deps of truncf first.
    %fused_mulf, %_1 = transform.structured.fuse_into_containing_op %mulf_op into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_extf, %_3 = transform.structured.fuse_into_containing_op %extf_op into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_fill, %_2 = transform.structured.fuse_into_containing_op %fill into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_reduce, %_4 = transform.structured.fuse_into_containing_op %reduce into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_sq, %_5 = transform.structured.fuse_into_containing_op %sq_op into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    //===================================================================
    // PHASE 3: Canonicalization
    //===================================================================
    %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func2 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func2 : !transform.any_op

    //===================================================================
    // PHASE 4: L1 Memory Allocation (softmax-style)
    //===================================================================
    // Fills to L1
    %fills2 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill_buf2, %fill_new2 = transform.structured.bufferize_to_allocation %fills2
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Fuse sq+reduce to reduce buffer count
    %generics3 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %reduces3 = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // After tiling+fusion, sq is now inside the forall before reduce
    // Split to get individual generics
    %g1, %g2, %g3, %g4 = transform.split_handle %generics3
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Fuse sq into reduce
    %fused_sq_red = transform.air.fuse_multi_op_linalg %g1, %reduces3 : (!transform.any_op, !transform.any_op) -> !transform.any_op

    // Promote reduce input to L1
    %reduce4 = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %reduce_inp = transform.get_operand %reduce4[0]
        : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %reduce_inp : !transform.any_value

    // Allocate remaining generics to L1
    %generics4 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %gen_buf4, %gen_new4 = transform.structured.bufferize_to_allocation %generics4
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote alloc_tensors (reduce init) to L1
    %at = transform.structured.match ops{["bufferization.alloc_tensor"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %at_buf, %at_new = transform.structured.bufferize_to_allocation %at
        {memory_space = 1, emit_dealloc} : !transform.any_op

    //===================================================================
    // PHASE 5: Canonicalization
    //===================================================================
    %func5 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func5 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func5 : !transform.any_op

    //===================================================================
    // PHASE 6: Bufferization
    //===================================================================
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

    //===================================================================
    // PHASE 7: Post-Bufferization Cleanup
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
    %memref_copies_conv = transform.structured.linalg_copy_to_memref %linalg_copies : (!transform.any_op) -> !transform.any_op
    %func_updated = transform.air.remove_uninitialized_copy %func6 : (!transform.any_op) -> !transform.any_op

    //===================================================================
    // PHASE 8: Vectorization
    //===================================================================
    %linalg_generics_f = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %inner_gen, %gen_lps:1 =
      transform.structured.tile_using_for %linalg_generics_f tile_sizes [16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %linalg_reduces_f = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %inner_red, %red_lps:1 =
      transform.structured.tile_using_for %linalg_reduces_f tile_sizes [16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %linalg_fills_f = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill_lps = transform.structured.convert_to_loops %linalg_fills_f : (!transform.any_op) -> !transform.any_op

    //===================================================================
    // PHASE 9: AIR Mapping
    //===================================================================
    %forall_h = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %par = transform.loop.forall_to_parallel %forall_h : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %par : (!transform.any_op) -> !transform.any_op

    %lc = transform.structured.match ops{["linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %mc = transform.structured.match ops{["memref.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %mc2 = transform.structured.linalg_copy_to_memref %lc : (!transform.any_op) -> !transform.any_op
    %all_c = transform.merge_handles %mc, %mc2 { deduplicate } : !transform.any_op
    %dmas = transform.air.copy_to_dma %all_c : (!transform.any_op) -> !transform.any_op

    %vh = transform.air.herd_vectorize %herd : (!transform.any_op) -> !transform.any_op

    // AIE2P bf16 casts
    %vm = transform.structured.match ops{["arith.mulf"]} in %vh : (!transform.any_op) -> !transform.any_op
    %mc3 = transform.air.vector_type_cast %vm {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op
    %va = transform.structured.match ops{["arith.addf"]} in %vh : (!transform.any_op) -> !transform.any_op
    %ac = transform.air.vector_type_cast %va {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %f7t = transform.air.convert_size1_vector_to_scalar %func7 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f7t {
        transform.apply_patterns.canonicalization
        transform.apply_patterns.vector.cast_away_vector_leading_one_dim
        transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
    } : !transform.any_op
    transform.apply_cse to %f7t : !transform.any_op

    transform.yield
  }
}
