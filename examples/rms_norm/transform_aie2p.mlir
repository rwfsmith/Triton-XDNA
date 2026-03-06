// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Triton RMS Norm Tiling Recipe Transform Script (AIE2P)
//===----------------------------------------------------------------------===//
// AIE2P variant with wider vector widths (32 vs 16 for AIE2).
// Computes: y = x * rsqrt(mean(x^2) + eps)
//===----------------------------------------------------------------------===//

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
        // PHASE 2: Transpose Reductions and Data-Flow Navigation
        //===================================================================
        %reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %transformed_reduces = transform.air.transpose_reduce %reduces : (!transform.any_op) -> !transform.any_op

        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func1 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func1 : !transform.any_op

        %reduce_sum = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %mul_sq = transform.get_producer_of_operand %reduce_sum[0]
            : (!transform.any_op) -> !transform.any_op
        %extf_op = transform.get_producer_of_operand %mul_sq[0]
            : (!transform.any_op) -> !transform.any_op

        %all_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op

        //===================================================================
        // PHASE 3: Tiling and Fusion
        //===================================================================
        %fused_reduce = transform.air.fuse_multi_op_linalg %mul_sq, %reduce_sum : (!transform.any_op, !transform.any_op) -> !transform.any_op

        %generics_after_fuse = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %output_buf, %new_output = transform.structured.bufferize_to_allocation %generics_after_fuse
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

        %all_generics_2 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op

        //===================================================================
        // PHASE 4: Post-Fusion Canonicalization
        //===================================================================
        %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func2 : !transform.any_op

        //===================================================================
        // PHASE 5: L1 Memory Allocation
        //===================================================================
        %fills_2 = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %fill1_buffer, %fill1_new = transform.structured.bufferize_to_allocation %fills_2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        %generics3 = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %gen_buf, %gen_new = transform.structured.bufferize_to_allocation %generics3
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        //===================================================================
        // PHASE 6: Pre-Bufferization Canonicalization
        //===================================================================
        %func5 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func5 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func5 : !transform.any_op

        //===================================================================
        // PHASE 7: Complete Bufferization
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
        %func_op_updated = transform.air.remove_uninitialized_copy %func6 : (!transform.any_op) -> !transform.any_op

        //===================================================================
        // PHASE 9: Vectorization Tiling (AIE2P: 32-lane vectors)
        //===================================================================
        %linalg_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_reduce, %vec_loops_reduce:1 =
          transform.structured.tile_using_for %linalg_reduces tile_sizes [0, 32]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_generics, %vec_loops_1:1 =
          transform.structured.tile_using_for %linalg_generics tile_sizes [32]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_fills = transform.structured.convert_to_loops %linalg_fills : (!transform.any_op) -> !transform.any_op

        %func_rsqrt = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_rsqrt_converted = transform.air.convert_divf_sqrt_to_rsqrt %func_rsqrt : (!transform.any_op) -> !transform.any_op

        //===================================================================
        // PHASE 10: AIR Constructs Mapping
        //===================================================================
        %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %parallel = transform.loop.forall_to_parallel %forall_as_herd  : (!transform.any_op) -> !transform.any_op
        %herd = transform.air.par_to_herd %parallel : (!transform.any_op) -> !transform.any_op

        %linalg_copies_in_herd = transform.structured.match ops{["linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
        %memref_copies_in_herd = transform.structured.match ops{["memref.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
        %memref_copies_from_linalg = transform.structured.linalg_copy_to_memref %linalg_copies_in_herd : (!transform.any_op) -> !transform.any_op
        %all_copies = transform.merge_handles %memref_copies_in_herd, %memref_copies_from_linalg { deduplicate } : !transform.any_op
        %dmas = transform.air.copy_to_dma %all_copies : (!transform.any_op) -> !transform.any_op

        %vectorized_herd = transform.air.herd_vectorize %herd : (!transform.any_op) -> !transform.any_op

        %vector_reductions = transform.structured.match ops{["vector.multi_reduction"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
        %result10 = transform.air.vector_type_cast %vector_reductions {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

        %func_bcast = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_bcast_done = transform.air.broadcast_before_unary %func_bcast : (!transform.any_op) -> !transform.any_op

        %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func7_transformed = transform.air.convert_size1_vector_to_scalar %func7 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func7_transformed {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.vector.cast_away_vector_leading_one_dim
            transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
        } : !transform.any_op
        transform.apply_cse to %func7_transformed : !transform.any_op
    transform.yield
  }
}
