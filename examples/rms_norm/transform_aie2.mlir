// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Triton RMS Norm Tiling Recipe Transform Script (AIE2)
//===----------------------------------------------------------------------===//
// Implements tiling and optimization for RMS normalization:
//   y = x * rsqrt(mean(x^2) + eps)
//
// RMS NORM DECOMPOSITION (expected from triton-shared-opt):
//   1. extf (bf16 -> f32)
//   2. mul (x * x -> x_sq)
//   3. reduce_sum (sum of x_sq along reduction dim)
//   4. generic (divide by N, add eps)
//   5. generic (rsqrt)
//   6. broadcast (rstd to full row)
//   7. generic (x * rstd)
//   8. truncf (f32 -> bf16)
//
// MEMORY HIERARCHY: L3 (DDR) -> L2 (memory_space=1) -> L1 (memory_space=2)
//
// Pattern derived from the softmax transform script with simplifications:
// - Only 1 reduction (vs 2 for softmax: max + sum)
// - No exp/sub operations
// - Uses rsqrt intrinsic (AIE2 native support)
//===----------------------------------------------------------------------===//

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

        //===================================================================
        // PHASE 1: Initial Canonicalization and Cleanup
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
        // Transpose linalg.reduce to ensure reduction dimension is innermost
        // (required for AIE vector reduction intrinsics).
        %reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %transformed_reduces = transform.air.transpose_reduce %reduces : (!transform.any_op) -> !transform.any_op

        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func1 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func1 : !transform.any_op

        // Navigate data-flow from linalg.reduce anchor.
        // RMS Norm chain:
        //   extf -> mul -> reduce_sum -> div_by_N -> add_eps -> rsqrt -> broadcast -> mul_rstd -> truncf
        %reduce_sum = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op

        // Walk upstream from reduce: mul_sq -> extf
        %mul_sq = transform.get_producer_of_operand %reduce_sum[0]
            : (!transform.any_op) -> !transform.any_op
        %extf_op = transform.get_producer_of_operand %mul_sq[0]
            : (!transform.any_op) -> !transform.any_op

        // Walk downstream from reduce: div_by_N -> add_eps -> rsqrt -> broadcast -> mul_rstd -> truncf
        %post_reduce = transform.get_consumers_of_result %reduce_sum[0]
            : (!transform.any_op) -> !transform.any_op
        // The downstream ops after reduce are a chain of generics.
        // The final output is the truncf at the end.

        // Match all generics to find the output op for tiling
        %all_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op

        //===================================================================
        // PHASE 3: Tiling and Fusion
        //===================================================================
        // Fuse elementwise ops adjacent to the reduce for better locality.
        %fused_reduce = transform.air.fuse_multi_op_linalg %mul_sq, %reduce_sum : (!transform.any_op, !transform.any_op) -> !transform.any_op

        // Find the last generic (truncf/output) for tiling -- it drives the forall.
        %generics_after_fuse = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op

        // Get the last generic in the chain (the output operation).
        // After fuse_multi_op_linalg, the generics include:
        //   extf, fused_reduce, post-reduce ops..., truncf
        // We tile the last one (truncf/output) with [1] for per-row processing.
        %last_generic = transform.structured.match ops{["linalg.generic"]} in %arg1
            attributes {__last__} : (!transform.any_op) -> !transform.any_op

        // Bufferize output to L2
        %output_buf, %new_output = transform.structured.bufferize_to_allocation %generics_after_fuse
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

        // Re-match after bufferization -- fuse all ops into a per-row forall loop
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
        // Allocate fills and intermediates to L1 (memory_space=2)
        %fills_2 = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %fill1_buffer, %fill1_new = transform.structured.bufferize_to_allocation %fills_2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        // Allocate intermediate generic outputs to L1
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
        // PHASE 9: Vectorization Tiling
        //===================================================================
        // Tile reductions and generics to match AIE vector intrinsic widths.

        // Reduce: 16-lane vector intrinsic for BF16
        %linalg_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_reduce, %vec_loops_reduce:1 =
          transform.structured.tile_using_for %linalg_reduces tile_sizes [0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        // Generics: 16-lane vector intrinsic
        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_generics, %vec_loops_1:1 =
          transform.structured.tile_using_for %linalg_generics tile_sizes [16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        // Fill: scalar (single element for reduction init)
        %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_fills = transform.structured.convert_to_loops %linalg_fills : (!transform.any_op) -> !transform.any_op

        // Convert divf+sqrt to rsqrt for AIE native intrinsic
        %func_rsqrt = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_rsqrt_converted = transform.air.convert_divf_sqrt_to_rsqrt %func_rsqrt : (!transform.any_op) -> !transform.any_op

        //===================================================================
        // PHASE 10: AIR Constructs Mapping
        //===================================================================
        // Convert forall to herd for multi-core execution
        %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %parallel = transform.loop.forall_to_parallel %forall_as_herd  : (!transform.any_op) -> !transform.any_op
        %herd = transform.air.par_to_herd %parallel : (!transform.any_op) -> !transform.any_op

        // Convert copies to DMA
        %linalg_copies_in_herd = transform.structured.match ops{["linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
        %memref_copies_in_herd = transform.structured.match ops{["memref.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
        %memref_copies_from_linalg = transform.structured.linalg_copy_to_memref %linalg_copies_in_herd : (!transform.any_op) -> !transform.any_op
        %all_copies = transform.merge_handles %memref_copies_in_herd, %memref_copies_from_linalg { deduplicate } : !transform.any_op
        %dmas = transform.air.copy_to_dma %all_copies : (!transform.any_op) -> !transform.any_op

        // Vectorize herd
        %vectorized_herd = transform.air.herd_vectorize %herd : (!transform.any_op) -> !transform.any_op

        // Cast vector reductions to BF16 for AIE intrinsic compatibility
        %vector_reductions = transform.structured.match ops{["vector.multi_reduction"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
        %result10 = transform.air.vector_type_cast %vector_reductions {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

        // Broadcast scalar rsqrt before unary multiply
        %func_bcast = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_bcast_done = transform.air.broadcast_before_unary %func_bcast : (!transform.any_op) -> !transform.any_op

        // Convert size-1 vectors to scalars
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
