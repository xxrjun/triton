// RUN: triton-opt %s -tritongpu-coalesce | FileCheck %s
// RUN: env TRITON_VERIFY_COALESCE_SLICE=1 triton-opt %s -tritongpu-coalesce 2>&1 | FileCheck %s --check-prefix=VERIFY
// VERIFY: Coalesce verified slices: {{[1-9][0-9]*}}

// CHECK-DAG: #[[VECTOR:[a-zA-Z0-9_]+]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @region_dependencies
  // CHECK: scf.if
  // CHECK: scf.for
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.load
  // CHECK: tt.store
  tt.func public @region_dependencies(%input: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %output: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %condition: i1) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %input_ptr = tt.addptr %input_base, %range : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %output_ptr = tt.addptr %output_base, %range : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %selected = scf.if %condition -> (tensor<128xf32, #blocked>) {
      %initial = tt.load %input_ptr : tensor<128x!tt.ptr<f32>, #blocked>
      %loop = scf.for %i = %zero to %two step %one iter_args(%current = %initial) -> (tensor<128xf32, #blocked>) {
        %value = tt.load %input_ptr : tensor<128x!tt.ptr<f32>, #blocked>
        %sum = arith.addf %current, %value : tensor<128xf32, #blocked>
        tt.store %output_ptr, %sum : tensor<128x!tt.ptr<f32>, #blocked>
        scf.yield %sum : tensor<128xf32, #blocked>
      }
      scf.yield %loop : tensor<128xf32, #blocked>
    } else {
      %other = tt.load %input_ptr : tensor<128x!tt.ptr<f32>, #blocked>
      scf.yield %other : tensor<128xf32, #blocked>
    }
    %outside = tt.load %input_ptr : tensor<128x!tt.ptr<f32>, #blocked>
    %combined = arith.addf %selected, %outside : tensor<128xf32, #blocked>
    tt.store %output_ptr, %combined : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // A tensor-pointer block argument must retain the original self-only guard.
  // CHECK-LABEL: tt.func public @pointer_block_argument
  // CHECK: tt.load
  // CHECK: tt.store
  tt.func public @pointer_block_argument(%ptr: tensor<128x!tt.ptr<f32>, #blocked>) {
    %value = tt.load %ptr : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %ptr, %value : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // Contiguous 1024-element accesses must actually change their vector width.
  // CHECK-LABEL: tt.func public @contiguous_vectorization
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, #[[VECTOR]]>
  // CHECK: tt.store {{.*}} : tensor<1024x!tt.ptr<f32>, #[[VECTOR]]>
  tt.func public @contiguous_vectorization(%input: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %output: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %input_ptr = tt.addptr %input_base, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %output_ptr = tt.addptr %output_base, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %value = tt.load %input_ptr : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.store %output_ptr, %value : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
