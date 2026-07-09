!memref_gmem_bf16 = !cute.memref<bf16, gmem, align<16>, "(?,8192):(8192,1)">
!memref_gmem_bf16_1 = !cute.memref<bf16, gmem, align<16>, "(8192):(1)">
!memref_gmem_i8 = !cute.memref<i8, gmem, align<16>, "(?,4096):(4096,1)">
!memref_gmem_i8_1 = !cute.memref<i8, gmem, align<16>, "(?):(1)">
!memref_gmem_i8_2 = !cute.memref<i8, gmem, align<16>, "(4096):(1)">
module attributes {gpu.container_module} {
  gpu.module @kernels {
    cuda.kernel @kernel_cutlass_kernel_flashinferquantizationkernelsmxfp4_quantizeMXFP4QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o819281921_tensorptri8gmemalign16o409640961_tensorptri8gmem_0(%arg0: !memref_gmem_bf16, %arg1: !memref_gmem_i8, %arg2: !memref_gmem_i8_1, %arg3: i32, %arg4: i32) attributes {cu_attrs = {max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}, cute.kernel, gpu.kernel, nvvm.maxntid = array<i32: 1024, 1, 1>, nvvm.minctasm = 4 : i32, smem.max_smem_per_mp = 233472 : i64, smem.partition_num = 2 : i32} {
      %cst = arith.constant 6.000000e+00 : f32
      %c2_i32 = arith.constant 2 : i32
      %c31_i32 = arith.constant 31 : i32
      %c1_i32 = arith.constant 1 : i32
      %c-1_i32 = arith.constant -1 : i32
      %c8_i32 = arith.constant 8 : i32
      %c0_i8 = arith.constant 0 : i8
      %c32768_i32 = arith.constant 32768 : i32
      %c16_i32 = arith.constant 16 : i32
      %c512_i32 = arith.constant 512 : i32
      %c128_i32 = arith.constant 128 : i32
      %c32_i32 = arith.constant 32 : i32
      %c0_i32 = arith.constant 0 : i32
      %c256_i32 = arith.constant 256 : i32
      %c4_i32 = arith.constant 4 : i32
      %0 = nvvm.read.ptx.sreg.tid.x : i32
      %1 = nvvm.read.ptx.sreg.ctaid.x : i32
      %2 = nvvm.read.ptx.sreg.nctaid.x : i32
      llvm.inline_asm has_side_effects asm_dialect = att "griddepcontrol.wait;", ""  : () -> ()
      %3 = arith.floordivsi %0, %c4_i32 : i32
      %4 = arith.remsi %0, %c4_i32 : i32
      %5 = scf.while (%arg5 = %1) : (i32) -> i32 {
        %6 = arith.cmpi slt, %arg5, %arg4 : i32
        scf.condition(%6) %arg5 : i32
      } do {
      ^bb0(%arg5: i32):
        %6 = arith.cmpi sge, %arg5, %arg3 : i32
        scf.if %6 {
          %8 = scf.while (%arg6 = %3) : (i32) -> i32 {
            %9 = arith.cmpi slt, %arg6, %c256_i32 : i32
            scf.condition(%9) %arg6 : i32
          } do {
          ^bb0(%arg6: i32):
            %9 = arith.cmpi eq, %4, %c0_i32 : i32
            scf.if %9 {
              %11 = arith.remsi %arg6, %c4_i32 : i32
              %12 = arith.floordivsi %arg6, %c4_i32 : i32
              %13 = arith.remsi %arg5, %c32_i32 : i32
              %14 = arith.remsi %arg5, %c128_i32 : i32
              %15 = arith.floordivsi %14, %c32_i32 : i32
              %16 = arith.floordivsi %arg5, %c128_i32 : i32
              %17 = arith.muli %12, %c512_i32 : i32
              %18 = arith.addi %11, %17 : i32
              %19 = arith.muli %13, %c16_i32 : i32
              %20 = arith.addi %18, %19 : i32
              %21 = arith.muli %15, %c4_i32 : i32
              %22 = arith.addi %20, %21 : i32
              %23 = arith.muli %16, %c32768_i32 : i32
              %24 = arith.addi %22, %23 : i32
              %coord = cute.make_coord(%24) : (i32) -> !cute.coord<"?">
              cute.memref.store(%arg2, %coord, %c0_i8) : (!memref_gmem_i8_1, !cute.coord<"?">, i8) -> ()
            }
            %10 = arith.addi %arg6, %c128_i32 : i32
            scf.yield %10 : i32
          }
        } else {
          %8 = scf.while (%arg6 = %3) : (i32) -> i32 {
            %11 = arith.cmpi slt, %arg6, %c256_i32 : i32
            scf.condition(%11) %arg6 : i32
          } do {
          ^bb0(%arg6: i32):
            %11 = arith.muli %arg6, %c32_i32 : i32
            %12 = arith.muli %4, %c8_i32 : i32
            %13 = arith.addi %11, %12 : i32
            %coord = cute.make_coord(%arg5) : (i32) -> !cute.coord<"(?,_)">
            %slice = cute.slice(%arg0, %coord) : !memref_gmem_bf16, !cute.coord<"(?,_)">
            %iter = cute.get_iter(%slice) : !memref_gmem_bf16_1
            %int_tuple = cute.make_int_tuple(%13) : (i32) -> !cute.int_tuple<"?">
            %ptr = cute.add_offset(%iter, %int_tuple) : (!cute.ptr<bf16, gmem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<bf16, gmem>
            %14 = builtin.unrealized_conversion_cast %ptr : !cute.ptr<bf16, gmem> to !llvm.ptr<1>
            %15 = llvm.ptrtoint %14 : !llvm.ptr<1> to i64
            %16 = llvm.inline_asm asm_dialect = att "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l" %15 : (i64) -> !llvm.struct<(i32, i32, i32, i32)>
            %17 = llvm.extractvalue %16[0] : !llvm.struct<(i32, i32, i32, i32)> 
            %18 = llvm.extractvalue %16[1] : !llvm.struct<(i32, i32, i32, i32)> 
            %19 = llvm.extractvalue %16[2] : !llvm.struct<(i32, i32, i32, i32)> 
            %20 = llvm.extractvalue %16[3] : !llvm.struct<(i32, i32, i32, i32)> 
            %21 = llvm.inline_asm asm_dialect = att "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r" %17 : (i32) -> i32
            %22 = llvm.inline_asm asm_dialect = att "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r" %18 : (i32) -> i32
            %23 = llvm.inline_asm asm_dialect = att "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r" %19 : (i32) -> i32
            %24 = llvm.inline_asm asm_dialect = att "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r" %20 : (i32) -> i32
            %25 = llvm.inline_asm asm_dialect = att "max.bf16x2 $0, $1, $2;", "=r,r,r" %21, %22 : (i32, i32) -> i32
            %26 = llvm.inline_asm asm_dialect = att "max.bf16x2 $0, $1, $2;", "=r,r,r" %23, %24 : (i32, i32) -> i32
            %27 = llvm.inline_asm asm_dialect = att "max.bf16x2 $0, $1, $2;", "=r,r,r" %25, %26 : (i32, i32) -> i32
            %28 = llvm.inline_asm asm_dialect = att "\0A            {\0A                .reg .b32 lo, hi;\0A                .reg .f32 f0, f1;\0A                and.b32 lo, $1, 0xFFFF;\0A                shr.b32 hi, $1, 16;\0A                shl.b32 lo, lo, 16;\0A                shl.b32 hi, hi, 16;\0A                mov.b32 f0, lo;\0A                mov.b32 f1, hi;\0A                max.f32 $0, f0, f1;\0A            }\0A            ", "=f,r" %27 : (i32) -> f32
            %29 = nvvm.shfl.sync  bfly %c-1_i32, %28, %c1_i32, %c31_i32 : f32 -> f32
            %30 = llvm.inline_asm asm_dialect = att "max.f32 $0, $1, $2;", "=f,f,f" %28, %29 : (f32, f32) -> f32
            %31 = nvvm.shfl.sync  bfly %c-1_i32, %30, %c2_i32, %c31_i32 : f32 -> f32
            %32 = llvm.inline_asm asm_dialect = att "max.f32 $0, $1, $2;", "=f,f,f" %30, %31 : (f32, f32) -> f32
            %33 = llvm.inline_asm asm_dialect = att "rcp.approx.ftz.f32 $0, $1;", "=f,f" %cst : (f32) -> f32
            %34 = arith.mulf %32, %33 : f32
            %35 = llvm.inline_asm asm_dialect = att "\0A            {\0A                .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\0A                .reg .u32 bits, exp_biased, mantissa, bump, result;\0A\0A                setp.le.f32 p_zero, $1, 0f00000000;\0A\0A                mov.b32 bits, $1;\0A                shr.b32 exp_biased, bits, 23;\0A                and.b32 exp_biased, exp_biased, 255;\0A                and.b32 mantissa, bits, 0x7FFFFF;\0A\0A                setp.ne.u32 p_has_mant, mantissa, 0;\0A                selp.u32 bump, 1, 0, p_has_mant;\0A                setp.eq.u32 p_exp_zero, exp_biased, 0;\0A                setp.le.u32 p_tiny_sub, mantissa, 0x400000;\0A                and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\0A                @p_tiny_sub mov.u32 bump, 0;\0A                add.u32 result, exp_biased, bump;\0A\0A                setp.gt.u32 p_ovf, result, 254;\0A                selp.u32 result, 254, result, p_ovf;\0A                selp.u32 $0, 0, result, p_zero;\0A            }\0A            ", "=r,f" %34 : (f32) -> i32
            %36 = llvm.inline_asm asm_dialect = att "\0A            {\0A                .reg .s32 new_exp;\0A                .reg .b32 float_bits;\0A                .reg .pred p_zero;\0A\0A                setp.eq.u32 p_zero, $1, 0;\0A                sub.s32 new_exp, 254, $1;\0A                max.s32 new_exp, new_exp, 0;\0A                shl.b32 float_bits, new_exp, 23;\0A                mov.b32 $0, float_bits;\0A                @p_zero mov.b32 $0, 0;\0A            }\0A            ", "=f,r" %35 : (i32) -> f32
            %37 = llvm.inline_asm asm_dialect = att "\0A        {\0A            .reg .b32 lo, hi;\0A            .reg .f32 f0, f1;\0A            and.b32 lo, $2, 0xFFFF;\0A            shr.b32 hi, $2, 16;\0A            shl.b32 lo, lo, 16;\0A            shl.b32 hi, hi, 16;\0A            mov.b32 f0, lo;\0A            mov.b32 f1, hi;\0A            mul.f32 $0, f0, $3;\0A            mul.f32 $1, f1, $3;\0A        }\0A        ", "=f,=f,r,f" %17, %36 : (i32, f32) -> !llvm.struct<(f32, f32)>
            %38 = llvm.extractvalue %37[0] : !llvm.struct<(f32, f32)> 
            %39 = llvm.extractvalue %37[1] : !llvm.struct<(f32, f32)> 
            %40 = llvm.inline_asm asm_dialect = att "\0A        {\0A            .reg .b32 lo, hi;\0A            .reg .f32 f0, f1;\0A            and.b32 lo, $2, 0xFFFF;\0A            shr.b32 hi, $2, 16;\0A            shl.b32 lo, lo, 16;\0A            shl.b32 hi, hi, 16;\0A            mov.b32 f0, lo;\0A            mov.b32 f1, hi;\0A            mul.f32 $0, f0, $3;\0A            mul.f32 $1, f1, $3;\0A        }\0A        ", "=f,=f,r,f" %18, %36 : (i32, f32) -> !llvm.struct<(f32, f32)>
            %41 = llvm.extractvalue %40[0] : !llvm.struct<(f32, f32)> 
            %42 = llvm.extractvalue %40[1] : !llvm.struct<(f32, f32)> 
            %43 = llvm.inline_asm asm_dialect = att "\0A        {\0A            .reg .b32 lo, hi;\0A            .reg .f32 f0, f1;\0A            and.b32 lo, $2, 0xFFFF;\0A            shr.b32 hi, $2, 16;\0A            shl.b32 lo, lo, 16;\0A            shl.b32 hi, hi, 16;\0A            mov.b32 f0, lo;\0A            mov.b32 f1, hi;\0A            mul.f32 $0, f0, $3;\0A            mul.f32 $1, f1, $3;\0A        }\0A        ", "=f,=f,r,f" %19, %36 : (i32, f32) -> !llvm.struct<(f32, f32)>
            %44 = llvm.extractvalue %43[0] : !llvm.struct<(f32, f32)> 
            %45 = llvm.extractvalue %43[1] : !llvm.struct<(f32, f32)> 
            %46 = llvm.inline_asm asm_dialect = att "\0A        {\0A            .reg .b32 lo, hi;\0A            .reg .f32 f0, f1;\0A            and.b32 lo, $2, 0xFFFF;\0A            shr.b32 hi, $2, 16;\0A            shl.b32 lo, lo, 16;\0A            shl.b32 hi, hi, 16;\0A            mov.b32 f0, lo;\0A            mov.b32 f1, hi;\0A            mul.f32 $0, f0, $3;\0A            mul.f32 $1, f1, $3;\0A        }\0A        ", "=f,=f,r,f" %20, %36 : (i32, f32) -> !llvm.struct<(f32, f32)>
            %47 = llvm.extractvalue %46[0] : !llvm.struct<(f32, f32)> 
            %48 = llvm.extractvalue %46[1] : !llvm.struct<(f32, f32)> 
            %49 = llvm.inline_asm asm_dialect = att "\0A            {\0A                .reg .b8 byte0, byte1, byte2, byte3;\0A                cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;\0A                cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;\0A                cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;\0A                cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;\0A                mov.b32 $0, {byte0, byte1, byte2, byte3};\0A            }\0A            ", "=r,f,f,f,f,f,f,f,f" %38, %39, %41, %42, %44, %45, %47, %48 : (f32, f32, f32, f32, f32, f32, f32, f32) -> i32
            %slice_0 = cute.slice(%arg1, %coord) : !memref_gmem_i8, !cute.coord<"(?,_)">
            %iter_1 = cute.get_iter(%slice_0) : !memref_gmem_i8_2
            %50 = arith.muli %arg6, %c16_i32 : i32
            %51 = arith.muli %4, %c4_i32 : i32
            %52 = arith.addi %50, %51 : i32
            %int_tuple_2 = cute.make_int_tuple(%52) : (i32) -> !cute.int_tuple<"?">
            %ptr_3 = cute.add_offset(%iter_1, %int_tuple_2) : (!cute.ptr<i8, gmem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i8, gmem>
            %53 = builtin.unrealized_conversion_cast %ptr_3 : !cute.ptr<i8, gmem> to !llvm.ptr<1>
            %54 = llvm.ptrtoint %53 : !llvm.ptr<1> to i64
            llvm.inline_asm has_side_effects asm_dialect = att "st.global.u32 [$0], $1;", "l,r" %54, %49 : (i64, i32) -> ()
            %55 = arith.cmpi eq, %4, %c0_i32 : i32
            scf.if %55 {
              %57 = arith.remsi %arg6, %c4_i32 : i32
              %58 = arith.floordivsi %arg6, %c4_i32 : i32
              %59 = arith.remsi %arg5, %c32_i32 : i32
              %60 = arith.remsi %arg5, %c128_i32 : i32
              %61 = arith.floordivsi %60, %c32_i32 : i32
              %62 = arith.floordivsi %arg5, %c128_i32 : i32
              %63 = arith.muli %58, %c512_i32 : i32
              %64 = arith.addi %57, %63 : i32
              %65 = arith.muli %59, %c16_i32 : i32
              %66 = arith.addi %64, %65 : i32
              %67 = arith.muli %61, %c4_i32 : i32
              %68 = arith.addi %66, %67 : i32
              %69 = arith.muli %62, %c32768_i32 : i32
              %70 = arith.addi %68, %69 : i32
              %71 = arith.trunci %35 : i32 to i8
              %coord_4 = cute.make_coord(%70) : (i32) -> !cute.coord<"?">
              cute.memref.store(%arg2, %coord_4, %71) : (!memref_gmem_i8_1, !cute.coord<"?">, i8) -> ()
            }
            %56 = arith.addi %arg6, %c128_i32 : i32
            scf.yield %56 : i32
          }
          %9 = arith.addi %3, %c256_i32 : i32
          %10 = scf.while (%arg6 = %9) : (i32) -> i32 {
            %11 = arith.cmpi slt, %arg6, %c256_i32 : i32
            scf.condition(%11) %arg6 : i32
          } do {
          ^bb0(%arg6: i32):
            %11 = arith.cmpi eq, %4, %c0_i32 : i32
            scf.if %11 {
              %13 = arith.remsi %arg6, %c4_i32 : i32
              %14 = arith.floordivsi %arg6, %c4_i32 : i32
              %15 = arith.remsi %arg5, %c32_i32 : i32
              %16 = arith.remsi %arg5, %c128_i32 : i32
              %17 = arith.floordivsi %16, %c32_i32 : i32
              %18 = arith.floordivsi %arg5, %c128_i32 : i32
              %19 = arith.muli %14, %c512_i32 : i32
              %20 = arith.addi %13, %19 : i32
              %21 = arith.muli %15, %c16_i32 : i32
              %22 = arith.addi %20, %21 : i32
              %23 = arith.muli %17, %c4_i32 : i32
              %24 = arith.addi %22, %23 : i32
              %25 = arith.muli %18, %c32768_i32 : i32
              %26 = arith.addi %24, %25 : i32
              %coord = cute.make_coord(%26) : (i32) -> !cute.coord<"?">
              cute.memref.store(%arg2, %coord, %c0_i8) : (!memref_gmem_i8_1, !cute.coord<"?">, i8) -> ()
            }
            %12 = arith.addi %arg6, %c128_i32 : i32
            scf.yield %12 : i32
          }
        }
        %7 = arith.addi %arg5, %2 : i32
        scf.yield %7 : i32
      }
      llvm.inline_asm has_side_effects asm_dialect = att "griddepcontrol.launch_dependents;", ""  : () -> ()
      return
    }
  }
  func.func @cutlass___call___flashinferquantizationkernelsmxfp4_quantizeMXFP4QuantizeSwizzledKernel_object_at__FakeTensorBFloat16div1819281921_FakeTensorUint8div1409640961_FakeTensorUint8div11(%arg0: !memref_gmem_bf16, %arg1: !memref_gmem_i8, %arg2: !memref_gmem_i8_1, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: !cuda.stream) -> i32 attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c512_i32 = arith.constant 512 : i32
    %c232448_i64 = arith.constant 232448 : i64
    %0 = cute.kernel_smem_size @kernels::@kernel_cutlass_kernel_flashinferquantizationkernelsmxfp4_quantizeMXFP4QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o819281921_tensorptri8gmemalign16o409640961_tensorptri8gmem_0 : i64
    %1 = arith.cmpi sgt, %0, %c232448_i64 : i64
    scf.if %1 {
      cute.print("\0AError: kernel '@kernels::@kernel_cutlass_kernel_flashinferquantizationkernelsmxfp4_quantizeMXFP4QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o819281921_tensorptri8gmemalign16o409640961_tensorptri8gmem_0' launch shared memory exceeds current GPU arch sm_90a allowed. Allocated: {} bytes. Max: 232448 bytes.\0A\0A", %0) : i64
    }
    %2 = cuda.launch_cfg.create<max_attrs = 17 : i32> (blockDim = (%c512_i32, %c1_i32, %c1_i32), dynamicSmemBytes = %0, gridDim = (%arg5, %c1_i32, %c1_i32), stream = %arg6) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 17>
    cuda.launch_cfg.programmatic_stream_serialization_allowed[%2] %c1_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    cuda.launch_cfg.cooperative[%2] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    %3 = cuda.launch_ex @kernels::@kernel_cutlass_kernel_flashinferquantizationkernelsmxfp4_quantizeMXFP4QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o819281921_tensorptri8gmemalign16o409640961_tensorptri8gmem_0<%2> (%arg0, %arg1, %arg2, %arg3, %arg4) {assume_kernel_attr = #cuda.assume_kernel_attr<true>} : !cuda.launch_cfg<max_attrs = 17>, (!memref_gmem_bf16, !memref_gmem_i8, !memref_gmem_i8_1, i32, i32) -> !cuda.result
    %4 = cuda.cast %3 : !cuda.result -> i32
    cuda.return_if_error %4 : i32
    return %c0_i32 : i32
  }
}

