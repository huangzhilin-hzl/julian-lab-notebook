!memref_gmem_bf16 = !cute.memref<bf16, gmem, align<16>, "(?,8192):(8192,1)">
!memref_gmem_bf16_1 = !cute.memref<bf16, gmem, align<16>, "(8192):(1)">
!memref_gmem_i8 = !cute.memref<i8, gmem, align<16>, "(?,4096):(4096,1)">
!memref_gmem_i8_1 = !cute.memref<i8, gmem, align<16>, "(?):(1)">
!memref_gmem_i8_2 = !cute.memref<i8, gmem, align<16>, "(4096):(1)">
module attributes {gpu.container_module} {
  gpu.module @kernels {
    cuda.kernel @kernel_cutlass_kernel_flashinferquantizationkernelsmxfp4_quantizeMXFP4QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o819281921_tensorptri8gmemalign16o409640961_tensorptri8gmem_0(%arg0: !memref_gmem_bf16, %arg1: !memref_gmem_i8, %arg2: !memref_gmem_i8_1, %arg3: i32, %arg4: i32) attributes {cu_attrs = {max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}, cute.kernel, gpu.kernel, nvvm.maxntid = array<i32: 1024, 1, 1>, nvvm.minctasm = 4 : i32, smem.max_smem_per_mp = 233472 : i64, smem.partition_num = 2 : i32} {
      %iter = cute.get_iter(%arg0) : !memref_gmem_bf16
      %iter_0 = cute.get_iter(%arg1) : !memref_gmem_i8
      %iter_1 = cute.get_iter(%arg2) : !memref_gmem_i8_1
      %iter_2 = cute.get_iter(%arg0) : !memref_gmem_bf16
      %iter_3 = cute.get_iter(%arg1) : !memref_gmem_i8
      %iter_4 = cute.get_iter(%arg2) : !memref_gmem_i8_1
      %lay = cute.get_layout(%arg0) : !memref_gmem_bf16
      %lay_5 = cute.get_layout(%arg1) : !memref_gmem_i8
      %lay_6 = cute.get_layout(%arg2) : !memref_gmem_i8_1
      %0 = nvvm.read.ptx.sreg.tid.x : i32
      %1 = nvvm.read.ptx.sreg.tid.y : i32
      %2 = nvvm.read.ptx.sreg.tid.z : i32
      %3 = nvvm.read.ptx.sreg.ctaid.x : i32
      %4 = nvvm.read.ptx.sreg.ctaid.y : i32
      %5 = nvvm.read.ptx.sreg.ctaid.z : i32
      %6 = nvvm.read.ptx.sreg.nctaid.x : i32
      %7 = nvvm.read.ptx.sreg.nctaid.y : i32
      %8 = nvvm.read.ptx.sreg.nctaid.z : i32
      llvm.inline_asm has_side_effects asm_dialect = att "griddepcontrol.wait;", ""  : () -> ()
      %c4_i32 = arith.constant 4 : i32
      %9 = arith.floordivsi %0, %c4_i32 : i32
      %10 = arith.remsi %0, %c4_i32 : i32
      %11:2 = scf.while (%arg5 = %arg2, %arg6 = %3) : (!memref_gmem_i8_1, i32) -> (!memref_gmem_i8_1, i32) {
        %iter_10 = cute.get_iter(%arg5) : !memref_gmem_i8_1
        %iter_11 = cute.get_iter(%arg5) : !memref_gmem_i8_1
        %12 = arith.cmpi slt, %arg6, %arg4 : i32
        scf.condition(%12) %arg5, %arg6 : !memref_gmem_i8_1, i32
      } do {
      ^bb0(%arg5: !memref_gmem_i8_1, %arg6: i32):
        %iter_10 = cute.get_iter(%arg5) : !memref_gmem_i8_1
        %iter_11 = cute.get_iter(%arg5) : !memref_gmem_i8_1
        %12 = arith.cmpi sge, %arg6, %arg3 : i32
        %13 = scf.if %12 -> (!memref_gmem_i8_1) {
          %iter_15 = cute.get_iter(%arg5) : !memref_gmem_i8_1
          %15:2 = scf.while (%arg7 = %arg5, %arg8 = %9) : (!memref_gmem_i8_1, i32) -> (!memref_gmem_i8_1, i32) {
            %iter_19 = cute.get_iter(%arg7) : !memref_gmem_i8_1
            %iter_20 = cute.get_iter(%arg7) : !memref_gmem_i8_1
            %c256_i32 = arith.constant 256 : i32
            %16 = arith.cmpi slt, %arg8, %c256_i32 : i32
            scf.condition(%16) %arg7, %arg8 : !memref_gmem_i8_1, i32
          } do {
          ^bb0(%arg7: !memref_gmem_i8_1, %arg8: i32):
            %iter_19 = cute.get_iter(%arg7) : !memref_gmem_i8_1
            %iter_20 = cute.get_iter(%arg7) : !memref_gmem_i8_1
            %c0_i32 = arith.constant 0 : i32
            %16 = arith.cmpi eq, %10, %c0_i32 : i32
            %17 = scf.if %16 -> (!memref_gmem_i8_1) {
              %iter_24 = cute.get_iter(%arg7) : !memref_gmem_i8_1
              %c4_i32_25 = arith.constant 4 : i32
              %19 = arith.remsi %arg8, %c4_i32_25 : i32
              %20 = arith.floordivsi %arg8, %c4_i32_25 : i32
              %c32_i32 = arith.constant 32 : i32
              %21 = arith.remsi %arg6, %c32_i32 : i32
              %c128_i32_26 = arith.constant 128 : i32
              %22 = arith.remsi %arg6, %c128_i32_26 : i32
              %23 = arith.floordivsi %22, %c32_i32 : i32
              %24 = arith.floordivsi %arg6, %c128_i32_26 : i32
              %c512_i32 = arith.constant 512 : i32
              %25 = arith.muli %20, %c512_i32 : i32
              %26 = arith.addi %19, %25 : i32
              %c16_i32 = arith.constant 16 : i32
              %27 = arith.muli %21, %c16_i32 : i32
              %28 = arith.addi %26, %27 : i32
              %29 = arith.muli %23, %c4_i32_25 : i32
              %30 = arith.addi %28, %29 : i32
              %c32768_i32 = arith.constant 32768 : i32
              %31 = arith.muli %24, %c32768_i32 : i32
              %32 = arith.addi %30, %31 : i32
              %c0_i8 = arith.constant 0 : i8
              %coord = cute.make_coord(%32) : (i32) -> !cute.coord<"?">
              cute.memref.store(%arg7, %coord, %c0_i8) : (!memref_gmem_i8_1, !cute.coord<"?">, i8) -> ()
              scf.yield %arg7 : !memref_gmem_i8_1
            } else {
              %iter_24 = cute.get_iter(%arg7) : !memref_gmem_i8_1
              scf.yield %arg7 : !memref_gmem_i8_1
            }
            %iter_21 = cute.get_iter(%17) : !memref_gmem_i8_1
            %iter_22 = cute.get_iter(%17) : !memref_gmem_i8_1
            %iter_23 = cute.get_iter(%17) : !memref_gmem_i8_1
            %c128_i32 = arith.constant 128 : i32
            %18 = arith.addi %arg8, %c128_i32 : i32
            scf.yield %17, %18 : !memref_gmem_i8_1, i32
          }
          %iter_16 = cute.get_iter(%15#0) : !memref_gmem_i8_1
          %iter_17 = cute.get_iter(%15#0) : !memref_gmem_i8_1
          %iter_18 = cute.get_iter(%15#0) : !memref_gmem_i8_1
          scf.yield %15#0 : !memref_gmem_i8_1
        } else {
          %iter_15 = cute.get_iter(%arg5) : !memref_gmem_i8_1
          %15:2 = scf.while (%arg7 = %arg5, %arg8 = %9) : (!memref_gmem_i8_1, i32) -> (!memref_gmem_i8_1, i32) {
            %iter_22 = cute.get_iter(%arg7) : !memref_gmem_i8_1
            %iter_23 = cute.get_iter(%arg7) : !memref_gmem_i8_1
            %c256_i32_24 = arith.constant 256 : i32
            %18 = arith.cmpi slt, %arg8, %c256_i32_24 : i32
            scf.condition(%18) %arg7, %arg8 : !memref_gmem_i8_1, i32
          } do {
          ^bb0(%arg7: !memref_gmem_i8_1, %arg8: i32):
            %iter_22 = cute.get_iter(%arg7) : !memref_gmem_i8_1
            %iter_23 = cute.get_iter(%arg7) : !memref_gmem_i8_1
            %c32_i32 = arith.constant 32 : i32
            %18 = arith.muli %arg8, %c32_i32 : i32
            %c8_i32 = arith.constant 8 : i32
            %19 = arith.muli %10, %c8_i32 : i32
            %20 = arith.addi %18, %19 : i32
            %coord = cute.make_coord(%arg6) : (i32) -> !cute.coord<"(?,_)">
            %slice = cute.slice(%arg0, %coord) : !memref_gmem_bf16, !cute.coord<"(?,_)">
            %iter_24 = cute.get_iter(%slice) : !memref_gmem_bf16_1
            %iter_25 = cute.get_iter(%slice) : !memref_gmem_bf16_1
            %int_tuple = cute.make_int_tuple(%20) : (i32) -> !cute.int_tuple<"?">
            %ptr = cute.add_offset(%iter_25, %int_tuple) : (!cute.ptr<bf16, gmem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<bf16, gmem>
            %21 = builtin.unrealized_conversion_cast %ptr : !cute.ptr<bf16, gmem> to !llvm.ptr<1>
            %22 = llvm.ptrtoint %21 : !llvm.ptr<1> to i64
            %23 = llvm.inline_asm asm_dialect = att "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l" %22 : (i64) -> !llvm.struct<(i32, i32, i32, i32)>
            %24 = llvm.extractvalue %23[0] : !llvm.struct<(i32, i32, i32, i32)> 
            %25 = llvm.extractvalue %23[1] : !llvm.struct<(i32, i32, i32, i32)> 
            %26 = llvm.extractvalue %23[2] : !llvm.struct<(i32, i32, i32, i32)> 
            %27 = llvm.extractvalue %23[3] : !llvm.struct<(i32, i32, i32, i32)> 
            %28 = llvm.inline_asm asm_dialect = att "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r" %24 : (i32) -> i32
            %29 = llvm.inline_asm asm_dialect = att "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r" %25 : (i32) -> i32
            %30 = llvm.inline_asm asm_dialect = att "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r" %26 : (i32) -> i32
            %31 = llvm.inline_asm asm_dialect = att "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r" %27 : (i32) -> i32
            %32 = llvm.inline_asm asm_dialect = att "max.bf16x2 $0, $1, $2;", "=r,r,r" %28, %29 : (i32, i32) -> i32
            %33 = llvm.inline_asm asm_dialect = att "max.bf16x2 $0, $1, $2;", "=r,r,r" %30, %31 : (i32, i32) -> i32
            %34 = llvm.inline_asm asm_dialect = att "max.bf16x2 $0, $1, $2;", "=r,r,r" %32, %33 : (i32, i32) -> i32
            %35 = llvm.inline_asm asm_dialect = att "\0A            {\0A                .reg .b32 lo, hi;\0A                .reg .f32 f0, f1;\0A                and.b32 lo, $1, 0xFFFF;\0A                shr.b32 hi, $1, 16;\0A                shl.b32 lo, lo, 16;\0A                shl.b32 hi, hi, 16;\0A                mov.b32 f0, lo;\0A                mov.b32 f1, hi;\0A                max.f32 $0, f0, f1;\0A            }\0A            ", "=f,r" %34 : (i32) -> f32
            %c-1_i32 = arith.constant -1 : i32
            %c1_i32 = arith.constant 1 : i32
            %c31_i32 = arith.constant 31 : i32
            %36 = nvvm.shfl.sync  bfly %c-1_i32, %35, %c1_i32, %c31_i32 : f32 -> f32
            %37 = llvm.inline_asm asm_dialect = att "max.f32 $0, $1, $2;", "=f,f,f" %35, %36 : (f32, f32) -> f32
            %c-1_i32_26 = arith.constant -1 : i32
            %c2_i32 = arith.constant 2 : i32
            %c31_i32_27 = arith.constant 31 : i32
            %38 = nvvm.shfl.sync  bfly %c-1_i32_26, %37, %c2_i32, %c31_i32_27 : f32 -> f32
            %39 = llvm.inline_asm asm_dialect = att "max.f32 $0, $1, $2;", "=f,f,f" %37, %38 : (f32, f32) -> f32
            %cst = arith.constant 6.000000e+00 : f32
            %40 = llvm.inline_asm asm_dialect = att "rcp.approx.ftz.f32 $0, $1;", "=f,f" %cst : (f32) -> f32
            %41 = arith.mulf %39, %40 : f32
            %42 = llvm.inline_asm asm_dialect = att "\0A            {\0A                .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\0A                .reg .u32 bits, exp_biased, mantissa, bump, result;\0A\0A                setp.le.f32 p_zero, $1, 0f00000000;\0A\0A                mov.b32 bits, $1;\0A                shr.b32 exp_biased, bits, 23;\0A                and.b32 exp_biased, exp_biased, 255;\0A                and.b32 mantissa, bits, 0x7FFFFF;\0A\0A                setp.ne.u32 p_has_mant, mantissa, 0;\0A                selp.u32 bump, 1, 0, p_has_mant;\0A                setp.eq.u32 p_exp_zero, exp_biased, 0;\0A                setp.le.u32 p_tiny_sub, mantissa, 0x400000;\0A                and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\0A                @p_tiny_sub mov.u32 bump, 0;\0A                add.u32 result, exp_biased, bump;\0A\0A                setp.gt.u32 p_ovf, result, 254;\0A                selp.u32 result, 254, result, p_ovf;\0A                selp.u32 $0, 0, result, p_zero;\0A            }\0A            ", "=r,f" %41 : (f32) -> i32
            %43 = llvm.inline_asm asm_dialect = att "\0A            {\0A                .reg .s32 new_exp;\0A                .reg .b32 float_bits;\0A                .reg .pred p_zero;\0A\0A                setp.eq.u32 p_zero, $1, 0;\0A                sub.s32 new_exp, 254, $1;\0A                max.s32 new_exp, new_exp, 0;\0A                shl.b32 float_bits, new_exp, 23;\0A                mov.b32 $0, float_bits;\0A                @p_zero mov.b32 $0, 0;\0A            }\0A            ", "=f,r" %42 : (i32) -> f32
            %44 = llvm.inline_asm asm_dialect = att "\0A        {\0A            .reg .b32 lo, hi;\0A            .reg .f32 f0, f1;\0A            and.b32 lo, $2, 0xFFFF;\0A            shr.b32 hi, $2, 16;\0A            shl.b32 lo, lo, 16;\0A            shl.b32 hi, hi, 16;\0A            mov.b32 f0, lo;\0A            mov.b32 f1, hi;\0A            mul.f32 $0, f0, $3;\0A            mul.f32 $1, f1, $3;\0A        }\0A        ", "=f,=f,r,f" %24, %43 : (i32, f32) -> !llvm.struct<(f32, f32)>
            %45 = llvm.extractvalue %44[0] : !llvm.struct<(f32, f32)> 
            %46 = llvm.extractvalue %44[1] : !llvm.struct<(f32, f32)> 
            %47 = llvm.inline_asm asm_dialect = att "\0A        {\0A            .reg .b32 lo, hi;\0A            .reg .f32 f0, f1;\0A            and.b32 lo, $2, 0xFFFF;\0A            shr.b32 hi, $2, 16;\0A            shl.b32 lo, lo, 16;\0A            shl.b32 hi, hi, 16;\0A            mov.b32 f0, lo;\0A            mov.b32 f1, hi;\0A            mul.f32 $0, f0, $3;\0A            mul.f32 $1, f1, $3;\0A        }\0A        ", "=f,=f,r,f" %25, %43 : (i32, f32) -> !llvm.struct<(f32, f32)>
            %48 = llvm.extractvalue %47[0] : !llvm.struct<(f32, f32)> 
            %49 = llvm.extractvalue %47[1] : !llvm.struct<(f32, f32)> 
            %50 = llvm.inline_asm asm_dialect = att "\0A        {\0A            .reg .b32 lo, hi;\0A            .reg .f32 f0, f1;\0A            and.b32 lo, $2, 0xFFFF;\0A            shr.b32 hi, $2, 16;\0A            shl.b32 lo, lo, 16;\0A            shl.b32 hi, hi, 16;\0A            mov.b32 f0, lo;\0A            mov.b32 f1, hi;\0A            mul.f32 $0, f0, $3;\0A            mul.f32 $1, f1, $3;\0A        }\0A        ", "=f,=f,r,f" %26, %43 : (i32, f32) -> !llvm.struct<(f32, f32)>
            %51 = llvm.extractvalue %50[0] : !llvm.struct<(f32, f32)> 
            %52 = llvm.extractvalue %50[1] : !llvm.struct<(f32, f32)> 
            %53 = llvm.inline_asm asm_dialect = att "\0A        {\0A            .reg .b32 lo, hi;\0A            .reg .f32 f0, f1;\0A            and.b32 lo, $2, 0xFFFF;\0A            shr.b32 hi, $2, 16;\0A            shl.b32 lo, lo, 16;\0A            shl.b32 hi, hi, 16;\0A            mov.b32 f0, lo;\0A            mov.b32 f1, hi;\0A            mul.f32 $0, f0, $3;\0A            mul.f32 $1, f1, $3;\0A        }\0A        ", "=f,=f,r,f" %27, %43 : (i32, f32) -> !llvm.struct<(f32, f32)>
            %54 = llvm.extractvalue %53[0] : !llvm.struct<(f32, f32)> 
            %55 = llvm.extractvalue %53[1] : !llvm.struct<(f32, f32)> 
            %56 = llvm.inline_asm asm_dialect = att "\0A            {\0A                .reg .b8 byte0, byte1, byte2, byte3;\0A                cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;\0A                cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;\0A                cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;\0A                cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;\0A                mov.b32 $0, {byte0, byte1, byte2, byte3};\0A            }\0A            ", "=r,f,f,f,f,f,f,f,f" %45, %46, %48, %49, %51, %52, %54, %55 : (f32, f32, f32, f32, f32, f32, f32, f32) -> i32
            %coord_28 = cute.make_coord(%arg6) : (i32) -> !cute.coord<"(?,_)">
            %slice_29 = cute.slice(%arg1, %coord_28) : !memref_gmem_i8, !cute.coord<"(?,_)">
            %iter_30 = cute.get_iter(%slice_29) : !memref_gmem_i8_2
            %iter_31 = cute.get_iter(%slice_29) : !memref_gmem_i8_2
            %c16_i32 = arith.constant 16 : i32
            %57 = arith.muli %arg8, %c16_i32 : i32
            %c4_i32_32 = arith.constant 4 : i32
            %58 = arith.muli %10, %c4_i32_32 : i32
            %59 = arith.addi %57, %58 : i32
            %int_tuple_33 = cute.make_int_tuple(%59) : (i32) -> !cute.int_tuple<"?">
            %ptr_34 = cute.add_offset(%iter_31, %int_tuple_33) : (!cute.ptr<i8, gmem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<i8, gmem>
            %60 = builtin.unrealized_conversion_cast %ptr_34 : !cute.ptr<i8, gmem> to !llvm.ptr<1>
            %61 = llvm.ptrtoint %60 : !llvm.ptr<1> to i64
            llvm.inline_asm has_side_effects asm_dialect = att "st.global.u32 [$0], $1;", "l,r" %61, %56 : (i64, i32) -> ()
            %c0_i32 = arith.constant 0 : i32
            %62 = arith.cmpi eq, %10, %c0_i32 : i32
            %63:2 = scf.if %62 -> (!memref_gmem_i8_1, i32) {
              %iter_38 = cute.get_iter(%arg7) : !memref_gmem_i8_1
              %c4_i32_39 = arith.constant 4 : i32
              %65 = arith.remsi %arg8, %c4_i32_39 : i32
              %66 = arith.floordivsi %arg8, %c4_i32_39 : i32
              %c32_i32_40 = arith.constant 32 : i32
              %67 = arith.remsi %arg6, %c32_i32_40 : i32
              %c128_i32_41 = arith.constant 128 : i32
              %68 = arith.remsi %arg6, %c128_i32_41 : i32
              %69 = arith.floordivsi %68, %c32_i32_40 : i32
              %70 = arith.floordivsi %arg6, %c128_i32_41 : i32
              %c512_i32 = arith.constant 512 : i32
              %71 = arith.muli %66, %c512_i32 : i32
              %72 = arith.addi %65, %71 : i32
              %c16_i32_42 = arith.constant 16 : i32
              %73 = arith.muli %67, %c16_i32_42 : i32
              %74 = arith.addi %72, %73 : i32
              %75 = arith.muli %69, %c4_i32_39 : i32
              %76 = arith.addi %74, %75 : i32
              %c32768_i32 = arith.constant 32768 : i32
              %77 = arith.muli %70, %c32768_i32 : i32
              %78 = arith.addi %76, %77 : i32
              %79 = arith.trunci %42 : i32 to i8
              %coord_43 = cute.make_coord(%78) : (i32) -> !cute.coord<"?">
              cute.memref.store(%arg7, %coord_43, %79) : (!memref_gmem_i8_1, !cute.coord<"?">, i8) -> ()
              scf.yield %arg7, %42 : !memref_gmem_i8_1, i32
            } else {
              %iter_38 = cute.get_iter(%arg7) : !memref_gmem_i8_1
              scf.yield %arg7, %42 : !memref_gmem_i8_1, i32
            }
            %iter_35 = cute.get_iter(%63#0) : !memref_gmem_i8_1
            %iter_36 = cute.get_iter(%63#0) : !memref_gmem_i8_1
            %iter_37 = cute.get_iter(%63#0) : !memref_gmem_i8_1
            %c128_i32 = arith.constant 128 : i32
            %64 = arith.addi %arg8, %c128_i32 : i32
            scf.yield %63#0, %64 : !memref_gmem_i8_1, i32
          }
          %iter_16 = cute.get_iter(%15#0) : !memref_gmem_i8_1
          %iter_17 = cute.get_iter(%15#0) : !memref_gmem_i8_1
          %iter_18 = cute.get_iter(%15#0) : !memref_gmem_i8_1
          %c256_i32 = arith.constant 256 : i32
          %16 = arith.addi %9, %c256_i32 : i32
          %17:2 = scf.while (%arg7 = %15#0, %arg8 = %16) : (!memref_gmem_i8_1, i32) -> (!memref_gmem_i8_1, i32) {
            %iter_22 = cute.get_iter(%arg7) : !memref_gmem_i8_1
            %iter_23 = cute.get_iter(%arg7) : !memref_gmem_i8_1
            %c256_i32_24 = arith.constant 256 : i32
            %18 = arith.cmpi slt, %arg8, %c256_i32_24 : i32
            scf.condition(%18) %arg7, %arg8 : !memref_gmem_i8_1, i32
          } do {
          ^bb0(%arg7: !memref_gmem_i8_1, %arg8: i32):
            %iter_22 = cute.get_iter(%arg7) : !memref_gmem_i8_1
            %iter_23 = cute.get_iter(%arg7) : !memref_gmem_i8_1
            %c0_i32 = arith.constant 0 : i32
            %18 = arith.cmpi eq, %10, %c0_i32 : i32
            %19 = scf.if %18 -> (!memref_gmem_i8_1) {
              %iter_27 = cute.get_iter(%arg7) : !memref_gmem_i8_1
              %c4_i32_28 = arith.constant 4 : i32
              %21 = arith.remsi %arg8, %c4_i32_28 : i32
              %22 = arith.floordivsi %arg8, %c4_i32_28 : i32
              %c32_i32 = arith.constant 32 : i32
              %23 = arith.remsi %arg6, %c32_i32 : i32
              %c128_i32_29 = arith.constant 128 : i32
              %24 = arith.remsi %arg6, %c128_i32_29 : i32
              %25 = arith.floordivsi %24, %c32_i32 : i32
              %26 = arith.floordivsi %arg6, %c128_i32_29 : i32
              %c512_i32 = arith.constant 512 : i32
              %27 = arith.muli %22, %c512_i32 : i32
              %28 = arith.addi %21, %27 : i32
              %c16_i32 = arith.constant 16 : i32
              %29 = arith.muli %23, %c16_i32 : i32
              %30 = arith.addi %28, %29 : i32
              %31 = arith.muli %25, %c4_i32_28 : i32
              %32 = arith.addi %30, %31 : i32
              %c32768_i32 = arith.constant 32768 : i32
              %33 = arith.muli %26, %c32768_i32 : i32
              %34 = arith.addi %32, %33 : i32
              %c0_i8 = arith.constant 0 : i8
              %coord = cute.make_coord(%34) : (i32) -> !cute.coord<"?">
              cute.memref.store(%arg7, %coord, %c0_i8) : (!memref_gmem_i8_1, !cute.coord<"?">, i8) -> ()
              scf.yield %arg7 : !memref_gmem_i8_1
            } else {
              %iter_27 = cute.get_iter(%arg7) : !memref_gmem_i8_1
              scf.yield %arg7 : !memref_gmem_i8_1
            }
            %iter_24 = cute.get_iter(%19) : !memref_gmem_i8_1
            %iter_25 = cute.get_iter(%19) : !memref_gmem_i8_1
            %iter_26 = cute.get_iter(%19) : !memref_gmem_i8_1
            %c128_i32 = arith.constant 128 : i32
            %20 = arith.addi %arg8, %c128_i32 : i32
            scf.yield %19, %20 : !memref_gmem_i8_1, i32
          }
          %iter_19 = cute.get_iter(%17#0) : !memref_gmem_i8_1
          %iter_20 = cute.get_iter(%17#0) : !memref_gmem_i8_1
          %iter_21 = cute.get_iter(%17#0) : !memref_gmem_i8_1
          scf.yield %17#0 : !memref_gmem_i8_1
        }
        %iter_12 = cute.get_iter(%13) : !memref_gmem_i8_1
        %iter_13 = cute.get_iter(%13) : !memref_gmem_i8_1
        %iter_14 = cute.get_iter(%13) : !memref_gmem_i8_1
        %14 = arith.addi %arg6, %6 : i32
        scf.yield %13, %14 : !memref_gmem_i8_1, i32
      }
      %iter_7 = cute.get_iter(%11#0) : !memref_gmem_i8_1
      %iter_8 = cute.get_iter(%11#0) : !memref_gmem_i8_1
      %iter_9 = cute.get_iter(%11#0) : !memref_gmem_i8_1
      llvm.inline_asm has_side_effects asm_dialect = att "griddepcontrol.launch_dependents;", ""  : () -> ()
      return
    }
  }
  func.func @cutlass___call___flashinferquantizationkernelsmxfp4_quantizeMXFP4QuantizeSwizzledKernel_object_at__FakeTensorBFloat16div1819281921_FakeTensorUint8div1409640961_FakeTensorUint8div11(%arg0: !memref_gmem_bf16, %arg1: !memref_gmem_i8, %arg2: !memref_gmem_i8_1, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: !cuda.stream) -> i32 attributes {llvm.emit_c_interface} {
    %iter = cute.get_iter(%arg0) : !memref_gmem_bf16
    %iter_0 = cute.get_iter(%arg1) : !memref_gmem_i8
    %iter_1 = cute.get_iter(%arg2) : !memref_gmem_i8_1
    %iter_2 = cute.get_iter(%arg0) : !memref_gmem_bf16
    %iter_3 = cute.get_iter(%arg1) : !memref_gmem_i8
    %iter_4 = cute.get_iter(%arg2) : !memref_gmem_i8_1
    %lay = cute.get_layout(%arg0) : !memref_gmem_bf16
    %lay_5 = cute.get_layout(%arg1) : !memref_gmem_i8
    %lay_6 = cute.get_layout(%arg2) : !memref_gmem_i8_1
    %0 = cute.kernel_smem_size @kernels::@kernel_cutlass_kernel_flashinferquantizationkernelsmxfp4_quantizeMXFP4QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o819281921_tensorptri8gmemalign16o409640961_tensorptri8gmem_0 : i64
    %c232448_i64 = arith.constant 232448 : i64
    %1 = arith.cmpi sgt, %0, %c232448_i64 : i64
    scf.if %1 {
      cute.print("\0AError: kernel '@kernels::@kernel_cutlass_kernel_flashinferquantizationkernelsmxfp4_quantizeMXFP4QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o819281921_tensorptri8gmemalign16o409640961_tensorptri8gmem_0' launch shared memory exceeds current GPU arch sm_90a allowed. Allocated: {} bytes. Max: 232448 bytes.\0A\0A", %0) : i64
    }
    %c512_i32 = arith.constant 512 : i32
    %c1_i32 = arith.constant 1 : i32
    %2 = cuda.launch_cfg.create<max_attrs = 17 : i32> (blockDim = (%c512_i32, %c1_i32, %c1_i32), dynamicSmemBytes = %0, gridDim = (%arg5, %c1_i32, %c1_i32), stream = %arg6) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 17>
    cuda.launch_cfg.programmatic_stream_serialization_allowed[%2] %c1_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    %c0_i32 = arith.constant 0 : i32
    cuda.launch_cfg.cooperative[%2] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32
    %3 = cuda.launch_ex @kernels::@kernel_cutlass_kernel_flashinferquantizationkernelsmxfp4_quantizeMXFP4QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o819281921_tensorptri8gmemalign16o409640961_tensorptri8gmem_0<%2> (%arg0, %arg1, %arg2, %arg3, %arg4) {assume_kernel_attr = #cuda.assume_kernel_attr<true>} : !cuda.launch_cfg<max_attrs = 17>, (!memref_gmem_bf16, !memref_gmem_i8, !memref_gmem_i8_1, i32, i32) -> !cuda.result
    %4 = cuda.cast %3 : !cuda.result -> i32
    cuda.return_if_error %4 : i32
    %c0_i32_7 = arith.constant 0 : i32
    return %c0_i32_7 : i32
  }
}

