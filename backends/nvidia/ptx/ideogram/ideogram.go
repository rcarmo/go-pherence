package ideogram

// IdeogramCFGStepPTX fuses asymmetric CFG and FlowMatch Euler update:
//
//	out[i] = latents[i] + sigma * (uncond[i] + guidance * (cond[i] - uncond[i]))
//
// It is a simple full-tensor vector kernel used by the Ideogram denoise loop
// once conditional and unconditional DiT velocities are available.
const IdeogramCFGStepPTX = `.version 7.0
.target sm_80
.address_size 64
.visible .entry ideogram_cfg_step_f32(
    .param .u64 LATENTS,
    .param .u64 COND,
    .param .u64 UNCOND,
    .param .u64 OUT,
    .param .f32 GUIDANCE,
    .param .f32 SIGMA,
    .param .u32 N
) {
    .reg .pred %p<2>;
    .reg .u32 %r<8>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<12>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    ld.param.u32 %r4, [N];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra done;

    ld.param.u64 %rd0, [LATENTS];
    ld.param.u64 %rd1, [COND];
    ld.param.u64 %rd2, [UNCOND];
    ld.param.u64 %rd3, [OUT];
    ld.param.f32 %f0, [GUIDANCE];
    ld.param.f32 %f1, [SIGMA];

    mul.wide.u32 %rd4, %r3, 4;
    add.u64 %rd5, %rd0, %rd4;
    add.u64 %rd6, %rd1, %rd4;
    add.u64 %rd7, %rd2, %rd4;
    add.u64 %rd8, %rd3, %rd4;

    ld.global.f32 %f2, [%rd5];      // latent
    ld.global.f32 %f3, [%rd6];      // cond
    ld.global.f32 %f4, [%rd7];      // uncond
    sub.f32 %f5, %f3, %f4;
    fma.rn.f32 %f6, %f0, %f5, %f4;  // guided
    fma.rn.f32 %f7, %f1, %f6, %f2;  // stepped
    st.global.f32 [%rd8], %f7;

done:
    ret;
}
`
