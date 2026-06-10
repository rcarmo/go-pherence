.text
.option arch, +v, +xsmtvdotii
.global k3_i8i4_m1
.type k3_i8i4_m1, @function
# void k3_i8i4_m1(const uint8_t *a, const uint8_t *b, float *c, size_t k_blks, size_t nblks)
# a0=a, a1=b, a2=c, a3=k_blks, a4=nblks
k3_i8i4_m1:
    addi         sp, sp, -1216
    sd           s2, 0(sp)
    sd           s3, 8(sp)
    sd           s4, 16(sp)
    sd           s5, 24(sp)
    sd           s6, 32(sp)
    vsetvli      t0, x0, e8, m1
    addi         t1, sp, 48
    vse8.v       v1, (t1)
    addi         t1, sp, 176
    vse8.v       v24, (t1)
    addi         t1, sp, 304
    vse8.v       v25, (t1)
    addi         t1, sp, 432
    vse8.v       v26, (t1)
    addi         t1, sp, 560
    vse8.v       v27, (t1)
    addi         t1, sp, 688
    vse8.v       v28, (t1)
    addi         t1, sp, 816
    vse8.v       v29, (t1)
    addi         t1, sp, 944
    vse8.v       v30, (t1)
    addi         t1, sp, 1072
    vse8.v       v31, (t1)
    mv           t3, a3
    mv           t4, a4
    vsetvli      t0, x0, e16, m1
    vmv.v.i      v0, 1
    mv           s2, a0
    addi         s3, a0, 6
    mv           s4, a1
    addi         s5, a1, 96
    mv           s6, a2
    vsll.vi      v1, v0, 4
    vxor.vv      v2, v0, v0
    vfcvt.f.x.v  v0, v0
    vfcvt.f.x.v  v1, v1
.align 4
1:
    vsetvli      t0, x0, e8, m1
    vl4r.v       v4, (s5)
    addi         s5, s5, 608
    vsetvli      t0, x0, e8, mf2
    vle8.v       v30, (s4)
    addi         s4, s4, 64
    vsetvli      t0, x0, e8, mf4
    vle8.v       v3, (s3)
    addi         s3, s3, 38
    flw          f0, (s2)
    lh           t2, 4(s2)
    addi         s2, s2, 38
    vsetvli      t0, x0, e16, m1
    vmv.v.i      v28, 8
    vsetvli      t0, x0, e8, mf2
    vle8.v       v29, (s4)
    addi         s4, s4, 544
    vsetvli      t0, x0, e8, m1
    vsrl.vi      v24, v3, 4
    vsetvli      t0, x0, e16, m1
    vwaddu.vx    v28, v29, x0
    vmul.vx      v26, v28, t2
    vnpack4.vv   v8, v3, v3, 3
    vnpack4.vv   v10, v24, v24, 3
    vfcvt.f.x.v  v16, v26
    vadd.vi      v18, v16, 0
    vadd.vi      v20, v16, 0
    vadd.vi      v22, v16, 0
    vmadotsu.hp  v16, v10, v4, v1, 0, i4
    vmadotsu.hp  v18, v10, v5, v1, 0, i4
    vmadotsu.hp  v20, v10, v6, v1, 0, i4
    vmadotsu.hp  v22, v10, v7, v1, 0, i4
    vmadotu.hp   v16, v8, v4, v0, 0, i4
    vmadotu.hp   v18, v8, v5, v0, 0, i4
    vmadotu.hp   v20, v8, v6, v0, 0, i4
    vmadotu.hp   v22, v8, v7, v0, 0, i4
    vpack.vv     v24, v16, v18, 1
    vpack.vv     v26, v20, v22, 1
    vpack.vv     v16, v24, v26, 2
    vsetvli      t0, x0, e16, mf2
    vfwmul.vv    v31, v30, v16
    vsetvli      t0, x0, e32, m1
    vfmacc.vf    v2, f0, v31
    addi         t3, t3, -1
    bgtz         t3, 1b
    vsetvli      t0, t4, e32, m1
    vse32.v      v2, (s6)
    vsetvli      t0, x0, e8, m1
    addi         t1, sp, 48
    vle8.v       v1, (t1)
    addi         t1, sp, 176
    vle8.v       v24, (t1)
    addi         t1, sp, 304
    vle8.v       v25, (t1)
    addi         t1, sp, 432
    vle8.v       v26, (t1)
    addi         t1, sp, 560
    vle8.v       v27, (t1)
    addi         t1, sp, 688
    vle8.v       v28, (t1)
    addi         t1, sp, 816
    vle8.v       v29, (t1)
    addi         t1, sp, 944
    vle8.v       v30, (t1)
    addi         t1, sp, 1072
    vle8.v       v31, (t1)
    ld           s2, 0(sp)
    ld           s3, 8(sp)
    ld           s4, 16(sp)
    ld           s5, 24(sp)
    ld           s6, 32(sp)
    addi         sp, sp, 1216
    ret
.size k3_i8i4_m1, .-k3_i8i4_m1
