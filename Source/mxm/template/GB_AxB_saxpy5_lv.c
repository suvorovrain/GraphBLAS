{

    //--------------------------------------------------------------------------
    // get C, A, and B
    //--------------------------------------------------------------------------

    const int64_t m = C->vlen; // # of rows of C and A
    const int64_t *restrict Bp = B->p;
    const int64_t *restrict Bh = B->h;
    const int64_t *restrict Bi = B->i;
#ifdef GB_JIT_KERNEL
#define B_iso GB_B_ISO
#else
    const bool B_iso = B->iso;
#endif
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *)A->x;
#if !GB_B_IS_PATTERN
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *)B->x;
#endif
    GB_C_TYPE *restrict Cx = (GB_C_TYPE *)C->x;
    int tid;
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
    for (tid = 0; tid < ntasks; tid++)
    {
        // get the task descriptor
        const int64_t jB_start = B_slice[tid];
        const int64_t jB_end = B_slice[tid + 1];

        // C(:,jB_start:jB_end-1) += A * B(:,jB_start:jB_end-1)
        for (int64_t jB = jB_start; jB < jB_end; jB++)
        {
            // get B(:,j) and C(:,j)
            const int64_t j = GBH_B(Bh, jB);
            GB_C_TYPE *restrict Cxj = Cx + (j * m);
            const int64_t pB_start = Bp[jB];
            const int64_t pB_end = Bp[jB + 1];

            size_t vl = __riscv_vsetvl_e64m8(m);

            vfloat64m8_t vc = __riscv_vle64_v_f64m8(Cxj, vl);
            for (int64_t pB = pB_start; pB < pB_end; pB++)
            {
                const int64_t k = Bi[pB];
                GB_DECLAREB(bkj);
                GB_GETB(bkj, Bx, pB, B_iso);
                // const GB_B_TYPE bkj = Bx[pB];
                vfloat64m8_t va = __riscv_vle64_v_f64m8(Ax + k * m, vl);
                vc = __riscv_vfmacc_vf_f64m8(vc, bkj, va, vl);
            }

            __riscv_vse64_v_f64m8(Cxj, vc, vl);
        }
    }
}
