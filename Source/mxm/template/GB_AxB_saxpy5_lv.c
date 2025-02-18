{
    const int64_t m = C->vlen;
    GB_Bp_DECLARE (Bp, const) ; GB_Bp_PTR (Bp, B) ;
    GB_Bh_DECLARE (Bh, const) ; GB_Bh_PTR (Bh, B) ;
    GB_Bi_DECLARE (Bi, const) ; GB_Bi_PTR (Bi, B) ;
    const bool B_iso = B->iso ;
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *)A->x;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *)B->x;
    size_t vl = VSETVL(m);
    GB_C_TYPE *restrict Cx = (GB_C_TYPE *)C->x;

#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
    for (int tid = 0; tid < ntasks; tid++)
    {
        const int64_t jB_start = B_slice[tid];
        const int64_t jB_end = B_slice[tid + 1];

        for (int64_t jB = jB_start; jB < jB_end; jB++)
        {
            const int64_t j = GBh_B (Bh, jB) ;
            GB_C_TYPE *restrict Cxj = Cx + (j * m) ;
            const int64_t pB_start = GB_IGET (Bp, jB) ;
            const int64_t pB_end   = GB_IGET (Bp, jB+1) ;
            for (int64_t i = 0; i < m && (m - i) >= vl; i += vl)
            {
                VECTORTYPE vc = VLE(Cxj + i, vl);
                for (int64_t pB = pB_start; pB < pB_end; pB++)
                {
                    const int64_t k = GB_IGET (Bi, pB) ;
                    GB_DECLAREB (bkj) ;
                    GB_GETB (bkj, Bx, pB, B_iso) ;
                    VECTORTYPE va = VLE(Ax + i + k * m, vl);
                    vc = VFMACC(vc, bkj, va, vl);
                }

                VSE(Cxj + i, vc, vl);
            }
            int64_t remaining = m % vl;
            if (remaining > 0)
            {
                int64_t i = m - remaining;
                VECTORTYPE vc = VLE(Cxj + i, remaining);
                for (int64_t pB = pB_start; pB < pB_end; pB++)
                {
                    const int64_t k = GB_IGET (Bi, pB) ;
                    GB_DECLAREB (bkj) ;
                    GB_GETB (bkj, Bx, pB, B_iso) ;
                    VECTORTYPE va = VLE(Ax + i + k * m, remaining);
                    vc = VFMACC(vc, bkj, va, remaining);
                }

                VSE(Cxj + i, vc, remaining);
            }
        }
    }
}
