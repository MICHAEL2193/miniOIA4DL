# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def im2col_forward_cython(cnp.ndarray[cnp.float32_t, ndim=4] input_padded,
                          int kernel_size,
                          int stride):
    cdef int B = input_padded.shape[0]
    cdef int C = input_padded.shape[1]
    cdef int H = input_padded.shape[2]
    cdef int W = input_padded.shape[3]

    cdef int KH = kernel_size
    cdef int KW = kernel_size
    cdef int SH = stride
    cdef int SW = stride

    cdef int out_h = (H - KH) // SH + 1
    cdef int out_w = (W - KW) // SW + 1
    cdef int patch_size = C * KH * KW

    cdef cnp.ndarray[cnp.float32_t, ndim=3] cols = np.empty(
        (B, out_h * out_w, patch_size), dtype=np.float32
    )

    cdef int b, c, i, j, kh, kw
    cdef int h_start, w_start
    cdef int col_idx, idx

    for b in range(B):
        col_idx = 0
        for i in range(out_h):
            h_start = i * SH
            for j in range(out_w):
                w_start = j * SW
                idx = 0

                for c in range(C):
                    for kh in range(KH):
                        for kw in range(KW):
                            cols[b, col_idx, idx] = input_padded[b, c, h_start + kh, w_start + kw]
                            idx += 1

                col_idx += 1

    return cols
