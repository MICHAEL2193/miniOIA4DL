# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def maxpool_forward_cython(cnp.ndarray[cnp.float32_t, ndim=4] input,
                           int kernel_size,
                           int stride):
    cdef int B = input.shape[0]
    cdef int C = input.shape[1]
    cdef int H = input.shape[2]
    cdef int W = input.shape[3]

    cdef int KH = kernel_size
    cdef int KW = kernel_size
    cdef int SH = stride
    cdef int SW = stride

    cdef int out_h = (H - KH) // SH + 1
    cdef int out_w = (W - KW) // SW + 1

    cdef cnp.ndarray[cnp.float32_t, ndim=4] output = np.empty((B, C, out_h, out_w), dtype=np.float32)
    cdef cnp.ndarray[cnp.int64_t, ndim=5] max_indices = np.empty((B, C, out_h, out_w, 2), dtype=np.int64)

    cdef int b, c, i, j, kh, kw
    cdef int h_start, w_start
    cdef int max_h, max_w
    cdef float max_val, val

    for b in range(B):
        for c in range(C):
            for i in range(out_h):
                h_start = i * SH
                for j in range(out_w):
                    w_start = j * SW

                    max_val = input[b, c, h_start, w_start]
                    max_h = h_start
                    max_w = w_start

                    for kh in range(KH):
                        for kw in range(KW):
                            val = input[b, c, h_start + kh, w_start + kw]
                            if val > max_val:
                                max_val = val
                                max_h = h_start + kh
                                max_w = w_start + kw

                    output[b, c, i, j] = max_val
                    max_indices[b, c, i, j, 0] = max_h
                    max_indices[b, c, i, j, 1] = max_w

    return output, max_indices
