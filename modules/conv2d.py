from modules.layer import Layer
from modules.utils import *
#from cython_modules.im2col import im2col_forward_cython

import numpy as np
try:
    from cython_modules.im2col import im2col_forward_cython
    CYTHON_IM2COL_AVAILABLE = True
except ImportError:
    CYTHON_IM2COL_AVAILABLE = False

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, conv_algo=0, weight_init="he"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # MODIFICAR: Añadir nuevo if-else para otros algoritmos de convolución
        if conv_algo == 0:
            self.mode = 'direct'
        elif conv_algo == 1:
            self.mode = 'im2col' #Se añade nuevo modo
        elif conv_algo == 2:
            self.mode = 'im2col_cython' #Se añade nuevo modo
        elif conv_algo == 3:
            self.mode = 'im2col_blocked'    #Se añade un nuevo modo
        else:
            print(f"Algoritmo {conv_algo} no soportado aún")
            self.mode = 'direct'

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size

        if weight_init == "he":
            std = np.sqrt(2.0 / fan_in)
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "custom":
            self.kernels = np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
        else:
            self.kernels = np.random.uniform(-0.1, 0.1, 
                          (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        

        self.biases = np.zeros(out_channels, dtype=np.float32)

        # PISTA: Y estos valores para qué las podemos utilizar?
        # Si los usas, no olvides utilizar el modelo explicado en teoría que maximiza la caché
        self.mc = 480
        self.nc = 3072
        self.kc = 384
        self.mr = 32
        self.nr = 12
        self.Ac = np.empty((self.mc, self.kc), dtype=np.float32)
        self.Bc = np.empty((self.kc, self.nc), dtype=np.float32)
        self.kernels_col_cached = None  #Caché para kernels
        self.kernels_col_valid = False


    def get_weights(self):
        return {'kernels': self.kernels, 'biases': self.biases}

    def set_weights(self, weights):
        self.kernels = weights['kernels']
        self.biases = weights['biases']
        self.kernels_col_valid = False #Invalida caché

    #Se añade método auxiliar
    def _get_kernels_col(self):
        if (not self.kernels_col_valid) or (self.kernels_col_cached is None):
            self.kernels_col_cached = np.ascontiguousarray(
                self.kernels.reshape(self.out_channels, -1).T,
                dtype=np.float32
            )
            self.kernels_col_valid = True
        return self.kernels_col_cached

    
    def forward(self, input, training=True):
        self.input = input
        # PISTA: Usar estos if-else si implementas más algoritmos de convolución
        if self.mode == 'direct':
            return self._forward_direct(input)
        elif self.mode == 'im2col':
            return self._forward_im2col(input)
        elif self.mode == 'im2col_cython':
            return self._forward_im2col_cython(input)
        else:
            raise ValueError("Mode must be 'direct', 'im2col' or 'im2col_cython'")

    def backward(self, grad_output, learning_rate):
        # ESTO NO ES NECESARIO YA QUE NO VAIS A HACER BACKPROPAGATION
        if self.mode == 'direct':
            return self._backward_direct(grad_output, learning_rate)
        else:
            raise ValueError("Mode must be 'direct' or 'im2col'")

    # --- DIRECT IMPLEMENTATION ---
    def _forward_direct(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size
        stride = self.stride
        padding = self.padding

        if padding > 0:
            input = np.pad(
                input,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode='constant'
            ).astype(np.float32)

        out_h = (input.shape[2] - k_h) // stride + 1
        out_w = (input.shape[3] - k_w) // stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)

        kernels = self.kernels
        biases = self.biases

        for b in range(batch_size):
            input_b = input[b]
            for out_c in range(self.out_channels):
                kernel_oc = kernels[out_c]
                bias_oc = biases[out_c]
                output_oc = output[b, out_c]

                for i in range(out_h):
                    r = i * stride
                    for j in range(out_w):
                        c = j * stride
                        region = input_b[:, r:r + k_h, c:c + k_w]
                        output_oc[i, j] = np.sum(region * kernel_oc) + bias_oc

        return output


    # IM2COL IMPLEMENTATION, lo he creado apoyandome con la herramienta de ChatGPT
    def _im2col_numpy(self, input_padded):
        batch_size, channels, h, w = input_padded.shape
        k_h, k_w = self.kernel_size, self.kernel_size
        stride = self.stride

        out_h = (h - k_h) // stride + 1
        out_w = (w - k_w) // stride + 1

        windows = np.lib.stride_tricks.sliding_window_view(
            input_padded, (k_h, k_w), axis=(2, 3)
        )

        windows = windows[:, :, ::stride, ::stride, :, :]

        cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
            batch_size, out_h * out_w, channels * k_h * k_w
        )

        return np.ascontiguousarray(cols, dtype=np.float32)

    def _forward_im2col(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size
        stride = self.stride
        padding = self.padding

        if padding > 0:
            input_padded = np.pad(
                input,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode='constant'
            ).astype(np.float32)
        else:
            input_padded = np.ascontiguousarray(input, dtype=np.float32)

        out_h = (input_padded.shape[2] - k_h) // stride + 1
        out_w = (input_padded.shape[3] - k_w) // stride + 1

        # Input -> columnas
        cols = self._im2col_numpy(input_padded)   # (B, out_h*out_w, C*k*k)

        # Kernels -> matriz
        kernels_col = self._get_kernels_col() # (C*k*k, out_channels)

        # GEMM
        output = cols @ kernels_col  # Modificado
        output += self.biases  # (B, out_h*out_w, out_channels)

        # Reorganizar a formato NCHW
        output = output.reshape(batch_size, out_h, out_w, self.out_channels)
        output = output.transpose(0, 3, 1, 2) # Modificado

        return np.ascontiguousarray(output, dtype=np.float32) # Modificado

    def _forward_im2col_cython(self, input):  #Nuevo método
        if not CYTHON_IM2COL_AVAILABLE:
            return self._forward_im2col(input)

        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size
        stride = self.stride
        padding = self.padding

        if padding > 0:
            input_padded = np.pad(
                input,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode='constant'
            ).astype(np.float32)
        else:
            input_padded = np.ascontiguousarray(input, dtype=np.float32)

        out_h = (input_padded.shape[2] - k_h) // stride + 1
        out_w = (input_padded.shape[3] - k_w) // stride + 1

        cols = im2col_forward_cython(
            np.ascontiguousarray(input_padded, dtype=np.float32),
            self.kernel_size,
            self.stride
        )

        kernels_col = self._get_kernels_col()

        output = cols @ kernels_col
        output += self.biases

        output = output.reshape(batch_size, out_h, out_w, self.out_channels)
        output = output.transpose(0, 3, 1, 2)

        return np.ascontiguousarray(output, dtype=np.float32)

    
    def _backward_direct(self, grad_output, learning_rate):
        batch_size, _, out_h, out_w = grad_output.shape
        _, _, in_h, in_w = self.input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input_padded = np.pad(self.input,
                                  ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                  mode='constant').astype(np.float32)
        else:
            input_padded = self.input

        grad_input_padded = np.zeros_like(input_padded, dtype=np.float32)
        grad_kernels = np.zeros_like(self.kernels, dtype=np.float32)
        grad_biases = np.zeros_like(self.biases, dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            r = i * self.stride
                            c = j * self.stride
                            region = input_padded[b, in_c, r:r + k_h, c:c + k_w]
                            grad_kernels[out_c, in_c] += grad_output[b, out_c, i, j] * region
                            grad_input_padded[b, in_c, r:r + k_h, c:c + k_w] += self.kernels[out_c, in_c] * grad_output[b, out_c, i, j]
                grad_biases[out_c] += np.sum(grad_output[b, out_c])

        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        return grad_input

    # PISTA: Se te ocurren otros algoritmos de convolución?