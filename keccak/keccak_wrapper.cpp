#include <Python.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern void keccak_kernel(uint64_t* input, uint64_t* output, int blocks);  // CUDA kernel

// Python 调用的接口
static PyObject* py_keccak(PyObject* self, PyObject* args) {
    Py_buffer input_buf;
    int blocks;

    if (!PyArg_ParseTuple(args, "y*i", &input_buf, &blocks)) {
        return NULL;
    }

    size_t total_size = input_buf.len;
    if (total_size != blocks * 25 * sizeof(uint64_t)) {
        PyErr_SetString(PyExc_ValueError, "Input size does not match blocks*25*8");
        return NULL;
    }

    // 分配 GPU 内存
    uint64_t* d_input;
    uint64_t* d_output;
    cudaMalloc(&d_input, total_size);
    cudaMalloc(&d_output, total_size);

    // 复制输入
    cudaMemcpy(d_input, input_buf.buf, total_size, cudaMemcpyHostToDevice);

    // 启动 kernel
    int threads = 256;
    int grid = (blocks + threads - 1) / threads;
    keccak_kernel<<<grid, threads>>>(d_input, d_output, blocks);
    cudaDeviceSynchronize();

    // 分配输出缓冲
    PyObject* result = PyBytes_FromStringAndSize(NULL, total_size);
    void* output_buf = PyBytes_AsString(result);
    cudaMemcpy(output_buf, d_output, total_size, cudaMemcpyDeviceToHost);

    // 释放资源
    cudaFree(d_input);
    cudaFree(d_output);
    PyBuffer_Release(&input_buf);

    return result;
}

static PyMethodDef KeccakMethods[] = {
    {"cuda_keccak", py_keccak, METH_VARARGS, "Run Keccak on GPU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef keccakmodule = {
    PyModuleDef_HEAD_INIT,
    "keccak_cuda",
    NULL,
    -1,
    KeccakMethods
};

PyMODINIT_FUNC PyInit_keccak_cuda(void) {
    return PyModule_Create(&keccakmodule);
}