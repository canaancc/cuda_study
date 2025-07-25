from setuptools import setup
from setuptools.extension import Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="keccak_cuda",
    ext_modules=[
        CUDAExtension(
            "keccak_cuda",
            ["keccak_wrapper.cpp", "keccak_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)