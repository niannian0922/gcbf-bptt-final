from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='dynamics_cuda',
    ext_modules=[
        CUDAExtension(
            name='dynamics_cuda',
            sources=[
                'gcbfplus/cuda/dynamics_wrapper.cpp',
                'gcbfplus/cuda/dynamics_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-Wno-deprecated-declarations'],
                'nvcc': ['-O3', '-Xcompiler', '-fPIC', '-allow-unsupported-compiler']
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

