from setuptools import setup, find_packages

setup(
    name="implicit_steerable_kernels",
    version="0.1",
    author="Maksim Zhdanov",
    author_email="maxxxzdn@gmail.com",
    description="code repository for the paper Implicit Convolutional Kernels for Steerable CNNs (NeurIPS 2023)",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/maxxxzdn/implicit-steerable-kernels",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
