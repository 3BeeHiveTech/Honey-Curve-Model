"""Python setup.py for honey_curve package"""
import io
import os
from setuptools import find_packages, setup
from setuptools import setup
from Cython.Build import cythonize
from setuptools import setup, Extension, Command
import numpy as np


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("honey_curve", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]



# Specifica i file sorgente per i moduli per ciascuna architettura
module_sources_x86 = [
    "honey_curve/sklearn_light/_predictor_x86.pyx",
    "honey_curve/sklearn_light/_bitset_x86.pyx",
    "honey_curve/sklearn_light/common_x86.pyx",
]

module_sources_arm64 = [
    "honey_curve/sklearn_light/_predictor_arm64.pyx",
    "honey_curve/sklearn_light/_bitset_arm64.pyx",
    "honey_curve/sklearn_light/common_arm64.pyx",
]

setup(
    name="honey_curve",
    version=read("honey_curve", "VERSION.txt"),
    description="Honey Curve Model repo",
    url="https://github.com/3BeeHiveTech/Honey-Curve-Model/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Axel Dolcemascolo",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={"console_scripts": ["honey_curve = honey_curve.__main__:main"]},
    extras_require={"test": read_requirements("requirements-test.txt")},
    ext_modules=cythonize(
        [
            Extension(
                "honey_curve.sklearn_light._predictor",
                sources=module_sources_x86,
                extra_compile_args=['-arch', 'x86_64'],  # Specifica l'architettura x86_64
                include_dirs=[np.get_include()],
            ),
            Extension(
                "honey_curve.sklearn_light._bitset",
                sources=module_sources_x86,
                include_dirs=[np.get_include()],
            ),
            Extension(
                "honey_curve.sklearn_light.common",
                sources=module_sources_x86,
                include_dirs=[np.get_include()],
            ),
            Extension(
                "honey_curve.sklearn_light._predictor",
                sources=module_sources_arm64,
                extra_compile_args=['-arch', 'arm64'],  # Specifica l'architettura ARM64
                include_dirs=[np.get_include()],
            ),
            Extension(
                "honey_curve.sklearn_light._bitset",
                sources=module_sources_arm64,
                include_dirs=[np.get_include()],
            ),
            Extension(
                "honey_curve.sklearn_light.common",
                sources=module_sources_arm64,
                include_dirs=[np.get_include()],
            ),
        ],
        annotate=True,
    ),
    include_dirs=[np.get_include()],
    include_package_data=True,
)


