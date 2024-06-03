import setuptools
from numpy.distutils.core import Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyDCM",
    version="0.1.0",
    author="Kai Toepfer",
    author_email="kai.toepfer@unibas.ch",
    description="<Template Setup.py package>",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ktoepfer/pydcm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

ext1 = Extension(name="dcm_fortran", sources=["dcm_fortran.F90"])

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(
        name="f2py_dcm",
        description="F2PY DCM module",
        author="Kai Toepfer",
        author_email="kai.toepfer@unibas.ch",
        ext_modules=[ext1],
    )
