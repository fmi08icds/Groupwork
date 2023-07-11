import os
import shutil
import subprocess
import sys
from distutils.cmd import Command
from runpy import run_path

from setuptools import find_packages, setup

# read the program version from version.py (without loading the module)
__version__ = run_path("src/regression_comparison/version.py")["__version__"]


def read(fname):
    """Utility function to read the README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="regression-comparison",
    version=__version__,
    author="Jane Doe",
    author_email="",
    description="A comparison of different regression models",
    license="proprietary",
    url="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"regression_comparison": ["res/*"]},
    long_description=read("README.md"),
    install_requires=[],
    tests_require=[
        "pytest",
        "pytest-cov",
        "pre-commit",
    ],
    platforms="any",
    python_requires=">=3.7",
)
