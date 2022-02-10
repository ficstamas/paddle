import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setuptools.setup(
    name="paddle",
    version="0.0.1",
    description="Collection of Utility functions for myself to ease the construction of deep learning projects.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Tamás Ficsor",
    author_email="ficsort@inf.u-szeged.hu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.9"
)