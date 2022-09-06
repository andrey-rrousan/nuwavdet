import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wavsource_nustar",
    version="1.0.0",
    author="Andrey_Mukhin",
    author_email="amukhin@phystech.edu",
    description="A package for source exclusion in NuStar observation data using wavelet decomposition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Andreyousan/wavsource_nustar",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)