import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nuwavdet",
    version="0.1.0",
    author="Andrey Mukhin",
    author_email="amukhin@phystech.edu",
    description="A package for source exclusion in NuStar observation data using wavelet decomposition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andrey-rrousan/nuwavdet",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires = [
        'astropy==5.1',
        'numpy==1.23.2',
        'pandas==1.4.4',
        'scipy==1.9.1',
        'setuptools==70.0.0',
    ]
)
