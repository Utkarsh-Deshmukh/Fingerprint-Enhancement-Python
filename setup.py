"""generate the fingerprint_enhancer python package."""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fingerprint_enhancer",
    version="0.0.14",
    author="utkarsh-deshmukh",
    author_email="utkarsh.deshmukh@gmail.com",
    description="enhance fingerprint images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python",
    download_url="https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python/archive/develop.zip",
    install_requires=["numpy", "opencv-python", "scipy"],
    license="MIT",
    keywords="Fingerprint Image Enhancement",
    package_dir={"": "src"},
    packages=["fingerprint_enhancer"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
