"""
Simplified setup without C++ extensions
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="unicorn-engine",
    version="1.0.0",
    author="Magic Unicorn Unconventional Technology & Stuff Inc.",
    author_email="hello@magicunicorn.tech",
    description="Hardware-accelerated AI inference with 220x speedup",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Unicorn-Commander/Unicorn-Execution-Engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "huggingface-hub>=0.19.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "npu": ["pyxrt>=2.0.0"],
        "gpu": ["torch>=2.0.0"],
        "dev": ["pytest>=7.4.0", "black>=23.0.0"],
    },
    entry_points={
        "console_scripts": [
            "unicorn-engine=unicorn_engine.cli:main",
        ],
    },
)