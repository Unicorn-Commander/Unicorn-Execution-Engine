"""
Unicorn Execution Engine - Setup Script
Multi-platform AI execution framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="unicorn-execution-engine",
    version="1.0.0",
    author="Magic Unicorn Unconventional Technology & Stuff Inc",
    author_email="info@magicunicorn.tech",
    description="Multi-platform AI execution framework with hardware acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Unicorn-Commander/Unicorn-Execution-Engine",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "onnxruntime>=1.17.0",
    ],
    extras_require={
        "intel-igpu": [
            "onnxruntime-openvino==1.17.0",
            "openvino>=2024.0.0",
        ],
        "tts": [
            "soundfile>=0.12.0",
            "scipy>=1.7.0",
        ],
        "all": [
            "onnxruntime-openvino==1.17.0",
            "openvino>=2024.0.0",
            "soundfile>=0.12.0",
            "scipy>=1.7.0",
        ],
    },
    package_data={
        "": ["*.onnx", "*.bin", "*.json"],
    },
    entry_points={
        "console_scripts": [
            "kokoro-tts=tts.kokoro_intel_igpu:main",
        ],
    },
)