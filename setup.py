"""
MLX Deterministic Inference Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlx-deterministic",
    version="0.1.0",
    author="Joshua",
    description="Batch-invariant operations for deterministic LLM inference on Apple Silicon using MLX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/mlx-deterministic",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "mlx>=0.29.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "lm": [
            "mlx-lm>=0.28.0",
        ],
    },
    keywords="mlx apple-silicon deterministic llm inference machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/YOUR_USERNAME/mlx-deterministic/issues",
        "Source": "https://github.com/YOUR_USERNAME/mlx-deterministic",
        "Documentation": "https://github.com/YOUR_USERNAME/mlx-deterministic/blob/main/INTEGRATION_GUIDE.md",
    },
)
