from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ttt",
    version="0.1.0",
    author="TTT Package",
    author_email="ttt@example.com",
    description="A PyTorch-based implementation of the Pair-Set Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/ttt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "tensorboard>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=21.0",
            "mypy>=0.800",
        ],
        "scripts": [
            "matplotlib>=3.3.0",
            "jupyter>=1.0.0",
            "pandas>=1.3.0",
            "pillow>=8.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="pytorch transformer attention set permutation-invariant machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/example/ttt/issues",
        "Source": "https://github.com/example/ttt",
        "Documentation": "https://github.com/example/ttt/blob/main/README.md",
    },
)