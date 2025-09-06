from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="doxa",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular, intelligent ML/DL library with automatic device management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/doxa",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "gpu": ["cupy-cuda11x>=9.0.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme",
        ],
    },
)