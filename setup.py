from setuptools import setup, find_packages
import sys

# Warning message to prevent accidental publishing
if 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
    print("\n" + "!"*80)
    print("WARNING: This package is intended to be installed directly from GitHub only.")
    print("It should not be published to PyPI or any other package repository.")
    print("The only official source is: https://github.com/ArpanChaudhary/ThinkML")
    print("!"*80 + "\n")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="thinkml",
    version="1.0.0",
    description="A comprehensive machine learning library that extends scikit-learn with advanced functionality (GitHub-only distribution)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Arpan Chaudhary",
    author_email="info@thinkml.org",
    url="https://github.com/ArpanChaudhary/ThinkML",
    project_urls={
        "Bug Tracker": "https://github.com/ArpanChaudhary/ThinkML/issues",
        "Documentation": "https://github.com/ArpanChaudhary/ThinkML#readme",
        "Source Code": "https://github.com/ArpanChaudhary/ThinkML",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "optuna>=2.10.0",
        "shap>=0.40.0",
        "lime>=0.2.0",
        "torch>=1.9.0",
        "joblib>=1.0.0",
        "tqdm>=4.60.0",
        "ipykernel>=6.0.0",
        "jupyter>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        "Private :: Do Not Upload",  # Prevents accidental upload to PyPI
    ],
    entry_points={
        "console_scripts": [
            "thinkml=thinkml.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "thinkml": ["helpers/*.py"],
    },
    data_files=[
        ("share/jupyter/nbextensions/thinkml", [
            "thinkml/helpers/notebook_helper.py",
        ]),
    ],
    zip_safe=False,
) 