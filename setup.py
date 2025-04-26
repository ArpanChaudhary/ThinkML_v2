from setuptools import setup, find_packages

setup(
    name="thinkml",
    version="0.1.0",
    description="A comprehensive machine learning library that extends scikit-learn with advanced functionality",
    author="ThinkML Team",
    author_email="info@thinkml.org",
    packages=find_packages(),
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
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
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
) 