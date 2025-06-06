from setuptools import setup, find_packages
import platform

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies that are essential and lightweight
CORE_DEPENDENCIES = [
    'numpy>=1.20.0',
    'pandas>=1.3.0',
    'scikit-learn>=1.0.0',
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
    'joblib>=1.1.0',
    'tqdm>=4.62.0',
]

# Optional dependencies for different use cases
EXTRAS_REQUIRE = {
    'full': [
        'dask>=2023.1.0',
        'plotly>=5.0.0',
        'xgboost>=1.5.0',
        'lightgbm>=3.3.0',
        'catboost>=1.0.0',
        'shap>=0.40.0',
        'optuna>=2.10.0',
    ],
    'viz': [
        'plotly>=5.0.0',
        'dash>=2.0.0',
    ],
    'boost': [
        'xgboost>=1.5.0',
        'lightgbm>=3.3.0',
    ],
    'deploy': [
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'onnx>=1.12.0',
    ],
    'deep': [
        'torch>=1.10.0',
        'tensorflow>=2.8.0',
    ],
}

# Add all dependencies to 'all'
EXTRAS_REQUIRE['all'] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    name="thinkml",
    version="1.1.0",
    author="Arpan Chaudhary",
    author_email="arpanchaudhary@gmail.com",
    description="End-to-End Machine Learning Workflow Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArpanChaudhary/ThinkML",
    project_urls={
        "Bug Tracker": "https://github.com/ArpanChaudhary/ThinkML/issues",
        "Documentation": "https://github.com/ArpanChaudhary/ThinkML#readme",
        "Source Code": "https://github.com/ArpanChaudhary/ThinkML",
    },
    packages=find_packages(),
    install_requires=CORE_DEPENDENCIES,
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.8',
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
    ],
    include_package_data=True,
    zip_safe=False,
) 