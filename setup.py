from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="drgnn",
    version="1.0.0",
    author="Salah Gamal Abdelkhabir",
    author_email="salahabdelkhabir@gmail.com",
    description="Interpretable Drug Repurposing Platform Based on Graph Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/salahabdelkhabir/drgnn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "dgl>=0.8.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
        "scipy>=1.7.0",
        "requests>=2.25.0",
    ],
)
